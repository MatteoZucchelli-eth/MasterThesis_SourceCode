# MIT License
"""Publishes the wall plane fitted from the head D435 depth stream.

Consumed by VLA_PLANNER (``vla_planner_depth_source: stereo``), which
intersects the monocular detector's ray with this plane to get the range.
A plane fitted from the whole visible wall is robust to the arm occluding
the target patch. Each plane is a single-frame fit; no smoothing here.
"""

from __future__ import annotations

import numpy as np
import rclpy
from g1_common.camera import PinholeIntrinsics
from g1_common.depth import depth_scale
from g1_definitions import topics
from g1_interfaces.msg import WallPlane
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from .wall_plane import WallEstimatorConfig, WallPlaneEstimator

#: Latest frame only, BEST_EFFORT (compatible with sensor publishers).
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class WallPlaneNode(Node):
    """Depth image -> wall plane, plus the cloud and marker to see it with."""

    def __init__(self) -> None:
        super().__init__("wall_plane")

        self._depth_topic = self._param("depth_topic", topics.TOPIC_CAMERA_DEPTH)
        self._info_topic = self._param(
            "camera_info_topic", topics.TOPIC_CAMERA_DEPTH_INFO
        )
        # Fit/publish frame; pelvis is wall-rigid while the base is locked.
        self._reference_frame = self._param("reference_frame", "pelvis")
        self._publish_cloud = bool(self._param("publish_cloud", True))
        self._estimator = WallPlaneEstimator(self._estimator_config())

        self._intrinsics: PinholeIntrinsics | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._plane_pub = self.create_publisher(
            WallPlane, topics.TOPIC_WALL_PLANE, SENSOR_QOS
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, topics.TOPIC_WALL_POINTS, SENSOR_QOS
        )
        self._marker_pub = self.create_publisher(
            Marker, topics.TOPIC_WALL_PLANE_MARKER, 1
        )
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._camera_info_callback, SENSOR_QOS
        )
        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._depth_callback, SENSOR_QOS
        )

        # Counted and reported every 10 s instead of logged per frame.
        self._fits = 0
        self._failures = 0
        self._last_reason = ""
        self._warned_latest_tf = False
        self._report_timer = self.create_timer(10.0, self._report)

        cfg = self._estimator.config
        self.get_logger().info(
            f"wall_plane: '{self._depth_topic}' + '{self._info_topic}' -> "
            f"'{topics.TOPIC_WALL_PLANE}' in '{self._reference_frame}'; "
            f"every {cfg.stride}th pixel between {cfg.min_depth_m:.2f} and "
            f"{cfg.max_depth_m:.2f} m, planes within {cfg.max_tilt_deg:.0f} deg "
            f"of vertical with >= {cfg.min_inliers} inliers at "
            f"{cfg.tolerance_m * 1e3:.0f} mm"
            + (
                f", every reading biased {cfg.depth_bias_m * 1e3:+.1f} mm."
                if cfg.depth_bias_m
                else "."
            )
        )

    # ── Setup helpers ────────────────────────────────────────────────────────

    def _param(self, name: str, default):
        """Declare-and-read, so every default above is also a ROS parameter."""
        return self.declare_parameter(name, default).value

    def _estimator_config(self) -> WallEstimatorConfig:
        """Every field of the estimator's config, exposed as a parameter."""
        defaults = WallEstimatorConfig()
        up = self._param("up_axis", list(defaults.up_axis))
        return WallEstimatorConfig(
            stride=int(self._param("stride", defaults.stride)),
            min_depth_m=float(self._param("min_depth_m", defaults.min_depth_m)),
            max_depth_m=float(self._param("max_depth_m", defaults.max_depth_m)),
            tolerance_m=float(self._param("tolerance_m", defaults.tolerance_m)),
            iterations=int(self._param("iterations", defaults.iterations)),
            max_tilt_deg=float(self._param("max_tilt_deg", defaults.max_tilt_deg)),
            min_inliers=int(self._param("min_inliers", defaults.min_inliers)),
            up_axis=(float(up[0]), float(up[1]), float(up[2])),
            depth_bias_m=float(self._param("depth_bias_m", defaults.depth_bias_m)),
        )

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Adopt the depth stream's intrinsics (size-checked at deprojection)."""
        intrinsics = PinholeIntrinsics.from_k(msg.k, msg.width, msg.height)
        if intrinsics != self._intrinsics:
            self.get_logger().info(
                f"wall_plane: deprojecting with fx={intrinsics.fx:.3f} "
                f"fy={intrinsics.fy:.3f} cx={intrinsics.cx:.3f} "
                f"cy={intrinsics.cy:.3f} at {intrinsics.width}x"
                f"{intrinsics.height}, from '{self._info_topic}'."
            )
        self._intrinsics = intrinsics

    def _depth_callback(self, msg: Image) -> None:
        """Fit and publish the wall from one depth frame."""
        if self._intrinsics is None:
            self._fail("no CameraInfo on the depth stream yet")
            return
        located = self._lookup(msg)
        if located is None:
            return
        transform, stamp = located

        # Malformed frames (e.g. mismatched CameraInfo) must not kill the node.
        try:
            estimate = self._estimator.estimate(
                _decode_depth(msg), self._intrinsics, transform
            )
        except ValueError as exc:
            self._fail(str(exc))
            return
        if self._publish_cloud:
            # Published even when the fit fails, for debugging in RViz.
            self._cloud_pub.publish(
                point_cloud2.create_cloud_xyz32(
                    _header(stamp, self._reference_frame),
                    estimate.points.astype(np.float32),
                )
            )
        if estimate.plane is None:
            self._fail(estimate.reason)
            return

        self._fits += 1
        self._plane_pub.publish(
            _plane_message(estimate.plane, stamp, self._reference_frame)
        )
        self._marker_pub.publish(
            _plane_marker(estimate.plane, stamp, self._reference_frame)
        )

    def _lookup(self, msg: Image) -> tuple[np.ndarray, object] | None:
        """``(T(reference <- depth optical), its stamp)`` for this frame, or None.

        Looks up at the image stamp (waist motion tilts the camera), falling back
        to the latest transform on failure (typically an image/TF clock-domain
        mismatch; warned once). Outputs are stamped with the transform's stamp
        so consumers can resolve them.
        """
        source = msg.header.frame_id
        try:
            tf = self._tf_buffer.lookup_transform(
                self._reference_frame, source, msg.header.stamp
            )
            return _transform_matrix(tf), tf.header.stamp
        except TransformException as stamped_exc:
            reason = str(stamped_exc)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._reference_frame, source, rclpy.time.Time()
            )
        except TransformException as exc:
            self._fail(f"no {source} -> {self._reference_frame} TF ({exc})")
            return None
        if not self._warned_latest_tf:
            self._warned_latest_tf = True
            self.get_logger().warn(
                f"wall_plane: no {source} -> {self._reference_frame} transform "
                f"at the depth frame's own stamp ({reason}); using the latest "
                "available instead. Usually a clock mismatch — the image is "
                "stamped from one clock and TF from another. Set use_sim_time "
                "consistently to get the exact-stamp lookup back. Warned once."
            )
        return _transform_matrix(tf), tf.header.stamp

    def _fail(self, reason: str) -> None:
        """Count a frame that produced no plane, and remember why."""
        self._failures += 1
        self._last_reason = reason

    def _report(self) -> None:
        """Warn about missing frames or failed fits over the last 10 s."""
        if not self._fits and not self._failures:
            self.get_logger().warn(
                f"wall_plane: no depth frames on '{self._depth_topic}' in the "
                "last 10 s. In sim, check G1_HEAD_DEPTH_TOPIC is not set to ''; "
                "on the robot, enable_depth in docker/head_camera.yaml."
            )
        elif self._failures:
            self.get_logger().warn(
                f"wall_plane: {self._fits} planes, {self._failures} frames "
                f"without one in the last 10 s — last: {self._last_reason}"
            )
        # Silent when healthy.
        self._fits = 0
        self._failures = 0


# ── Message plumbing ────────────────────────────────────────────────────────


def _decode_depth(msg: Image) -> np.ndarray:
    """A depth Image as a (height, width) float array of metres.

    Honours row ``step`` padding; refuses big-endian frames. Scales come from
    ``g1_common.depth.DEPTH_SCALES``.
    """
    if msg.is_bigendian:
        raise ValueError("big-endian depth images are not supported")
    scale = depth_scale(msg.encoding)
    dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
    itemsize = np.dtype(dtype).itemsize
    if msg.step % itemsize:
        raise ValueError(f"depth row step {msg.step} is not a whole number of pixels")
    flat = np.frombuffer(msg.data, dtype=dtype)
    depth = flat.reshape(msg.height, msg.step // itemsize)[:, : msg.width]
    return depth.astype(float) * scale


def _transform_matrix(tf) -> np.ndarray:
    """4x4 rigid transform from a geometry_msgs/TransformStamped."""
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm < 1e-12:
        raise ValueError("transform carries a zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _header(stamp, frame_id: str) -> Header:
    """A header in the plane's frame, stamped with its transform's stamp."""
    return Header(stamp=stamp, frame_id=frame_id)


def _plane_message(plane, stamp, frame_id: str) -> WallPlane:
    return WallPlane(
        header=_header(stamp, frame_id),
        point=Point(
            x=float(plane.point[0]), y=float(plane.point[1]), z=float(plane.point[2])
        ),
        normal=Vector3(
            x=float(plane.normal[0]), y=float(plane.normal[1]), z=float(plane.normal[2])
        ),
        inliers=int(plane.inliers),
        sampled=int(plane.sampled),
        rms_m=float(plane.rms_m),
    )


def _plane_marker(plane, stamp, frame_id: str) -> Marker:
    """A thin disc on the fitted plane at its centroid, for RViz (extent is nominal)."""
    marker = Marker()
    marker.header = _header(stamp, frame_id)
    marker.ns = "wall_plane"
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    marker.pose.position.x = float(plane.point[0])
    marker.pose.position.y = float(plane.point[1])
    marker.pose.position.z = float(plane.point[2])
    # CYLINDER axis is local +z.
    marker.pose.orientation = Quaternion(**_align_z_to(plane.normal))
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 0.004
    marker.color.a = 0.35
    marker.color.r = 0.2
    marker.color.g = 0.8
    marker.color.b = 1.0
    return marker


def _align_z_to(normal: np.ndarray) -> dict:
    """Shortest-arc quaternion (x/y/z/w kwargs) rotating +z onto ``normal``.

    The antipodal case (``normal`` = -z) returns a fixed half-turn about x.
    """
    z = np.array([0.0, 0.0, 1.0])
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / float(np.linalg.norm(n))
    half = z + n
    length = float(np.linalg.norm(half))
    if length < 1e-9:
        return {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}
    half = half / length
    axis = np.cross(z, half)
    return {
        "x": float(axis[0]),
        "y": float(axis[1]),
        "z": float(axis[2]),
        "w": float(z @ half),
    }


def main() -> None:
    rclpy.init()
    node = WallPlaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
