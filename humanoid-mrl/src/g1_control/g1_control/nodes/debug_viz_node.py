# MIT License
# RViz debug visualization node for the G1 controller.

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from g1_common.camera import preparer_from_camera_info
from g1_common.image_transport import subscribe_image, transport_name
from g1_definitions import definitions
from g1_interfaces.msg import ControlStep
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray, UInt8
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from g1_control import topics
from g1_control.utils import broadcast_se3
from g1_control.utils.wbc_robot_model import RobotModel
from g1_control.vla.perception import backproject_uv

# Parent frame for all debug poses, matching the controller's convention.
_FRAME = "pelvis"

# Per-FSM-state child frame names for the commanded eef targets.
# Right is the drill tip, not the palm (definitions.RIGHT_EEF_FRAME).
_TARGET_FRAMES = {
    definitions.FSMState.VLA.value: ("vla_target_left_palm", "vla_target_right_tip"),
    definitions.FSMState.TELEOP.value: (
        "teleop_target_left_palm",
        "teleop_target_right_tip",
    ),
}


class DebugVizNode(Node):
    """Renders the controller's debug data for RViz.

    Subscribes to the recorder snapshot and predicted-chunk topics and publishes
    TF frames, nav_msgs/Path trajectories and a JPEG perception overlay on
    ``perception/target_overlay/compressed``. Not required for control.
    """

    def __init__(self):
        super().__init__("debug_viz_node")

        self._fsm_state = definitions.FSMState.DAMPING.value
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            UInt8, definitions.TOPIC_FSMSTATE, self._fsm_state_callback, 1
        )
        self.create_subscription(
            ControlStep,
            definitions.TOPIC_RECORDED_STEP,
            self._recorded_step_callback,
            10,
        )

        # Predicted-chunk paths: each raw 9D chunk topic maps to a Path topic.
        chunk_to_path = {
            topics.TOPIC_VLA_CHUNK_RIGHT: "right_palm_path",
            topics.TOPIC_VLA_CHUNK_RIGHT_PREV: "right_palm_path_prev",
        }
        self._path_pubs = {
            name: self.create_publisher(Path, f"vla_vis/{name}", 1)
            for name in chunk_to_path.values()
        }
        for chunk_topic, path_name in chunk_to_path.items():
            self.create_subscription(
                Float32MultiArray,
                chunk_topic,
                self._make_chunk_callback(path_name),
                1,
            )

        # VLA's red-target estimate, broadcast as a drill-tip child frame.
        self.create_subscription(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET, self._tip_target_callback, 1
        )
        self.create_subscription(
            Float32MultiArray,
            topics.TOPIC_VLA_TIP_TARGET_DRIVER,
            self._tip_target_driver_callback,
            1,
        )
        # Filtered/batch estimate (pelvis frame): servo EMA in VLA, accumulator
        # mean the finalize was planned on in VLA_PLANNER.
        self.create_subscription(
            Float32MultiArray,
            topics.TOPIC_VLA_TIP_TARGET_FUSED,
            self._tip_target_fused_callback,
            1,
        )
        # VLA_PLANNER finalize: `vla_tip_target_fused` is where the plan aimed,
        # `vla_tip_target_live` where it aims now, `vla_tip_target_live_fine_only`
        # where it would aim on the detector's own depth.
        self.create_subscription(
            Float32MultiArray,
            topics.TOPIC_VLA_TIP_TARGET_LIVE,
            self._tip_target_live_callback,
            1,
        )
        self.create_subscription(
            Float32MultiArray,
            topics.TOPIC_VLA_TIP_TARGET_LIVE_FINE_ONLY,
            self._tip_target_live_fine_only_callback,
            1,
        )

        # 2D perception overlay (VLA_PLANNER); same camera QoS as the controller.
        self._cv_bridge = CvBridge()
        self._latest_wrist: np.ndarray | None = None
        # Canvas is prepared like the model input (square centre crop +
        # rectification from CameraInfo), since `target_uv_px` is in those pixels.
        self._wrist_preparer = None
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Compressed by default (~55 KB vs 2.76 MB/frame raw). Overridable: in
        # sim the Isaac bridge publishes raw on the same host, so full_launch.py
        # passes the bare topic. The message type is inferred from the topic name.
        self.declare_parameter(
            "wrist_image_topic", definitions.TOPIC_CAMERA_WRIST + "/compressed"
        )
        wrist_topic = self.get_parameter("wrist_image_topic").value
        subscribe_image(
            self,
            wrist_topic,
            self._wrist_image_callback,
            cam_qos,
            bridge=self._cv_bridge,
        )
        self.get_logger().info(
            f"perception overlay canvas: '{wrist_topic}' "
            f"[{transport_name(wrist_topic)}]"
        )
        self.create_subscription(
            CameraInfo,
            definitions.TOPIC_CAMERA_WRIST_INFO,
            self._camera_info_callback,
            cam_qos,
        )
        # Viewer-only overlay: JPEG on '<base>/compressed' with sensor QoS, which
        # an RViz2 Image display on that topic subscribes to via compressed transport.
        self._overlay_pub = self.create_publisher(
            CompressedImage,
            topics.TOPIC_PERCEPTION_TARGET_OVERLAY + "/compressed",
            cam_qos,
        )
        self.create_subscription(
            Float32MultiArray,
            topics.TOPIC_PERCEPTION_DETECTION,
            self._perception_detection_callback,
            1,
        )

        # CoM markers: one sphere per URDF link at its inertial origin, in the
        # link frame. Sphere volume ~ mass.
        default_urdf = os.path.join(
            get_package_share_directory("g1_state_publisher"),
            "urdf",
            "g1_29_dof_with_hand_drill.urdf.xml",
        )
        self.declare_parameter("urdf_path", default_urdf)
        self._com_links = self._parse_urdf_coms(self.get_parameter("urdf_path").value)
        self._com_pub = self.create_publisher(MarkerArray, "debug/com_markers", 1)
        self.create_timer(0.5, self._publish_com_markers)  # 2 Hz

        self.get_logger().info("Debug Viz Node initialized")

    @staticmethod
    def _parse_urdf_coms(urdf_path: str) -> list[tuple[str, float, tuple]]:
        """Return [(link_name, mass, (cx, cy, cz)), ...] for links with inertia."""
        out: list[tuple[str, float, tuple]] = []
        try:
            root = ET.parse(urdf_path).getroot()
        except (OSError, ET.ParseError):
            return out
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass_el = inertial.find("mass")
            mass = float(mass_el.get("value")) if mass_el is not None else 0.0
            if mass <= 0.0:
                continue
            origin = inertial.find("origin")
            xyz = origin.get("xyz") if origin is not None else "0 0 0"
            com = tuple(float(v) for v in xyz.split())
            out.append((link.get("name"), mass, com))
        return out

    def _publish_com_markers(self) -> None:
        """Publish per-link CoM spheres (+ mass labels) for RViz."""
        if not self._com_links:
            return
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for i, (link, mass, com) in enumerate(self._com_links):
            dia = 0.08 * (mass ** (1.0 / 3.0))  # sphere volume ~ mass
            sphere = Marker()
            sphere.header.frame_id = link
            sphere.header.stamp = stamp
            sphere.ns = "com"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = com
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = dia
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (
                1.0,
                0.3,
                0.0,
                0.85,
            )
            arr.markers.append(sphere)

            label = Marker()
            label.header.frame_id = link
            label.header.stamp = stamp
            label.ns = "com_label"
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = com[0], com[1]
            label.pose.position.z = com[2] + dia
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{mass:.3f}kg"
            arr.markers.append(label)
        self._com_pub.publish(arr)

    def _fsm_state_callback(self, msg: UInt8) -> None:
        self._fsm_state = msg.data

    def _recorded_step_callback(self, msg: ControlStep) -> None:
        """Broadcast the commanded eef targets as TF frames for the active state."""
        frames = _TARGET_FRAMES.get(self._fsm_state)
        if frames is None:
            return
        left_frame, right_frame = frames
        self._broadcast_pose(np.array(msg.left_eef_target), left_frame)
        self._broadcast_pose(np.array(msg.right_eef_target), right_frame)

    def _broadcast_pose(
        self, pose_9d: np.ndarray, child_frame: str, parent_frame: str = _FRAME
    ) -> None:
        broadcast_se3(
            self._tf_broadcaster,
            RobotModel.action_9d_to_se3(pose_9d),
            parent_frame,
            child_frame,
            self.get_clock(),
        )

    def _tip_target_callback(self, msg: Float32MultiArray) -> None:
        """Broadcast the fine role's gated, lifted target as a drill-tip child frame."""
        if not msg.data:
            return  # no prediction yet this session
        pose_9d = np.asarray(msg.data, dtype=np.float32)
        self._broadcast_pose(pose_9d, "vla_tip_target", parent_frame="drill_tip_link")

    def _tip_target_driver_callback(self, msg: Float32MultiArray) -> None:
        """Broadcast the driving model's tip_target head as a tip child frame.

        Absent while the driving model reports no target estimate.
        """
        if not msg.data:
            return  # the driver has reported no estimate
        pose_9d = np.asarray(msg.data, dtype=np.float32)
        self._broadcast_pose(
            pose_9d, "vla_tip_target_driver", parent_frame="drill_tip_link"
        )

    def _tip_target_fused_callback(self, msg: Float32MultiArray) -> None:
        """Broadcast the filtered/batch target estimate (pelvis frame).

        Servo EMA in VLA; in VLA_PLANNER the accumulator mean the finalize was
        planned on (fixed for the leg).
        """
        if not msg.data:
            return  # servo disabled or no prediction yet this session
        pose_9d = np.asarray(msg.data, dtype=np.float32)
        self._broadcast_pose(pose_9d, "vla_tip_target_fused")

    def _tip_target_live_callback(self, msg: Float32MultiArray) -> None:
        """Broadcast the target the finalize leg is steering to (pelvis frame).

        Planned estimate plus the current correction; published even with the
        correction disabled.
        """
        if not msg.data:
            return  # not finalizing, or no plan yet this session
        pose_9d = np.asarray(msg.data, dtype=np.float32)
        self._broadcast_pose(pose_9d, "vla_tip_target_live")

    def _tip_target_live_fine_only_callback(self, msg: Float32MultiArray) -> None:
        """Broadcast the never-applied fine-only shadow of the live target.

        Uses the detector's own depth instead of the donated range; only present
        when ``vla_planner_depth_source`` is ``stereo``.
        """
        if not msg.data:
            return  # no shadow: depth_source fine, switched off, or not finalizing
        pose_9d = np.asarray(msg.data, dtype=np.float32)
        self._broadcast_pose(pose_9d, "vla_tip_target_live_fine_only")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Build the canvas preparer once, from the wrist stream's own geometry.

        Crop side is the short edge; an uncorrectable lens leaves the preparer
        unset, which disables the canvas.
        """
        if self._wrist_preparer is not None:
            return
        side = min(int(msg.width), int(msg.height))
        try:
            self._wrist_preparer = preparer_from_camera_info(msg, side)
        except ValueError as exc:
            self.get_logger().error(
                f"Overlay canvas unusable: {exc}", throttle_duration_sec=10.0
            )
            return
        self.get_logger().info(
            f"perception overlay canvas prepared: {msg.width}x{msg.height} -> "
            f"{side}x{side} via {type(self._wrist_preparer).__name__}"
        )

    def _wrist_image_callback(self, frame: np.ndarray, _msg) -> None:
        if self._wrist_preparer is None:
            self.get_logger().warning(
                "Overlay canvas dropped: no CameraInfo on "
                f"{definitions.TOPIC_CAMERA_WRIST_INFO} yet",
                throttle_duration_sec=5.0,
            )
            return
        self._latest_wrist = self._wrist_preparer(frame)

    def _perception_detection_callback(self, msg: Float32MultiArray) -> None:
        """Draw the detector's target pixel on the latest prepared wrist frame.

        ``msg.data`` is ``[u, v, visibility, visible, depth_m]`` in prepared-frame
        pixels. Green crosshair when visible, red otherwise.
        """
        if not msg.data:
            return
        u, v, visibility, visible, depth = (float(x) for x in msg.data[:5])
        # 3D broadcast needs only CameraInfo, so it runs before the wrist-frame check.
        self._broadcast_detection(u, v, depth)
        if self._latest_wrist is None:
            return
        frame = self._latest_wrist.copy()
        if np.all(np.isfinite([u, v, visibility, visible, depth])):
            color = (0, 255, 0) if visible >= 0.5 else (255, 0, 0)  # rgb8
            center = (int(round(u)), int(round(v)))
            cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 40, 2)
            cv2.circle(frame, center, 18, color, 2)
            cv2.putText(
                frame,
                f"detection vis={visibility:.2f} d={depth:.3f}m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        tip = self._pixel_from(msg.data, 5)
        target = self._pixel_from(msg.data, 7)
        if tip is not None:
            tip_color = (0, 255, 255)  # cyan in the RGB canvas
            cv2.drawMarker(frame, tip, tip_color, cv2.MARKER_TILTED_CROSS, 34, 2)
            cv2.circle(frame, tip, 12, tip_color, 2)
            cv2.putText(
                frame,
                "drill tip",
                (tip[0] + 14, tip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tip_color,
                2,
            )
        if target is not None:
            target_color = (255, 0, 255)  # magenta in the RGB canvas
            cv2.drawMarker(frame, target, target_color, cv2.MARKER_DIAMOND, 34, 2)
            cv2.putText(
                frame,
                "servo target",
                (target[0] + 14, target[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                target_color,
                2,
            )
        if tip is not None and target is not None:
            cv2.line(frame, tip, target, (255, 255, 0), 2)
            error_px = float(np.linalg.norm(np.subtract(target, tip)))
            cv2.putText(
                frame,
                f"visual servo error={error_px:.1f}px",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
        # cv2.imencode expects BGR (frame is rgb8), and image_transport expects
        # the '<encoding>; <codec> compressed <encoding>' format string.
        out = self._cv_bridge.cv2_to_compressed_imgmsg(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), dst_format="jpg"
        )
        out.format = "bgr8; jpeg compressed bgr8"
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "hand_camera_optical_frame"
        self._overlay_pub.publish(out)

    def _broadcast_detection(self, u: float, v: float, depth: float) -> None:
        """Broadcast the raw detection as a TF frame in the wrist camera.

        Ungated: includes low-visibility and not-visible readings, so it shows
        where the model is looking even when ``vla_tip_target`` does not update.
        NaN pixels and non-positive ranges are skipped.
        """
        if self._wrist_preparer is None or not np.all(np.isfinite([u, v, depth])):
            return
        if depth <= 0.0:
            return
        # Same back-projection as the lifter, with the prepared frame's intrinsics.
        xyz = backproject_uv(
            np.array([u, v], dtype=float), depth, self._wrist_preparer.intrinsics
        )
        # Position only (identity rotation): the wall normal is not in this message.
        pose_9d = np.concatenate([xyz, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).astype(
            np.float32
        )
        self._broadcast_pose(
            pose_9d, "perception_detection", parent_frame="hand_camera_optical_frame"
        )

    @staticmethod
    def _pixel_from(data, start: int) -> tuple[int, int] | None:
        if len(data) < start + 2:
            return None
        uv = np.asarray(data[start : start + 2], dtype=float)
        if not np.all(np.isfinite(uv)):
            return None
        return int(round(float(uv[0]))), int(round(float(uv[1])))

    def _make_chunk_callback(self, path_name: str):
        def callback(msg: Float32MultiArray) -> None:
            chunk = np.asarray(msg.data, dtype=np.float32).reshape(-1, 9)
            self._path_pubs[path_name].publish(self._chunk_to_path(chunk))

        return callback

    def _chunk_to_path(self, chunk: np.ndarray) -> Path:
        """Convert an (N, 9) palm chunk into a Path (empty input clears RViz)."""
        path = Path()
        path.header.frame_id = _FRAME
        path.header.stamp = self.get_clock().now().to_msg()
        for pose_9d in chunk:
            se3 = RobotModel.action_9d_to_se3(pose_9d)
            quat = R.from_matrix(se3.rotation).as_quat()  # [x, y, z, w]
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(se3.translation[0])
            ps.pose.position.y = float(se3.translation[1])
            ps.pose.position.z = float(se3.translation[2])
            ps.pose.orientation.x = float(quat[0])
            ps.pose.orientation.y = float(quat[1])
            ps.pose.orientation.z = float(quat[2])
            ps.pose.orientation.w = float(quat[3])
            path.poses.append(ps)
        return path


def main():
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = DebugVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
