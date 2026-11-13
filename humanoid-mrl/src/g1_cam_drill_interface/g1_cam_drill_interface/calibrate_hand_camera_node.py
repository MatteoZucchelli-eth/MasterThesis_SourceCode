# MIT License
"""Calibrate the wrist camera's extrinsics against the AprilTag on the drill body.

Collects N detections of the drill-body tag and reports the camera transform for
both consumers, which must be updated together:

  * ``right_drill_link -> hand_camera_link`` in g1_29_dof_with_hand_drill.urdf.xml
    (TF), and
  * ``hand_drill_hilti/hand_cam_link_01`` in g1_29dof_rev_1_0_w_hilti_drill.usd
    (Isaac rendering).

The report ends with the delta against live TF.

    ros2 launch g1_cam_drill_interface calibrate_hand_camera_launch.py

Frames
------
``D``   drill body (USD ``hand_drill_hilti``); tag offset is measured here.
        +x toward the bit, +z up.
``L``   URDF ``right_drill_link``. Parallel to D, offset along the drill axis.
``C``   camera optical frame (+z view, +x right, +y down): AprilTag, CameraInfo
        and the USD ``hand_cam_link_01`` xform use it. The URDF joint targets the
        body frame ``hand_camera_link`` (+x forward), i.e. rotated by
        R_BODY_TO_OPTICAL relative to the USD output.
``T``   pupil_apriltags tag frame: +x right, +y down, +z out of the face away
        from the camera.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from g1_common.image_transport import subscribe_image, transport_name
from geometry_msgs.msg import PoseStamped
from pupil_apriltags import Detection, Detector
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo

from g1_cam_drill_interface.extrinsics import (
    average_poses,
    camera_pose_in_drill_body,
    DEFAULT_TAG_XYZ_IN_DRILL_BODY,
    DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY,
    tag_corners_in_tag_frame,
    tag_pose_in_camera,
    urdf_joint_origin,
    usd_xform,
)


class HandCameraCalibrationNode(Node):
    """Accumulate AprilTag detections and report the wrist camera extrinsics."""

    def __init__(self):
        super().__init__("calibrate_hand_camera_node")

        self._declare_parameters()

        self._bridge = CvBridge()
        self._detector = Detector(families=self.tag_family, nthreads=1, debug=False)

        # Completed once the report is out; main() spins until then.
        self.finished = Future()

        # Filled by the first CameraInfo, which then unsubscribes itself.
        self._camera_params: Optional[List[float]] = None
        self._mapx = None
        self._mapy = None

        self._translations: List[np.ndarray] = []
        self._rotations: List[np.ndarray] = []
        self._rejected = 0
        self._done = False
        self._overlays_saved = 0

        self._tf_buffer = None
        self._start_tf_listener()

        # BEST_EFFORT is required: it matches both BEST_EFFORT and RELIABLE
        # publishers, whereas a RELIABLE subscriber never matches a BEST_EFFORT one.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Transport (raw vs compressed) is inferred from the topic name. If the
        # reprojection error looks poor, try image_topic:=/camera/hand/color/image_raw
        # to avoid JPEG artefacts on the corners.
        self._image_sub = subscribe_image(
            self,
            self.image_topic,
            self._image_callback,
            sensor_qos,
            encoding="mono8",
            bridge=self._bridge,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback, sensor_qos
        )

        self._tag_pose_pub = self.create_publisher(
            PoseStamped, "/detected_tag_drill/tag_pose", 1
        )

        # Periodic status, so "no tag seen" is distinguishable from "no images".
        self._progress_timer = self.create_timer(5.0, self._report_progress)

        self.get_logger().info(
            f"collecting {self.num_samples} samples of tag id {self.tag_id} "
            f"({self.tag_family}, {self.tag_size_m * 1e3:.1f} mm) from "
            f"'{self.image_topic}' [{transport_name(self.image_topic)}] "
            f"+ '{self.camera_info_topic}'"
        )

    # ------------------------------------------------------------------ setup

    def _declare_parameters(self):
        self.declare_parameter("image_topic", "/camera/hand/color/image_raw/compressed")
        self.declare_parameter("camera_info_topic", "/camera/hand/color/camera_info")
        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("tag_size_m", 0.011)
        self.declare_parameter("tag_id", 1)
        self.declare_parameter("num_samples", 50)
        self.declare_parameter("tag_xyz_in_drill_body", DEFAULT_TAG_XYZ_IN_DRILL_BODY)
        self.declare_parameter(
            "urdf_link_xyz_in_drill_body", DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY
        )
        self.declare_parameter("urdf_link_frame", "right_drill_link")
        self.declare_parameter("camera_optical_frame", "hand_camera_optical_frame")
        self.declare_parameter("output_path", "~/.ros/hand_camera_extrinsics.yaml")
        # In the workspace so the images survive a non-persistent home directory.
        self.declare_parameter("overlay_dir", "/workspace/calibration/hand_camera")
        self.declare_parameter("overlay_count", 5)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.tag_family = self.get_parameter("tag_family").value
        self.tag_size_m = self.get_parameter("tag_size_m").value
        self.tag_id = self.get_parameter("tag_id").value
        self.num_samples = self.get_parameter("num_samples").value
        self.tag_xyz = np.asarray(
            self.get_parameter("tag_xyz_in_drill_body").value, dtype=float
        )
        self.urdf_link_xyz = np.asarray(
            self.get_parameter("urdf_link_xyz_in_drill_body").value, dtype=float
        )
        self.urdf_link_frame = self.get_parameter("urdf_link_frame").value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").value
        self.output_path = self.get_parameter("output_path").value
        self.overlay_dir = self.get_parameter("overlay_dir").value
        self.overlay_count = self.get_parameter("overlay_count").value

    def _start_tf_listener(self):
        """Start filling a TF buffer for the final delta; optional (skipped if
        absent)."""
        try:
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        except Exception as exc:  # noqa: BLE001 - TF is a convenience here
            self.get_logger().warning(f"no TF listener ({exc}); delta will be skipped")

    # -------------------------------------------------------------- callbacks

    def _camera_info_callback(self, msg: CameraInfo):
        fx, fy = msg.k[0], msg.k[4]
        cx, cy = msg.k[2], msg.k[5]
        self._camera_params = [fx, fy, cx, cy]

        # The detector has no distortion model (See3CAM k1 = -0.35), so undistort first.
        k = np.array(msg.k).reshape(3, 3)
        d = np.array(msg.d)
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            k, d, None, k, (msg.width, msg.height), cv2.CV_16SC2
        )

        self.get_logger().info(f"intrinsics: K={list(msg.k)}, D={list(msg.d)}")
        self.destroy_subscription(self._camera_info_sub)

    def _image_callback(self, image, msg):
        """Detect the tag in one decoded mono8 frame.

        ``msg`` supplies the header for the published tag pose.
        """
        if self._camera_params is None or self._done:
            return

        image = cv2.remap(image, self._mapx, self._mapy, interpolation=cv2.INTER_LINEAR)

        detections = self._detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=self._camera_params,
            tag_size=self.tag_size_m,
        )

        # Duplicate ids mean one is spurious; drop the frame.
        matches = [d for d in detections if d.tag_id == self.tag_id]
        if len(matches) != 1:
            self._rejected += 1
            return

        detection = matches[0]
        self._accumulate(detection)
        self._publish_tag_pose(detection, msg.header)
        # Rendering is costlier than detection; only while overlays remain to save.
        if self._overlays_saved < self.overlay_count:
            self._render_overlay(detection, image)

        if len(self._translations) >= self.num_samples:
            self._done = True
            self._finish()

    # ------------------------------------------------------------ calibration

    def _accumulate(self, detection: Detection):
        """Add one detection to the sample set, as a camera pose in the drill body."""
        translation, rotation = camera_pose_in_drill_body(
            detection.pose_t, detection.pose_R, self.tag_xyz
        )
        self._translations.append(translation)
        self._rotations.append(rotation)

    def _finish(self):
        t_drill_cam, r_drill_cam, spread_mm, max_dev_deg = average_poses(
            self._translations, self._rotations
        )
        t_link_cam, rpy = urdf_joint_origin(
            t_drill_cam, r_drill_cam, self.urdf_link_xyz
        )
        _, quat_wxyz = usd_xform(t_drill_cam, r_drill_cam)

        delta = self._compare_with_tf(t_link_cam, r_drill_cam)

        self._log_report(
            t_drill_cam, spread_mm, max_dev_deg, t_link_cam, rpy, quat_wxyz, delta
        )
        self._write_report(
            t_drill_cam, spread_mm, max_dev_deg, t_link_cam, rpy, quat_wxyz, delta
        )
        self.finished.set_result(True)

    def _compare_with_tf(self, t_link_cam, r_drill_cam) -> Optional[dict]:
        """Delta between the new calibration and current TF, or None if unavailable."""
        if self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self.urdf_link_frame, self.camera_optical_frame, rclpy.time.Time()
            ).transform
        except Exception as exc:  # noqa: BLE001 - any TF failure is non-fatal
            self.get_logger().warning(
                f"no TF {self.urdf_link_frame} -> {self.camera_optical_frame} "
                f"to compare against ({exc}); skipping the delta"
            )
            return None

        tf_t = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        tf_r = Rotation.from_quat(
            [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]
        )
        position_delta = t_link_cam - tf_t
        return {
            "position_delta_mm": (position_delta * 1e3).tolist(),
            "position_delta_norm_mm": float(np.linalg.norm(position_delta) * 1e3),
            "rotation_delta_deg": float(
                np.degrees((tf_r.inv() * r_drill_cam).magnitude())
            ),
        }

    # -------------------------------------------------------------- reporting

    def _log_report(
        self, t_drill_cam, spread_mm, max_dev_deg, t_link_cam, rpy, quat_wxyz, delta
    ):
        lines = [
            "",
            "=" * 72,
            f"wrist camera extrinsics from {len(self._translations)} samples "
            f"({self._rejected} frames rejected)",
            f"  spread: x {spread_mm[0]:.2f} mm  y {spread_mm[1]:.2f} mm  "
            f"z {spread_mm[2]:.2f} mm | rotation max {max_dev_deg:.3f} deg from mean",
            "",
            "g1_state_publisher/urdf/g1_29_dof_with_hand_drill.urdf.xml,",
            f"  joint hand_camera_joint ({self.urdf_link_frame} -> hand_camera_link):",
            f'    <origin xyz="{t_link_cam[0]:.6f} {t_link_cam[1]:.6f} '
            f'{t_link_cam[2]:.6f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>',
            "",
            "then, so Isaac renders from the same pose (it reads the USD, never",
            "the URDF):",
            "  src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py",
            "  (needs pxr: Isaac's python.sh, or x86_64 + `pip install usd-core`)",
            "",
            "for reference, that is what the prim hand_drill_hilti/hand_cam_link_01",
            "will end up at, in the optical convention it is authored in:",
            f"    xformOp:translate = ({t_drill_cam[0]:.8f}, {t_drill_cam[1]:.8f}, "
            f"{t_drill_cam[2]:.8f})",
            f"    xformOp:orient    = ({quat_wxyz[0]:.10f}, {quat_wxyz[1]:.10f}, "
            f"{quat_wxyz[2]:.10f}, {quat_wxyz[3]:.10f})  # (w, x, y, z)",
            "",
        ]
        if delta is None:
            lines.append("live TF unavailable, no delta computed")
        else:
            dx, dy, dz = delta["position_delta_mm"]
            lines += [
                f"vs live TF {self.urdf_link_frame} -> {self.camera_optical_frame}:",
                f"  position {delta['position_delta_norm_mm']:.1f} mm "
                f"(dx {dx:+.1f}, dy {dy:+.1f}, dz {dz:+.1f} mm)",
                f"  rotation {delta['rotation_delta_deg']:.3f} deg",
                "  a non-zero delta means the URDF still needs the edit above,",
                "  and the sync after it, or sim and robot part ways.",
            ]
        lines.append("=" * 72)
        self.get_logger().info("\n".join(lines))

    def _write_report(
        self, t_drill_cam, spread_mm, max_dev_deg, t_link_cam, rpy, quat_wxyz, delta
    ):
        if not self.output_path:
            return

        path = os.path.expanduser(self.output_path)
        report = {
            "samples": len(self._translations),
            "rejected_frames": self._rejected,
            "tag": {
                "family": self.tag_family,
                "id": self.tag_id,
                "size_m": self.tag_size_m,
                "xyz_in_drill_body": self.tag_xyz.tolist(),
            },
            "spread": {
                "translation_mm": spread_mm.tolist(),
                "rotation_max_deg": float(max_dev_deg),
            },
            "urdf_hand_camera_joint": {
                "parent": self.urdf_link_frame,
                "child": "hand_camera_link",
                "xyz": t_link_cam.tolist(),
                "rpy": rpy.tolist(),
            },
            "usd_hand_cam_link_01": {
                "translate": t_drill_cam.tolist(),
                "orient_wxyz": quat_wxyz,
            },
            "vs_live_tf": delta,
        }

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as handle:
                handle.write(
                    "# Wrist camera extrinsics, measured by "
                    "g1_cam_drill_interface/calibrate_hand_camera_node.\n"
                    "# The record of one measurement, not a config anything reads:\n"
                    "# the numbers still have to be pasted into the URDF, and the\n"
                    "# USD synced from it by\n"
                    "# g1_simulation/scripts/sync_camera_extrinsics_to_usd.py.\n"
                )
                yaml.safe_dump(
                    report, handle, default_flow_style=False, sort_keys=False
                )
            self.get_logger().info(f"wrote {path}")
        except OSError as exc:
            self.get_logger().error(f"could not write {path}: {exc}")

    def _camera_matrix(self) -> np.ndarray:
        fx, fy, cx, cy = self._camera_params
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def _project(self, points_tag_frame, t_cam_tag, r_cam_tag) -> np.ndarray:
        """Project tag-frame points through the intrinsics. No distortion: the
        image these land on has already been undistorted with the same K."""
        r_vec, _ = cv2.Rodrigues(np.asarray(r_cam_tag, dtype=float))
        projected, _ = cv2.projectPoints(
            np.asarray(points_tag_frame, dtype=float),
            r_vec,
            np.asarray(t_cam_tag, dtype=float).reshape(3, 1),
            self._camera_matrix(),
            None,
        )
        return projected.reshape(-1, 2)

    def _predicted_tag_pose(self):
        """Tag pose in the camera predicted by current TF, or None if TF is
        unavailable."""
        if self._tf_buffer is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self.urdf_link_frame, self.camera_optical_frame, rclpy.time.Time()
            ).transform
        except Exception:  # noqa: BLE001 - overlay is a convenience
            return None

        # L and D are parallel frames, so moving the camera pose from
        # right_drill_link into the drill body is a pure translation.
        t_drill_cam = self.urdf_link_xyz + np.array(
            [tf.translation.x, tf.translation.y, tf.translation.z]
        )
        r_drill_cam = Rotation.from_quat(
            [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]
        ).as_matrix()
        return tag_pose_in_camera(t_drill_cam, r_drill_cam, self.tag_xyz)

    def _render_overlay(self, detection: Detection, image_mono8):
        """Draw the detected tag against the TF-predicted one.

        * GREEN: detection; its corner reprojection error checks the intrinsics.
        * MAGENTA: pose implied by live TF plus the known tag offset; its offset
          from green checks the extrinsics.
        """
        image_bgr = cv2.cvtColor(image_mono8, cv2.COLOR_GRAY2BGR)
        corners_model = tag_corners_in_tag_frame(self.tag_size_m)
        axes_model = np.vstack(
            [[0.0, 0.0, 0.0], np.eye(3) * 0.5 * float(self.tag_size_m)]
        )

        detected_corners = np.asarray(detection.corners, dtype=float)
        reprojected = self._project(corners_model, detection.pose_t, detection.pose_R)
        reprojection_px = float(
            np.linalg.norm(reprojected - detected_corners, axis=1).mean()
        )

        self._draw_tag(image_bgr, reprojected,
                       self._project(axes_model, detection.pose_t, detection.pose_R),
                       (0, 255, 0), "detected")

        predicted = self._predicted_tag_pose()
        centre_offset_px = None
        if predicted is not None:
            t_cam_tag, r_cam_tag = predicted
            predicted_corners = self._project(corners_model, t_cam_tag, r_cam_tag)
            self._draw_tag(image_bgr, predicted_corners,
                           self._project(axes_model, t_cam_tag, r_cam_tag),
                           (255, 0, 255), "TF predicts")
            centre_offset_px = float(
                np.linalg.norm(predicted_corners.mean(axis=0)
                               - detected_corners.mean(axis=0))
            )

        self._annotate(image_bgr, reprojection_px, centre_offset_px)
        self._save_overlay(image_bgr)

    @staticmethod
    def _draw_tag(image_bgr, corners, axes, colour, label):
        """Outline plus RGB-per-axis triad, so orientation is readable at a glance."""
        pts = np.round(corners).astype(int)
        cv2.polylines(image_bgr, [pts], isClosed=True, color=colour, thickness=2)
        # Mark the first corner to disambiguate 90 deg rotations.
        cv2.circle(image_bgr, tuple(pts[0]), 5, colour, -1)

        origin = tuple(np.round(axes[0]).astype(int))
        for index, axis_colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            tip = tuple(np.round(axes[index + 1]).astype(int))
            cv2.arrowedLine(image_bgr, origin, tip, axis_colour, 2, tipLength=0.25)
        cv2.putText(image_bgr, label, (pts[0][0] + 8, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    def _annotate(self, image_bgr, reprojection_px, centre_offset_px):
        lines = [
            f"tag {self.tag_id} ({self.tag_family}, {self.tag_size_m * 1e3:.1f} mm)",
            f"intrinsics: corner reprojection {reprojection_px:.2f} px",
        ]
        if centre_offset_px is None:
            lines.append("extrinsics: no TF, nothing to compare against")
        else:
            lines.append(f"extrinsics: green vs magenta {centre_offset_px:.1f} px")
        for row, text in enumerate(lines):
            cv2.putText(image_bgr, text, (10, 24 + 22 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image_bgr, text, (10, 24 + 22 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _save_overlay(self, image_bgr):
        """Write the first ``overlay_count`` overlays to disk."""
        if not self.overlay_dir or self._overlays_saved >= self.overlay_count:
            return
        path = os.path.expanduser(self.overlay_dir)
        try:
            os.makedirs(path, exist_ok=True)
            name = os.path.join(path, f"overlay_{self._overlays_saved:02d}.png")
            cv2.imwrite(name, image_bgr)
            if self._overlays_saved == 0:
                self.get_logger().info(f"writing overlay images to {path}")
        except OSError as exc:
            self.get_logger().warning(f"could not write overlay image: {exc}")
            self.overlay_dir = ""
            return
        self._overlays_saved += 1

    def _publish_tag_pose(self, detection: Detection, header):
        """Republish the raw detection as a PoseStamped, for rviz / TF checks."""
        pose = PoseStamped()
        pose.header = header
        t_cam_tag = np.asarray(detection.pose_t).reshape(3)
        quat = Rotation.from_matrix(np.asarray(detection.pose_R)).as_quat()
        (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ) = (float(v) for v in t_cam_tag)
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = (float(v) for v in quat)
        self._tag_pose_pub.publish(pose)

    def _report_progress(self):
        if self._done:
            return
        if self._camera_params is None:
            self.get_logger().warning(
                f"waiting for CameraInfo on '{self.camera_info_topic}'"
            )
        elif not self._translations:
            self.get_logger().warning(
                f"no tag id {self.tag_id} seen yet ({self._rejected} frames without "
                f"exactly one match) -- check the tag is in view, and that "
                f"tag_size_m={self.tag_size_m} matches the printed tag"
            )
        else:
            self.get_logger().info(
                f"{len(self._translations)}/{self.num_samples} samples"
            )


def main(args=None):
    rclpy.init(args=args)
    node = HandCameraCalibrationNode()
    try:
        rclpy.spin_until_future_complete(node, node.finished)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
