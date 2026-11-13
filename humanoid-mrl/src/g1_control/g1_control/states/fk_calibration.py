# MIT License
"""FK_CALIBRATION: camera-measured drill-tip positioning and error capture.

- The hand camera is always subscribed as ``CompressedImage`` so calibration
  never attaches a reader to the raw writer on the robot.
- The goal is the averaged tag centre offset along tag +y (image-down) and +z
  (normal), with the tip's current orientation preserved (point-coincidence
  FK test).
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pinocchio as pin
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from pupil_apriltags import Detector
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster, TransformException

from g1_control import topics
from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils.fk_calibration import (
    average_tag_poses,
    CameraModel,
    point_offset_from_tag,
    project_camera_point,
)
from g1_control.utils.ros_helpers import LATCHED_QOS
from g1_control.utils.ros_tf_utils import se3_to_transform_stamped
from g1_definitions import definitions


_CSV_COLUMNS = [
    "IK error",
    "Joint tracking error (desired - actual joint pose)",
    "Cartesian tracking error of the drill tip (desired 3D pose - actual pose from FK)",
    "Final camera-measured displacement",
]


class FkCalibrationHandler(StateHandler):
    """Detect tag 0, move ``drill_tip_link`` to it, and record one CSV row."""

    _PELVIS_FRAME = "pelvis"
    _TAG_FRAME = "apriltag_detection"
    _TIP_FRAME = definitions.DRILL_TIP_FRAME

    def __init__(self, node, ctx: ControlContext, controller) -> None:
        self._node = node
        self._ctx = ctx
        self._controller = controller
        self._logger = node.get_logger()
        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(node)

        self._tag_id = int(self._param("fk_calibration_tag_id", 0))
        self._tag_size_m = float(self._param("fk_calibration_tag_size_m", 0.080))
        self._sample_count = int(self._param("fk_calibration_sample_count", 30))
        self._target_below_m = float(
            self._param("fk_calibration_target_below_m", 0.030)
        )
        self._target_normal_m = float(
            self._param("fk_calibration_target_normal_m", 0.0)
        )
        self._goal_duration_s = float(
            self._param("fk_calibration_goal_duration_s", 5.0)
        )
        self._tracking_threshold_m = float(
            self._param("fk_calibration_tracking_threshold_m", 0.002)
        )
        self._tracking_timeout_s = float(
            self._param("fk_calibration_tracking_timeout_s", 15.0)
        )
        self._csv_path_template = str(
            self._param(
                "fk_calibration_csv_path",
                "~/.ros/fk_calibration_{timestamp}.csv",
            )
        )
        tag_family = str(self._param("fk_calibration_tag_family", "tag36h11"))
        self._hand_topic = self._compressed_topic(
            str(
                self._param(
                    "fk_calibration_hand_image_topic",
                    definitions.TOPIC_CAMERA_WRIST + "/compressed",
                )
            )
        )
        self._head_topic = self._compressed_topic(
            str(
                self._param(
                    "fk_calibration_head_image_topic",
                    definitions.TOPIC_CAMERA + "/compressed",
                )
            )
        )
        self._hand_frame = str(
            self._param(
                "fk_calibration_hand_camera_frame", "hand_camera_optical_frame"
            )
        )
        self._head_frame = str(
            self._param(
                "fk_calibration_head_camera_frame", "d435_color_optical_frame"
            )
        )

        if self._sample_count <= 0:
            raise ValueError("fk_calibration_sample_count must be positive")
        if self._tag_size_m <= 0.0:
            raise ValueError("fk_calibration_tag_size_m must be positive")
        if not np.isfinite(self._target_below_m) or self._target_below_m < 0.0:
            raise ValueError(
                "fk_calibration_target_below_m must be finite and non-negative"
            )
        if not np.isfinite(self._target_normal_m):
            raise ValueError("fk_calibration_target_normal_m must be finite")

        self._detector = Detector(families=tag_family, nthreads=1)
        self._sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._hand_model: CameraModel | None = None
        self._head_model: CameraModel | None = None

        # Explicit message types so a launch override cannot switch to raw Image.
        self._hand_sub = node.create_subscription(
            CompressedImage, self._hand_topic, self._hand_image_callback, self._sensor_qos
        )
        self._head_sub = node.create_subscription(
            CompressedImage, self._head_topic, self._head_image_callback, self._sensor_qos
        )
        self._hand_info_sub = node.create_subscription(
            CameraInfo,
            definitions.TOPIC_CAMERA_WRIST_INFO,
            self._hand_info_callback,
            self._sensor_qos,
        )
        self._head_info_sub = node.create_subscription(
            CameraInfo,
            definitions.TOPIC_CAMERA_INFO,
            self._head_info_callback,
            self._sensor_qos,
        )
        self._start_topic = str(
            self._param(
                "fk_calibration_start_topic", topics.TOPIC_FK_CALIBRATION_START
            )
        )
        self._start_sub = node.create_subscription(
            Empty, self._start_topic, self._start_callback, 1
        )
        self._pose_pub = node.create_publisher(
            PoseStamped, topics.TOPIC_FK_CALIBRATION_POSE, LATCHED_QOS
        )
        self._hand_overlay_pub = node.create_publisher(
            CompressedImage,
            topics.TOPIC_FK_CALIBRATION_HAND_OVERLAY,
            self._sensor_qos,
        )
        self._head_overlay_pub = node.create_publisher(
            CompressedImage,
            topics.TOPIC_FK_CALIBRATION_HEAD_OVERLAY,
            self._sensor_qos,
        )

        self._right_arm_indices = ctx.robot_model.get_joint_group_indices("right_arm")
        self._active = False
        self._reset_session()
        self._logger.info(
            "FK Calibration ready: R1 + X, then publish std_msgs/Empty on "
            f"'{self._start_topic}'; tag "
            f"{self._tag_id}, black size {self._tag_size_m * 1e3:.0f} mm, "
            f"{self._sample_count} consecutive detections from '{self._hand_topic}'."
        )

    def _param(self, name: str, default):
        return self._node.get_parameter(name).value if self._node.has_parameter(name) else default

    def _compressed_topic(self, topic: str) -> str:
        topic = topic.rstrip("/")
        if not topic.endswith("/compressed"):
            raise ValueError(
                f"FK Calibration requires a compressed image topic, got '{topic}'"
            )
        return topic

    def _reset_session(self) -> None:
        """Clear all rows on state entry, then initialise an idle round."""
        self._rows: list[dict[str, str]] = []
        self._reset_round()

    def _reset_round(self) -> None:
        """Clear one round without discarding completed rows."""
        self._translations: list[np.ndarray] = []
        self._rotations: list[np.ndarray] = []
        self._average_camera: pin.SE3 | None = None
        self._average_pelvis: pin.SE3 | None = None
        self._goal_pelvis: pin.SE3 | None = None
        self._ik_q: np.ndarray | None = None
        self._goal_started_at: float | None = None
        self._terminal = False
        self._terminal_metrics: dict[str, np.ndarray] | None = None
        self._final_displacement: np.ndarray | None = None
        self._await_final_detection = False
        self._round_active = False
        self._current_row_staged = False

    # ── State lifecycle and control tick ──────────────────────────────────

    def on_enter(self, previous_state: int) -> None:
        del previous_state
        self._reset_session()
        self._active = True
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.targets.right_9d = None
        if not self._ctx.has_drill_tip:
            self._logger.error(
                f"FK Calibration cannot run: '{self._TIP_FRAME}' is absent from the URDF."
            )
        self._logger.info(
            f"[fk-calibration] entered; holding until std_msgs/Empty arrives on "
            f"'{self._start_topic}'."
        )

    def on_exit(self) -> None:
        self._active = False
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.targets.right_9d = None
        if self._terminal_metrics is not None and not self._current_row_staged:
            # Early exit still records the row, with NaN camera measurements.
            self._append_row(np.full(3, np.nan))
        self._write_csv()

    def update(self) -> None:
        if not self._active:
            return
        rs = self._ctx.robot_state
        rs.twist_command = np.zeros(3)
        rs.upper_body_target = None
        rs.upper_body_target_chunk = None

        if self._average_pelvis is not None:
            stamped = se3_to_transform_stamped(
                self._average_pelvis,
                self._PELVIS_FRAME,
                self._TAG_FRAME,
                self._node.get_clock().now().to_msg(),
            )
            self._tf_broadcaster.sendTransform(stamped)

        if self._goal_started_at is None or self._terminal:
            return
        actual = self._current_tip_pose()
        if actual is None:
            return
        error = self._goal_pelvis.translation - actual.translation
        elapsed = time.monotonic() - self._goal_started_at
        if np.linalg.norm(error) < self._tracking_threshold_m:
            self._finish_motion("reached", elapsed, actual)
        elif elapsed >= self._tracking_timeout_s:
            self._finish_motion("timed out", elapsed, actual)

    # ── Camera input and detection ────────────────────────────────────────

    def _start_callback(self, _msg: Empty) -> None:
        """Start one new round only while the state is active and idle."""
        if not self._active:
            self._logger.warning(
                f"[fk-calibration] ignored start on '{self._start_topic}': "
                "FK_CALIBRATION is not active.",
                throttle_duration_sec=2.0,
            )
            return
        if self._round_active:
            self._logger.warning(
                "[fk-calibration] ignored start: a calibration round is already "
                "running.",
                throttle_duration_sec=2.0,
            )
            return

        self._reset_round()
        self._round_active = True
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.targets.right_9d = None
        self._logger.info(
            f"[fk-calibration] starting round {len(self._rows) + 1}; collecting "
            f"{self._sample_count} consecutive tag-{self._tag_id} detections."
        )

    def _hand_info_callback(self, msg: CameraInfo) -> None:
        try:
            self._hand_model = CameraModel.from_msg(msg, self._hand_frame)
            self._hand_frame = self._hand_model.frame
        except ValueError as exc:
            self._logger.error(f"Invalid hand CameraInfo: {exc}")

    def _head_info_callback(self, msg: CameraInfo) -> None:
        try:
            self._head_model = CameraModel.from_msg(msg, self._head_frame)
            self._head_frame = self._head_model.frame
        except ValueError as exc:
            self._logger.error(f"Invalid head CameraInfo: {exc}")

    def _decode(self, msg: CompressedImage) -> np.ndarray | None:
        try:
            return self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except (CvBridgeError, ValueError) as exc:
            self._logger.error(
                f"Dropped FK Calibration image: {exc}", throttle_duration_sec=5.0
            )
            return None

    def _detect(self, frame_bgr: np.ndarray, model: CameraModel):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # pupil_apriltags takes pinhole parameters only; rectify when distorted.
        if model.distortion.size and np.any(model.distortion):
            gray = cv2.undistort(gray, model.matrix, model.distortion, None, model.matrix)
        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=model.detector_params,
            tag_size=self._tag_size_m,
        )
        matches = [d for d in detections if int(d.tag_id) == self._tag_id]
        return matches[0] if len(matches) == 1 else None

    def _hand_image_callback(self, msg: CompressedImage) -> None:
        if not self._active:
            return
        frame = self._decode(msg)
        model = self._hand_model
        if frame is None or model is None:
            if model is None:
                self._logger.warning(
                    "[fk-calibration] waiting for hand CameraInfo",
                    throttle_duration_sec=5.0,
                )
            return

        collecting = self._round_active and self._average_pelvis is None
        detection = self._detect(frame, model)
        live_tag_camera = None
        if detection is None:
            self._logger.warning(
                f"[fk-calibration] AprilTag {self._tag_id} not detected.",
                throttle_duration_sec=2.0,
            )
            if collecting and self._translations:
                self._translations.clear()
                self._rotations.clear()
                self._logger.warning(
                    "[fk-calibration] detection sequence broken; restarting the "
                    f"{self._sample_count}-frame average.",
                    throttle_duration_sec=2.0,
                )
        else:
            self._logger.info(
                f"[fk-calibration] AprilTag {self._tag_id} detected successfully.",
                throttle_duration_sec=2.0,
            )
            live_tag_camera = np.asarray(detection.pose_t, dtype=float).reshape(3)
            if collecting:
                self._accept_sample(detection, msg)
            elif self._await_final_detection:
                self._record_final_displacement(live_tag_camera, msg)

        tag_camera = self._point_in_camera(
            self._average_pelvis.translation if self._average_pelvis is not None else None,
            model.frame,
            msg.header.stamp,
        )
        if tag_camera is None:
            tag_camera = live_tag_camera
        tip_camera = self._frame_origin_in_camera(
            self._TIP_FRAME, model.frame, msg.header.stamp
        )
        tip_camera_inverse = self._frame_origin_in_camera_via_inverse(
            self._TIP_FRAME, model.frame, msg.header.stamp
        )
        self._publish_overlay(
            frame,
            msg,
            model,
            tag_camera,
            tip_camera,
            tip_camera_inverse,
            self._hand_overlay_pub,
        )

    def _head_image_callback(self, msg: CompressedImage) -> None:
        if not self._active:
            return
        frame = self._decode(msg)
        model = self._head_model
        if frame is None or model is None:
            return
        # Head image only shows projections of the hand-camera-measured point.
        tag_camera = self._point_in_camera(
            self._average_pelvis.translation if self._average_pelvis is not None else None,
            model.frame,
            msg.header.stamp,
        )
        tip_camera = self._frame_origin_in_camera(
            self._TIP_FRAME, model.frame, msg.header.stamp
        )
        tip_camera_inverse = self._frame_origin_in_camera_via_inverse(
            self._TIP_FRAME, model.frame, msg.header.stamp
        )
        self._publish_overlay(
            frame,
            msg,
            model,
            tag_camera,
            tip_camera,
            tip_camera_inverse,
            self._head_overlay_pub,
        )

    def _accept_sample(self, detection, msg: CompressedImage) -> None:
        self._translations.append(
            np.asarray(detection.pose_t, dtype=np.float64).reshape(3)
        )
        self._rotations.append(np.asarray(detection.pose_R, dtype=np.float64))
        count = len(self._translations)
        self._logger.info(
            f"[fk-calibration] detections {count}/{self._sample_count}",
            throttle_duration_sec=0.5,
        )
        if count < self._sample_count:
            return

        self._average_camera = average_tag_poses(self._translations, self._rotations)
        camera_frame = msg.header.frame_id or self._hand_model.frame
        pelvis_from_camera = self._lookup_se3(
            self._PELVIS_FRAME, camera_frame, msg.header.stamp
        )
        if pelvis_from_camera is None:
            # Keep the finished batch and retry the transform on the next image.
            self._logger.warning(
                f"[fk-calibration] waiting for TF {self._PELVIS_FRAME} <- "
                f"{camera_frame}",
                throttle_duration_sec=2.0,
            )
            self._translations.pop()
            self._rotations.pop()
            return
        self._average_pelvis = pelvis_from_camera * self._average_camera
        self._publish_average_pose()
        self._start_goal()

    # ── Goal, convergence, and CSV metrics ────────────────────────────────

    def _start_goal(self) -> None:
        if not self._ctx.has_drill_tip or self._average_pelvis is None:
            return
        current_tip = self._current_tip_pose()
        if current_tip is None:
            return
        # pupil_apriltags: tag +y is image-down, +z points into the surface.
        # Offsets are applied through the averaged tag rotation.
        target_position = point_offset_from_tag(
            self._average_pelvis, self._target_below_m, self._target_normal_m
        )
        # Preserve the tool orientation; this is a point-coincidence FK test.
        self._goal_pelvis = pin.SE3(
            current_tip.rotation.copy(), target_position
        )
        self._ik_q = self._ctx.robot_model.solve_ik(
            self._TIP_FRAME, self._goal_pelvis, self._ctx.full_q
        )
        upper_target = self._ctx.robot_model.assemble_upper_body_target(
            self._ctx.default_upper_body_pose, self._ctx.full_q, self._ik_q
        )
        self._controller.schedule_upper_body_target(
            upper_target, self._goal_duration_s
        )
        self._ctx.targets.right_9d = self._ctx.robot_model.frame_pose_9d(
            self._ik_q, self._TIP_FRAME
        )
        self._goal_started_at = time.monotonic()
        p = self._goal_pelvis.translation
        self._logger.info(
            f"[fk-calibration] averaged {self._sample_count} detections; "
            f"scheduled drill-tip goal {self._target_below_m * 1e3:.0f} mm below "
            f"and {self._target_normal_m * 1e3:+.0f} mm along the tag normal "
            f"from the tag centre at [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m "
            f"in pelvis over {self._goal_duration_s:.1f} s."
        )

    def _finish_motion(self, result: str, elapsed: float, actual: pin.SE3) -> None:
        self._terminal = True
        self._await_final_detection = True
        model = self._ctx.robot_model
        model.cache_forward_kinematics(self._ik_q, auto_clip=False)
        ik_pose = model.frame_placement(self._TIP_FRAME)
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        self._terminal_metrics = {
            "ik": self._goal_pelvis.translation - ik_pose.translation,
            "joint": np.subtract(
                self._ik_q[self._right_arm_indices],
                self._ctx.full_q[self._right_arm_indices],
            ),
            "cartesian": self._goal_pelvis.translation - actual.translation,
        }
        error_mm = np.linalg.norm(self._terminal_metrics["cartesian"]) * 1e3
        self._logger.info(
            f"[fk-calibration] {result} after {elapsed:.2f} s; Cartesian "
            f"tracking error={error_mm:.2f} mm. Waiting for one final hand-camera "
            "detection for the physical displacement."
        )

    def _record_final_displacement(
        self, tag_camera: np.ndarray, msg: CompressedImage
    ) -> None:
        camera_frame = msg.header.frame_id or self._hand_model.frame
        tip_camera = self._frame_origin_in_camera(
            self._TIP_FRAME, camera_frame, msg.header.stamp
        )
        if tip_camera is None:
            return
        self._final_displacement = tag_camera - tip_camera
        self._await_final_detection = False
        self._append_row(self._final_displacement)
        self._round_active = False
        d = self._final_displacement * 1e3
        self._logger.info(
            "[fk-calibration] final camera displacement tag - drill_tip_link = "
            f"[{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}] mm; row staged for CSV "
            "creation when the state exits. Waiting for another start message."
        )

    def _append_row(self, final_displacement: np.ndarray) -> None:
        if self._terminal_metrics is None or self._current_row_staged:
            return

        def cell(value) -> str:
            return json.dumps([float(x) for x in np.asarray(value).reshape(-1)])

        self._rows.append(
            {
                _CSV_COLUMNS[0]: cell(self._terminal_metrics["ik"]),
                _CSV_COLUMNS[1]: cell(self._terminal_metrics["joint"]),
                _CSV_COLUMNS[2]: cell(self._terminal_metrics["cartesian"]),
                _CSV_COLUMNS[3]: cell(final_displacement),
            }
        )
        self._current_row_staged = True

    def _write_csv(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.expanduser(self._csv_path_template.format(timestamp=timestamp))
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self._rows)
            self._logger.info(
                f"[fk-calibration] wrote {len(self._rows)} row(s) to '{path}'."
            )
        except (OSError, KeyError, ValueError) as exc:
            self._logger.error(f"[fk-calibration] could not write CSV: {exc}")

    # ── TF and visualization helpers ─────────────────────────────────────

    def _lookup_se3(self, target: str, source: str, stamp) -> pin.SE3 | None:
        try:
            tf = self._ctx.tf_buffer.lookup_transform(target, source, stamp)
        except TransformException:
            try:
                tf = self._ctx.tf_buffer.lookup_transform(
                    target, source, rclpy.time.Time()
                )
            except TransformException:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return pin.SE3(
            pin.Quaternion(q.w, q.x, q.y, q.z).normalized().matrix(),
            np.array([t.x, t.y, t.z], dtype=float),
        )

    def _point_in_camera(
        self, point_pelvis: np.ndarray | None, camera_frame: str, stamp
    ) -> np.ndarray | None:
        if point_pelvis is None:
            return None
        camera_from_pelvis = self._lookup_se3(
            camera_frame, self._PELVIS_FRAME, stamp
        )
        return (
            None
            if camera_from_pelvis is None
            else camera_from_pelvis.act(np.asarray(point_pelvis))
        )

    def _frame_origin_in_camera(
        self, frame: str, camera_frame: str, stamp
    ) -> np.ndarray | None:
        camera_from_frame = self._lookup_se3(camera_frame, frame, stamp)
        return (
            None
            if camera_from_frame is None
            else camera_from_frame.translation.copy()
        )

    def _frame_origin_in_camera_via_inverse(
        self, frame: str, camera_frame: str, stamp
    ) -> np.ndarray | None:
        """Camera-frame origin obtained by inverting ``T(frame <- camera)``.

        Mirrors the visual-servo projection path for comparison with the direct
        ``T(camera <- frame)`` lookup.
        """
        frame_from_camera = self._lookup_se3(frame, camera_frame, stamp)
        return (
            None
            if frame_from_camera is None
            else frame_from_camera.inverse().translation.copy()
        )

    def _current_tip_pose(self) -> pin.SE3 | None:
        if not self._ctx.has_drill_tip:
            return None
        self._ctx.robot_model.cache_forward_kinematics(
            self._ctx.full_q, auto_clip=False
        )
        return self._ctx.robot_model.frame_placement(self._TIP_FRAME)

    def _publish_average_pose(self) -> None:
        pose = PoseStamped()
        pose.header.frame_id = self._PELVIS_FRAME
        pose.header.stamp = self._node.get_clock().now().to_msg()
        p = self._average_pelvis.translation
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, p)
        quat = pin.Quaternion(self._average_pelvis.rotation)
        quat.normalize()
        pose.pose.orientation.x = float(quat.x)
        pose.pose.orientation.y = float(quat.y)
        pose.pose.orientation.z = float(quat.z)
        pose.pose.orientation.w = float(quat.w)
        self._pose_pub.publish(pose)

    def _publish_overlay(
        self,
        frame: np.ndarray,
        source: CompressedImage,
        model: CameraModel,
        tag_camera: np.ndarray | None,
        tip_camera: np.ndarray | None,
        tip_camera_inverse: np.ndarray | None,
        publisher,
    ) -> None:
        tag_uv = None if tag_camera is None else project_camera_point(tag_camera, model)
        tip_uv = None if tip_camera is None else project_camera_point(tip_camera, model)
        tip_inverse_uv = (
            None
            if tip_camera_inverse is None
            else project_camera_point(tip_camera_inverse, model)
        )
        if model.frame == self._hand_frame:
            if tip_uv is not None:
                height, width = frame.shape[:2]
                self._logger.info(
                    f"[fk-calibration] drill_tip_link projected pixel: "
                    f"u={tip_uv[0]}, v={tip_uv[1]} (image {width}x{height}).",
                    throttle_duration_sec=2.0,
                )
                if tip_inverse_uv is not None:
                    delta = np.subtract(tip_inverse_uv, tip_uv)
                    self._logger.info(
                        "[fk-calibration] inverse-extrinsic drill-tip pixel: "
                        f"u={tip_inverse_uv[0]}, v={tip_inverse_uv[1]}; "
                        f"inverse - direct = du={delta[0]:+d}, "
                        f"dv={delta[1]:+d} px.",
                        throttle_duration_sec=2.0,
                    )
            else:
                self._logger.warning(
                    "[fk-calibration] drill_tip_link could not be projected into "
                    "the hand-camera image.",
                    throttle_duration_sec=2.0,
                )
        if tag_uv is not None:
            cv2.drawMarker(frame, tag_uv, (0, 0, 255), cv2.MARKER_CROSS, 34, 3)
            cv2.putText(
                frame,
                "AprilTag 0",
                (tag_uv[0] + 12, tag_uv[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        if tip_uv is not None:
            cv2.drawMarker(frame, tip_uv, (0, 255, 0), cv2.MARKER_CROSS, 34, 3)
            cv2.putText(
                frame,
                "drill_tip_link",
                (tip_uv[0] + 12, tip_uv[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        if tip_inverse_uv is not None:
            cv2.drawMarker(
                frame,
                tip_inverse_uv,
                (0, 165, 255),
                cv2.MARKER_TILTED_CROSS,
                34,
                3,
            )
            cv2.putText(
                frame,
                "drill_tip_link (inverse extrinsic)",
                (tip_inverse_uv[0] + 12, tip_inverse_uv[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
        if tip_uv is not None and tip_inverse_uv is not None:
            cv2.line(frame, tip_uv, tip_inverse_uv, (255, 255, 0), 2)
        try:
            out = self._bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpg")
        except CvBridgeError as exc:
            self._logger.error(
                f"Could not encode FK Calibration overlay: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        out.header = source.header
        out.format = "bgr8; jpeg compressed bgr8"
        publisher.publish(out)


__all__ = [
    "FkCalibrationHandler",
]
