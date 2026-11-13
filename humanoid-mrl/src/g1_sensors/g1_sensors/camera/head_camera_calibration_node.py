# MIT License
"""Calibrate the head-camera pose against hand-camera and mocap references.

Each sample records the drill tip's signed wall standoff (positive in front of
the wall) from the head tag, hand tag and mocap; samples and a summary are
written to the calibration YAML.
"""

import os
import select
import sys
import termios
import tty

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from g1_common.image_transport import subscribe_image
from g1_definitions import definitions, topics
from geometry_msgs.msg import PoseStamped
from pupil_apriltags import Detector
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener

from .head_camera_calibration import (
    CalibrationSession,
    corrected_mount_rotation,
    corrected_mount_translation,
    corrected_optical_rotation,
    corrected_optical_translation,
    measured_wall_rotation,
    normal_angle_deg,
    rotation_angle_deg,
    signed_plane_standoff,
    WALL_FROM_TAG,
)

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class TerminalKeys:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self._saved = termios.tcgetattr(self._fd) if self._fd is not None else None
        if self._fd is not None:
            tty.setcbreak(self._fd)

    @property
    def active(self) -> bool:
        return self._fd is not None

    def poll(self) -> str:
        keys = ""
        while self._fd is not None and select.select([sys.stdin], [], [], 0)[0]:
            keys += sys.stdin.read(1)
        return keys

    def close(self) -> None:
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None


class HeadCameraCalibrationNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("head_camera_calibration", **kwargs)
        self._read_parameters()
        self.get_logger().info(
            f"Loaded AprilTag id {self._tag_id} with size {self._tag_size_m} m"
        )

        self._detector = Detector(families=self._tag_family, nthreads=1)
        self._session = CalibrationSession()
        self._keys = TerminalKeys()
        self.finished = Future()
        self._camera_params = None
        self._maps = None
        self._hand_camera_params = None
        self._hand_maps = None
        self._last_sample_time = 0.0
        self._last_hand_sample_time = 0.0
        self._started_at = None
        self._no_tag = 0
        self._no_tf = 0
        self._latest_tf_samples = 0
        self._warned_latest_tf = False
        self._tag_range_m = []
        self._latest_hand_standoff = None
        self._standoff_samples: list[dict] = []
        self._hand_standoff_samples: list[dict] = []
        self._hand_frames = 0
        self._hand_no_camera_info = 0
        self._hand_no_tag = 0
        self._standoff_tf_failures = 0
        self._warned_standoff_tf = False
        self._reset_placement_diagnostics()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        bridge = CvBridge()
        self._image_sub = subscribe_image(
            self,
            self._image_topic,
            self._image_callback,
            SENSOR_QOS,
            encoding="mono8",
            bridge=bridge,
        )
        self._hand_image_sub = subscribe_image(
            self,
            self._hand_image_topic,
            self._hand_image_callback,
            SENSOR_QOS,
            encoding="mono8",
            bridge=bridge,
        )
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._camera_info_callback, SENSOR_QOS
        )
        self._hand_info_sub = self.create_subscription(
            CameraInfo,
            self._hand_info_topic,
            self._hand_camera_info_callback,
            SENSOR_QOS,
        )
        self._tag_pose_pub = self.create_publisher(
            PoseStamped, "head_camera_calibration/tag_pose", 1
        )
        self.create_timer(0.05, self._keyboard_timer)
        self.get_logger().info(
            "s: start placement, t: stop placement, e: finish "
            f"({self._wall_frame} -> {self._camera_frame})"
        )
        self.get_logger().info(
            f"Independent hand-tag standoff: '{self._hand_image_topic}' + "
            f"'{self._hand_info_topic}' in '{self._hand_camera_frame}' -> "
            f"'{self._tip_frame}'."
        )

    def _param(self, name, default):
        return self.declare_parameter(name, default).value

    def _read_parameters(self) -> None:
        self._image_topic = self._param("image_topic", topics.TOPIC_CAMERA)
        self._info_topic = self._param("camera_info_topic", topics.TOPIC_CAMERA_INFO)
        self._camera_frame = self._param(
            "camera_optical_frame", "d435_color_optical_frame"
        )
        self._hand_image_topic = self._param(
            "hand_image_topic", topics.TOPIC_CAMERA_WRIST + "/compressed"
        )
        self._hand_info_topic = self._param(
            "hand_camera_info_topic", topics.TOPIC_CAMERA_WRIST_INFO
        )
        self._hand_camera_frame = self._param(
            "hand_camera_optical_frame", "hand_camera_optical_frame"
        )
        self._wall_frame = self._param("wall_frame", "head_calibration_tag")
        self._mount_frame = self._param("camera_mount_frame", "d435_link")
        self._mount_parent_frame = self._param(
            "camera_mount_parent_frame", "torso_link"
        )
        self._tag_family = self._param("tag_family", "tag36h11")
        self._tag_id = int(self._param("tag_id", 0))
        self._tag_size_m = float(self._param("tag_size_m", 0.12))
        self._rate_hz = float(self._param("accumulate_rate_hz", 5.0))
        self._settle_s = float(self._param("settle_s", 1.0))
        self._output_path = self._param(
            "output_path", "~/.ros/head_camera_extrinsics.yaml"
        )
        wall_rpy = Rotation.from_matrix(WALL_FROM_TAG).as_euler("xyz", degrees=True)
        self._wall_from_tag = Rotation.from_euler(
            "xyz", self._param("wall_from_tag_rpy_deg", wall_rpy.tolist()), degrees=True
        ).as_matrix()
        expected = self._param("expected_rotation_rpy_deg", "")
        self._expected_rpy_deg = (
            [float(value) for value in expected.replace(",", " ").split()]
            if expected
            else None
        )
        self._tip_frame = self._param("drill_tip_frame", definitions.DRILL_TIP_FRAME)
        self._standoff_log_period_s = float(self._param("standoff_log_period_s", 1.0))
        self._hand_max_age_s = float(self._param("hand_standoff_max_age_s", 0.5))

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._camera_params = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
        distortion = np.asarray(msg.d)
        if distortion.size and np.any(distortion):
            k = np.asarray(msg.k).reshape(3, 3)
            self._maps = cv2.initUndistortRectifyMap(
                k, distortion, None, k, (msg.width, msg.height), cv2.CV_16SC2
            )
        self.destroy_subscription(self._info_sub)

    def _hand_camera_info_callback(self, msg: CameraInfo) -> None:
        """Adopt and, when necessary, rectify the hand camera's own model."""
        self._hand_camera_params = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
        distortion = np.asarray(msg.d)
        if distortion.size and np.any(distortion):
            k = np.asarray(msg.k).reshape(3, 3)
            self._hand_maps = cv2.initUndistortRectifyMap(
                k, distortion, None, k, (msg.width, msg.height), cv2.CV_16SC2
            )
        self.destroy_subscription(self._hand_info_sub)

    def _hand_image_callback(self, image: np.ndarray, msg) -> None:
        """Detect the wall tag from the drill-mounted camera."""
        if not self._session.accumulating:
            return
        self._hand_frames += 1
        self._placement_hand_frames += 1
        if self._hand_camera_params is None:
            self._hand_no_camera_info += 1
            self._placement_hand_no_camera_info += 1
            return
        now = self._seconds()
        if now - (self._started_at or now) < self._settle_s:
            return
        if now - self._last_hand_sample_time < 1 / self._rate_hz:
            return
        if self._hand_maps is not None:
            image = cv2.remap(image, *self._hand_maps, interpolation=cv2.INTER_LINEAR)
        detections = self._detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=self._hand_camera_params,
            tag_size=self._tag_size_m,
        )
        matches = [d for d in detections if d.tag_id == self._tag_id]
        if len(matches) != 1:
            self._hand_no_tag += 1
            self._placement_hand_no_tag += 1
            return
        if self._record_hand_standoff(matches[0], msg.header.stamp):
            self._last_hand_sample_time = now

    def _record_hand_standoff(self, detection, stamp) -> bool:
        """Measure tip-to-tag-plane range entirely through the hand camera TF."""
        tip_lookup = self._lookup_optional(
            self._hand_camera_frame, self._tip_frame, stamp
        )
        if tip_lookup is None:
            return False
        (_, tip_hand), used_latest = tip_lookup
        tag_point = np.asarray(detection.pose_t, dtype=float).reshape(3)
        hand_from_wall = measured_wall_rotation(detection.pose_R, self._wall_from_tag)
        tag_normal = -hand_from_wall[:, 0]
        standoff = signed_plane_standoff(tip_hand, tag_point, tag_normal)
        sample = {
            "placement": len(self._session.placements),
            "image_stamp_s": _stamp_seconds(stamp),
            "tag_range_m": float(np.linalg.norm(tag_point)),
            "hand_tag_standoff_m": standoff,
            "tip_hand_camera_xyz_m": tip_hand.tolist(),
            "tag_plane_point_hand_camera_m": tag_point.tolist(),
            "tag_normal_hand_camera": tag_normal.tolist(),
            "tip_tf_used_latest": bool(used_latest),
        }
        self._hand_standoff_samples.append(sample)
        self._placement_hand_samples += 1
        self._latest_hand_standoff = (sample, self._seconds())
        self.get_logger().info(
            f"[hand-standoff] tag {standoff * 1000.0:+.1f} mm | "
            f"camera range {np.linalg.norm(tag_point) * 1000.0:.1f} mm",
            throttle_duration_sec=self._standoff_log_period_s,
        )
        return True

    def _image_callback(self, image: np.ndarray, msg) -> None:
        if not self._session.accumulating:
            return
        self._placement_frames += 1
        if self._camera_params is None:
            self._placement_no_camera_info += 1
            return
        now = self._seconds()
        if now - (self._started_at or now) < self._settle_s:
            self._placement_settling += 1
            return
        if now - self._last_sample_time < 1 / self._rate_hz:
            self._placement_rate_limited += 1
            return
        if self._maps is not None:
            image = cv2.remap(image, *self._maps, interpolation=cv2.INTER_LINEAR)
        p05, p95 = np.percentile(image, [5, 95])
        self._last_image_levels = (float(np.mean(image)), float(p05), float(p95))
        detections = self._detector.detect(
            image,
            estimate_tag_pose=True,
            camera_params=self._camera_params,
            tag_size=self._tag_size_m,
        )
        self._placement_detection_attempts += 1
        self._placement_detected_ids.update(int(d.tag_id) for d in detections)
        matches = [d for d in detections if d.tag_id == self._tag_id]
        if len(matches) != 1:
            self._no_tag += 1
            self._placement_no_tag += 1
            return
        latest_before = self._latest_tf_samples
        predicted_transform = self._lookup(msg.header.stamp)
        if predicted_transform is None:
            self._no_tf += 1
            self._placement_no_tf += 1
            return
        detection = matches[0]
        predicted_rotation, predicted_translation = predicted_transform
        self._session.add(
            measured_wall_rotation(detection.pose_R, self._wall_from_tag),
            predicted_rotation,
            detection.pose_t,
            predicted_translation,
        )
        self._last_sample_time = now
        self._tag_range_m.append(float(np.linalg.norm(detection.pose_t)))
        self._record_standoff_sample(
            detection,
            predicted_rotation,
            predicted_translation,
            msg.header.stamp,
            mocap_used_latest=self._latest_tf_samples > latest_before,
        )
        self._publish_tag_pose(detection, msg.header)

    def _lookup_optional(self, parent: str, child: str, stamp):
        """A diagnostic TF at the image stamp, degrading to latest if needed."""
        try:
            tf = self._tf_buffer.lookup_transform(parent, child, stamp)
            return _rotation_translation(tf), False
        except TransformException:
            try:
                tf = self._tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
                return _rotation_translation(tf), True
            except TransformException:
                self._standoff_tf_failures += 1
                if not self._warned_standoff_tf:
                    self._warned_standoff_tf = True
                    self.get_logger().warn(
                        f"Standoff diagnostics need TF '{parent}' <- '{child}'; "
                        "calibration continues, but distance samples will be "
                        "incomplete. Warned once."
                    )
                return None

    def _record_standoff_sample(
        self,
        detection,
        predicted_rotation,
        predicted_translation,
        stamp,
        *,
        mocap_used_latest: bool = False,
    ) -> None:
        """Record like-for-like normal-axis ranges for head, hand and mocap."""
        tip_lookup = self._lookup_optional(self._camera_frame, self._tip_frame, stamp)
        if tip_lookup is None:
            return
        (camera_from_tip, tip_camera), tip_used_latest = tip_lookup
        del camera_from_tip  # Only the tip origin matters for point-to-plane range.

        tag_point = np.asarray(detection.pose_t, dtype=float).reshape(3)
        camera_from_tag_wall = measured_wall_rotation(
            detection.pose_R, self._wall_from_tag
        )
        # The wall frame's +x points into the plate; standoff uses the opposite,
        # outward direction so a clear tip reads positive.
        tag_normal = -camera_from_tag_wall[:, 0]
        mocap_point = np.asarray(predicted_translation, dtype=float).reshape(3)
        mocap_normal = -np.asarray(predicted_rotation, dtype=float)[:, 0]

        tag_standoff = signed_plane_standoff(tip_camera, tag_point, tag_normal)
        mocap_standoff = signed_plane_standoff(tip_camera, mocap_point, mocap_normal)
        sample = {
            "placement": len(self._session.placements),
            "image_stamp_s": _stamp_seconds(stamp),
            "tag_range_m": float(np.linalg.norm(tag_point)),
            "tip_camera_xyz_m": tip_camera.tolist(),
            "tag_plane_point_camera_m": tag_point.tolist(),
            "mocap_plane_point_camera_m": mocap_point.tolist(),
            "tag_standoff_m": tag_standoff,
            "mocap_standoff_m": mocap_standoff,
            "tag_minus_mocap_m": tag_standoff - mocap_standoff,
            "tag_plane_at_mocap_origin_m": signed_plane_standoff(
                mocap_point, tag_point, tag_normal
            ),
            "tag_mocap_normal_angle_deg": normal_angle_deg(tag_normal, mocap_normal),
            "tag_normal_camera": tag_normal.tolist(),
            "mocap_normal_camera": mocap_normal.tolist(),
            "tip_tf_used_latest": bool(tip_used_latest),
            "mocap_tf_used_latest": bool(mocap_used_latest),
            "hand_tag_standoff_m": None,
            "hand_minus_head_tag_m": None,
            "hand_minus_mocap_m": None,
            "hand_standoff_age_s": None,
            "hand_source_stamp_s": None,
            "hand_tip_tf_used_latest": None,
        }

        if self._latest_hand_standoff is not None:
            hand, received_s = self._latest_hand_standoff
            hand_age = max(0.0, self._seconds() - received_s)
            if hand_age <= self._hand_max_age_s:
                hand_standoff = hand["hand_tag_standoff_m"]
                sample.update(
                    {
                        "hand_tag_standoff_m": hand_standoff,
                        "hand_minus_head_tag_m": hand_standoff - tag_standoff,
                        "hand_minus_mocap_m": hand_standoff - mocap_standoff,
                        "hand_standoff_age_s": hand_age,
                        "hand_source_stamp_s": hand["image_stamp_s"],
                        "hand_tip_tf_used_latest": hand["tip_tf_used_latest"],
                    }
                )

        self._standoff_samples.append(sample)
        self._log_standoff_sample(sample)

    def _log_standoff_sample(self, sample: dict) -> None:
        def millimetres(key: str) -> str:
            value = sample[key]
            return "missing" if value is None else f"{value * 1000.0:+.1f} mm"

        self.get_logger().info(
            f"[standoff] head {millimetres('tag_standoff_m')} | "
            f"hand {millimetres('hand_tag_standoff_m')} | "
            f"mocap {millimetres('mocap_standoff_m')} | "
            f"head-mocap {millimetres('tag_minus_mocap_m')}",
            throttle_duration_sec=self._standoff_log_period_s,
        )

    def _lookup(self, stamp) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._camera_frame, self._wall_frame, stamp
            )
        except TransformException as exact_error:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._camera_frame, self._wall_frame, rclpy.time.Time()
                )
            except TransformException as latest_error:
                self._last_tf_error = (
                    f"image stamp: {exact_error}; latest transform: {latest_error}"
                )
                return None
            self._latest_tf_samples += 1
            self._warned_latest_tf = True
        q = tf.transform.rotation
        t = tf.transform.translation
        return (
            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
            np.array([t.x, t.y, t.z]),
        )

    def _publish_tag_pose(self, detection, header) -> None:
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(
            float, np.asarray(detection.pose_t).reshape(3)
        )
        quat = Rotation.from_matrix(detection.pose_R).as_quat()
        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = map(float, quat)
        self._tag_pose_pub.publish(pose)

    def _keyboard_timer(self) -> None:
        for key in self._keys.poll():
            self._command(key)

    def _command(self, key: str) -> None:
        if self.finished.done():
            return
        if key.lower() == "s":
            if self._session.accumulating:
                return
            self._reset_placement_diagnostics()
            self._session.start()
            self._started_at = self._seconds()
            self.get_logger().info("Collecting calibration samples")
        elif key.lower() == "t":
            samples = (
                len(self._session.placements[-1]) if self._session.accumulating else 0
            )
            self._session.pause()
            self._log_placement_diagnostics(samples)
        elif key.lower() == "e":
            self._finish()

    def _seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _reset_placement_diagnostics(self) -> None:
        self._placement_frames = 0
        self._placement_no_camera_info = 0
        self._placement_settling = 0
        self._placement_rate_limited = 0
        self._placement_detection_attempts = 0
        self._placement_no_tag = 0
        self._placement_no_tf = 0
        self._placement_detected_ids: set[int] = set()
        self._last_image_levels = None
        self._last_tf_error = None
        self._placement_hand_frames = 0
        self._placement_hand_no_camera_info = 0
        self._placement_hand_no_tag = 0
        self._placement_hand_samples = 0

    def _log_placement_diagnostics(self, samples: int) -> None:
        self.get_logger().info(
            f"Placement stopped: {samples} samples; "
            f"images={self._placement_frames}, "
            f"no_camera_info={self._placement_no_camera_info}, "
            f"settling={self._placement_settling}, "
            f"rate_limited={self._placement_rate_limited}, "
            f"detection_attempts={self._placement_detection_attempts}, "
            f"no_matching_tag={self._placement_no_tag}, "
            f"no_tf={self._placement_no_tf}; "
            f"hand_samples={self._placement_hand_samples}, "
            f"hand_images={self._placement_hand_frames}, "
            f"hand_no_camera_info={self._placement_hand_no_camera_info}, "
            f"hand_no_matching_tag={self._placement_hand_no_tag}"
        )
        if self._last_image_levels is not None:
            mean, p05, p95 = self._last_image_levels
            detected_ids = sorted(self._placement_detected_ids)
            self.get_logger().info(
                f"AprilTag diagnostics: detected_ids={detected_ids}, "
                f"last_image_gray mean={mean:.1f}/255, p05={p05:.1f}, "
                f"p95={p95:.1f}, contrast={p95 - p05:.1f}"
            )
        if self._placement_frames == 0:
            self.get_logger().warn(f"No images received on '{self._image_topic}'")
        elif self._placement_no_camera_info == self._placement_frames:
            self.get_logger().warn(f"No camera info received on '{self._info_topic}'")
        if self._last_tf_error is not None:
            self.get_logger().warn(f"Last TF failure: {self._last_tf_error}")
        if self._placement_hand_frames == 0:
            self.get_logger().warn(
                f"No hand-camera images received on '{self._hand_image_topic}'"
            )
        elif self._placement_hand_no_camera_info == self._placement_hand_frames:
            self.get_logger().warn(
                f"No hand-camera info received on '{self._hand_info_topic}'"
            )
        elif self._placement_hand_samples == 0:
            self.get_logger().warn(
                f"Hand camera saw no matching AprilTag id {self._tag_id}"
            )

    def _tf_transform(
        self, parent: str, child: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf = self._tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
        except TransformException:
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        return (
            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
            np.array([t.x, t.y, t.z]),
        )

    def _corrections(
        self, correction_rotation: np.ndarray, correction_translation: np.ndarray
    ) -> dict | None:
        optical_transform = self._tf_transform(self._mount_frame, self._camera_frame)
        if optical_transform is None:
            return None
        optical_rotation, optical_translation = optical_transform
        result = {
            "sim_optical_frame_parent": self._mount_frame,
            "sim_optical_frame_xyz": corrected_optical_translation(
                optical_rotation,
                optical_translation,
                correction_rotation,
                correction_translation,
            ).tolist(),
            "sim_optical_frame_quat_xyzw": Rotation.from_matrix(
                corrected_optical_rotation(optical_rotation, correction_rotation)
            )
            .as_quat()
            .tolist(),
        }
        mount_transform = self._tf_transform(
            self._mount_parent_frame, self._mount_frame
        )
        if mount_transform is not None:
            mount_rotation, mount_translation = mount_transform
            result["urdf_mount_joint"] = {
                "parent": self._mount_parent_frame,
                "child": self._mount_frame,
                "xyz": corrected_mount_translation(
                    mount_rotation,
                    mount_translation,
                    optical_rotation,
                    optical_translation,
                    correction_rotation,
                    correction_translation,
                ).tolist(),
                "rpy": Rotation.from_matrix(
                    corrected_mount_rotation(
                        mount_rotation, optical_rotation, correction_rotation
                    )
                )
                .as_euler("xyz")
                .tolist(),
            }
        return result

    def _finish(self) -> None:
        solution = self._session.solve()
        if solution is None:
            self.get_logger().error("No calibration samples collected")
            self.finished.set_result(False)
            return
        placements = self._session.placement_solutions()
        translation = solution.translation
        if translation is None:
            self.get_logger().error("Calibration samples have no translation data")
            self.finished.set_result(False)
            return
        report = {
            "samples": solution.samples,
            "placements": [
                {
                    "samples": p.samples,
                    "rpy_deg": Rotation.from_matrix(p.rotation)
                    .as_euler("xyz", degrees=True)
                    .tolist(),
                    "mean_residual_deg": p.mean_residual_deg,
                    "t_hh_prime_m": p.translation.tolist(),
                    "mean_translation_residual_m": p.mean_translation_residual_m,
                }
                for p in placements
            ],
            "r_hh_prime": {
                "rpy_deg": Rotation.from_matrix(solution.rotation)
                .as_euler("xyz", degrees=True)
                .tolist(),
                "quat_xyzw": Rotation.from_matrix(solution.rotation).as_quat().tolist(),
                "matrix": solution.rotation.tolist(),
                "angle_deg": solution.angle_deg,
            },
            "t_hh_prime_m": {
                "xyz": translation.tolist(),
                "norm": float(np.linalg.norm(translation)),
            },
            "fit": {
                "mean_residual_deg": solution.mean_residual_deg,
                "max_residual_deg": solution.max_residual_deg,
                "consistency": solution.consistency,
                "placement_spread_deg": self._session.spread_deg(),
                "mean_translation_residual_m": solution.mean_translation_residual_m,
                "max_translation_residual_m": solution.max_translation_residual_m,
            },
            "standoff_definition": (
                "signed perpendicular point-to-plane distance; positive in front "
                "of the wall, negative beyond it"
            ),
            "standoff_summary": _standoff_summary(self._standoff_samples),
            "standoff_samples": self._standoff_samples,
            "hand_tag_summary": _hand_standoff_summary(self._hand_standoff_samples),
            "hand_tag_samples": self._hand_standoff_samples,
            "hand_tag_diagnostics": {
                "frames": self._hand_frames,
                "no_camera_info": self._hand_no_camera_info,
                "no_matching_tag": self._hand_no_tag,
                "image_topic": self._hand_image_topic,
                "camera_info_topic": self._hand_info_topic,
                "camera_frame": self._hand_camera_frame,
                "tip_frame": self._tip_frame,
            },
            "corrections": self._corrections(solution.rotation, translation),
        }
        if self._expected_rpy_deg is not None:
            expected = Rotation.from_euler(
                "xyz", self._expected_rpy_deg, degrees=True
            ).as_matrix()
            report["vs_expected"] = {
                "error_deg": rotation_angle_deg(solution.rotation, expected)
            }
        rpy = report["r_hh_prime"]["rpy_deg"]
        self.get_logger().info(
            f"Calibration: rpy={rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f} deg, "
            f"xyz={translation[0]:.6f} {translation[1]:.6f} "
            f"{translation[2]:.6f} m, residual={solution.mean_residual_deg:.4f} deg, "
            f"{solution.mean_translation_residual_m:.6f} m"
        )
        if self._output_path:
            path = os.path.expanduser(self._output_path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as handle:
                yaml.safe_dump(report, handle, sort_keys=False)
            self.get_logger().info(f"Wrote {path}")
        self.finished.set_result(True)

    def destroy_node(self) -> bool:
        self._keys.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeadCameraCalibrationNode()
    try:
        rclpy.spin_until_future_complete(node, node.finished)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _rotation_translation(tf) -> tuple[np.ndarray, np.ndarray]:
    q = tf.transform.rotation
    t = tf.transform.translation
    return (
        Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
        np.array([t.x, t.y, t.z]),
    )


def _standoff_summary(samples: list[dict]) -> dict:
    """Compact statistics while retaining every sample for later regression."""
    fields = (
        "tag_standoff_m",
        "mocap_standoff_m",
        "tag_minus_mocap_m",
        "tag_mocap_normal_angle_deg",
        "hand_tag_standoff_m",
        "hand_minus_head_tag_m",
        "hand_minus_mocap_m",
    )
    result = {"samples": len(samples)}
    for field in fields:
        values = np.asarray(
            [sample[field] for sample in samples if sample.get(field) is not None],
            dtype=float,
        )
        result[field] = {
            "count": int(len(values)),
            "mean": None if not len(values) else float(np.mean(values)),
            "std": None if not len(values) else float(np.std(values)),
            "min": None if not len(values) else float(np.min(values)),
            "max": None if not len(values) else float(np.max(values)),
        }
    return result


def _hand_standoff_summary(samples: list[dict]) -> dict:
    values = np.asarray(
        [sample["hand_tag_standoff_m"] for sample in samples], dtype=float
    )
    return {
        "samples": int(len(values)),
        "mean_m": None if not len(values) else float(np.mean(values)),
        "std_m": None if not len(values) else float(np.std(values)),
        "min_m": None if not len(values) else float(np.min(values)),
        "max_m": None if not len(values) else float(np.max(values)),
    }


if __name__ == "__main__":
    main()
