# MIT License
"""Check whether the episode's target square is in shot.

AUTO_COLLECT skips a configuration unless the target projects inside the
prepared (squared, rectified) frame of at least one recorded camera.

- Lens: each camera's ``CameraInfo`` via the :mod:`g1_common.camera` preparer's
  intrinsics (cropped frame, not the sensor K).
- Camera pose: FK over the controller's URDF, not TF (the headless collection
  worker has no robot_state_publisher). Matches the offline label projector.
- The head is projected from ``d435_link`` with :data:`BODY_TO_OPTICAL`; on the
  robot realsense2_camera adds a factory-calibration hop (~15 mm, 0.6 deg,
  ~11 px at 2 m), fine for an in-shot test but not for metric use.
- Field of view only; occlusion is not modelled.
- Pinhole math is duplicated from ``g1_control.vla.perception`` to avoid
  importing the gr00t client.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from g1_common.camera import PinholeIntrinsics, preparer_from_camera_info
from g1_definitions import definitions
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


BODY_TO_OPTICAL = _rz(-np.pi / 2) @ _rx(-np.pi / 2)
"""rpy (-pi/2, 0, -pi/2): ROS body -> optical change of basis.

Equals the published ``d435_link -> d435_color_optical_frame`` rotation; that
frame is not in the URDF."""

_MIN_DEPTH_M = 1e-6
"""Below this a point is at or behind the image plane, and has no pixel."""


@dataclass(frozen=True, eq=False)
class RecordedCamera:
    """One camera an episode is recorded from: its stream and its URDF mount.

    ``eq=False``: the numpy field would make the generated ``__eq__`` elementwise.
    """

    view: str
    info_topic: str
    urdf_frame: str
    optical_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    """Rotation taking :attr:`urdf_frame` to the optical frame K assumes."""

    def optical_placement(self, robot_model) -> pin.SE3:
        """Pelvis-frame pose of this camera's optical frame, from the cached FK."""
        placement = robot_model.frame_placement(self.urdf_frame)
        return pin.SE3(
            placement.rotation @ self.optical_rotation, placement.translation
        )


RECORDED_CAMERAS: tuple[RecordedCamera, ...] = (
    RecordedCamera(
        view="ego_view",
        info_topic=definitions.TOPIC_CAMERA_INFO,
        # Body frame; the optical frame is not in the URDF.
        urdf_frame="d435_link",
        optical_rotation=BODY_TO_OPTICAL,
    ),
    RecordedCamera(
        view="wrist_view",
        info_topic=definitions.TOPIC_CAMERA_WRIST_INFO,
        # Already optical in the drill URDF.
        urdf_frame="hand_camera_optical_frame",
    ),
)
"""Recorded cameras, keyed by GR00T video modality.

Must match g1_teleoperation ``recorder_node.CAMERAS`` / ``CAMERA_INFOS``
(restated: g1_control does not depend on g1_teleoperation)."""


@dataclass(frozen=True)
class CameraView:
    """A recorded camera whose stream has described its lens: ready to project."""

    camera: RecordedCamera
    intrinsics: PinholeIntrinsics
    declared_frame: str
    """``CameraInfo.header.frame_id``; logged only (pose comes from FK)."""

    def sees(self, t_cam_target: pin.SE3, margin_m: float = 0.0) -> "ViewCheck":
        """Where the target lands in this view, and whether that counts as in shot.

        ``t_cam_target`` is the target's pose in this camera's optical frame.
        The border margin is metric, converted at the target depth as
        ``margin_px = fx * margin_m / depth_m``. Points behind the lens are
        never visible.
        """
        point = np.asarray(t_cam_target.translation, dtype=float).reshape(3)
        depth = float(point[2])
        if depth <= _MIN_DEPTH_M:
            return ViewCheck(self.camera.view, np.full(2, np.nan), depth, False)

        intr = self.intrinsics
        uv = np.array(
            [
                intr.fx * point[0] / depth + intr.cx,
                intr.fy * point[1] / depth + intr.cy,
            ]
        )
        margin_px = intr.fx * float(margin_m) / depth
        inside = (
            margin_px <= uv[0] <= intr.width - 1 - margin_px
            and margin_px <= uv[1] <= intr.height - 1 - margin_px
        )
        return ViewCheck(self.camera.view, uv, depth, bool(inside))


@dataclass(frozen=True)
class ViewCheck:
    """Result of projecting the target into one view."""

    view: str
    uv_px: np.ndarray
    depth_m: float
    visible: bool

    def describe(self) -> str:
        """One phrase for the log: where it landed and whether that counts."""
        if self.depth_m <= _MIN_DEPTH_M:
            return f"{self.view} behind the camera ({self.depth_m:+.2f} m)"
        return (
            f"{self.view} ({self.uv_px[0]:.0f}, {self.uv_px[1]:.0f}) px at "
            f"{self.depth_m:.2f} m {'IN' if self.visible else 'OUT'}"
        )


@dataclass(frozen=True)
class VisibilityReport:
    """Per-view results of one check, plus the verdict over all of them."""

    checks: tuple[ViewCheck, ...]

    @property
    def visible(self) -> bool:
        """Whether at least one recorded camera has the target in frame."""
        return any(check.visible for check in self.checks)

    @property
    def summary(self) -> str:
        return "; ".join(check.describe() for check in self.checks)


class TargetVisibility:
    """Projects the current target into every recorded camera, on demand.

    Owns one ``CameraInfo`` subscription per camera. Cameras whose mount frame
    is missing from ``robot_model`` are dropped with a warning; :attr:`cameras`
    may end up empty.
    """

    def __init__(self, node, robot_model, cameras=RECORDED_CAMERAS) -> None:
        self._node = node
        self._robot_model = robot_model
        usable, dropped = [], []
        for camera in cameras:
            (usable if robot_model.has_frame(camera.urdf_frame) else dropped).append(
                camera
            )
        self.cameras: tuple[RecordedCamera, ...] = tuple(usable)
        for camera in dropped:
            node.get_logger().warning(
                f"Target visibility: no '{camera.urdf_frame}' frame in the "
                f"loaded URDF, so {camera.view} cannot be checked."
            )
        self._views: dict[str, CameraView] = {}

        # Latest-only, best-effort, like the other consumers.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._subs = {
            camera.view: node.create_subscription(
                CameraInfo, camera.info_topic, self._make_callback(camera), qos
            )
            for camera in self.cameras
        }

    def _make_callback(self, camera: RecordedCamera):
        """Return the callback that adopts ``camera``'s lens from the first message."""

        def _cb(msg: CameraInfo) -> None:
            if camera.view in self._views:
                return
            side = min(int(msg.width), int(msg.height))
            try:
                preparer = preparer_from_camera_info(msg, side)
            except ValueError as exc:
                # Unsupported lens: stay unresolved so AUTO_COLLECT waits.
                self._node.get_logger().error(
                    f"Target visibility: cannot prepare {camera.view} frames "
                    f"from {camera.info_topic}: {exc}",
                    throttle_duration_sec=10.0,
                )
                return
            self._views[camera.view] = CameraView(
                camera=camera,
                intrinsics=preparer.intrinsics,
                declared_frame=msg.header.frame_id,
            )
            self._node.get_logger().info(
                f"Target visibility: {camera.view} {msg.width}x{msg.height} -> "
                f"{side}x{side}, declared '{msg.header.frame_id}', projected "
                f"from '{camera.urdf_frame}' by FK."
            )

        return _cb

    def pending(self) -> list[str]:
        """Cameras that have not described their lens yet."""
        return [c.view for c in self.cameras if c.view not in self._views]

    def check(
        self, q: np.ndarray, t_pelvis_target: pin.SE3, margin_m: float = 0.0
    ) -> VisibilityReport | None:
        """Project a pelvis-frame target into every camera, or None if it cannot yet.

        ``q`` is the full (measured) configuration. None means undecidable
        (some camera has no ``CameraInfo`` yet); the caller should wait.
        """
        if not self.cameras or self.pending():
            return None
        # One unclipped FK pass for all cameras.
        self._robot_model.cache_forward_kinematics(q, auto_clip=False)
        checks = []
        for camera in self.cameras:
            view = self._views[camera.view]
            t_cam_target = camera.optical_placement(self._robot_model).inverse() * (
                t_pelvis_target
            )
            checks.append(view.sees(t_cam_target, margin_m))
        return VisibilityReport(checks=tuple(checks))

    def waiting_on(self) -> str:
        """Why the last :meth:`check` was undecidable, for the log."""
        if not self.cameras:
            return "no camera has a mount frame in the loaded URDF"
        return "no CameraInfo yet on " + ", ".join(
            camera.info_topic
            for camera in self.cameras
            if camera.view in self.pending()
        )
