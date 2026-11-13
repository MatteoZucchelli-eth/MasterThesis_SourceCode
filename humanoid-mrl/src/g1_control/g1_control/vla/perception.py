# MIT License
"""Geometry for the close-range perception model: 2D pixel + depth -> 3D pose.

The perception head is a monocular target detector on the wrist ("hand")
camera. Per inference it emits:

  ``target_uv_px``           (2,) target centre pixel in the wrist image
  ``target_uv_processed_px`` (2,) same, in the model's internal processed frame
  ``visibility``             (1,) detection confidence 0..1
  ``visible``                (1,) boolean gate (visibility past the threshold)
  ``target_depth_m``         (1,) camera->target distance along the optical axis
  ``wall_normal_cam``        (3,) wall normal (drill axis) in the camera frame

The detector-only model returns these as its action dict; a combined model
returns them in the info dict and has no ``visible`` head. ``target_uv_px`` is
in the *prepared* (squared, rectified) wrist frame, so back-projection uses the
preparer's intrinsics, not the raw CameraInfo.

Converts ``(u, v, depth, wall_normal)`` into a pose in
``hand_camera_optical_frame`` (+z forward, +x right, +y down). Pure numpy; the
client composes it with the TF extrinsic to ``drill_tip_link``.

9D pose layout (matching ``RobotModel.se3_to_9d_pose``): xyz(3) + rot6d(6), where
rot6d is the first two columns of the rotation matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tip_estimate import mat_to_pose_9d


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the prepared hand-camera frame, plus its size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_k(cls, k, width: int, height: int) -> "CameraIntrinsics":
        """Build from a row-major 3x3 ``camera_info.k`` (9 floats) and a size."""
        k = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            width=int(width),
            height=int(height),
        )


@dataclass(frozen=True)
class PerceptionDetection:
    """Parsed output of one perception inference (camera-frame quantities)."""

    uv_px: np.ndarray  # (2,) target centre pixel in the *prepared* wrist frame
    visibility: float  # 0..1 confidence
    visible: bool  # model's own boolean gate
    depth_m: float  # camera->target distance along the optical axis
    wall_normal_cam: np.ndarray  # (3,) drill/wall axis in the camera optical frame

    #: Keys a model must emit for its perception head to be usable (``visible``
    #optional).
    REQUIRED_KEYS = (
        "target_uv_px",
        "visibility",
        "target_depth_m",
        "wall_normal_cam",
    )

    @classmethod
    def present_in(cls, outputs: dict | None) -> bool:
        """Whether ``outputs`` carries a perception head this class can parse."""
        return isinstance(outputs, dict) and all(
            key in outputs for key in cls.REQUIRED_KEYS
        )

    @classmethod
    def from_reply(cls, action_dict: dict | None, info: dict | None):
        """Find the perception head in the action dict or info of a ``get_action``
        reply.

        Returns None when neither carries a perception head.
        """
        for outputs in (action_dict, info):
            if cls.present_in(outputs):
                return cls.from_outputs(outputs)
        return None

    @classmethod
    def from_outputs(cls, outputs: dict) -> "PerceptionDetection":
        """Parse one model's perception outputs, squeezing any batch dims.

        ``visible`` defaults to True when absent and is informational only;
        gating uses ``visibility`` (see ``TipTargetLifter.lift``).
        """

        def flat(key: str) -> np.ndarray:
            return np.asarray(outputs[key], dtype=float).reshape(-1)

        return cls(
            uv_px=flat("target_uv_px")[:2],
            visibility=float(flat("visibility")[0]),
            visible=bool(flat("visible")[0]) if "visible" in outputs else True,
            depth_m=float(flat("target_depth_m")[0]),
            wall_normal_cam=flat("wall_normal_cam")[:3],
        )


def backproject_uv(
    uv_px: np.ndarray, depth_m: float, intr: CameraIntrinsics
) -> np.ndarray:
    """Pinhole back-projection of a pixel at a given optical-axis depth.

    Returns the 3D point in the camera optical frame (+z forward). ``depth_m`` is
    the model's ``target_depth_m`` (the z coordinate), not the ray length.
    """
    u, v = float(uv_px[0]), float(uv_px[1])
    z = float(depth_m)
    return np.array(
        [(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], dtype=float
    )


def project_point(point_cam: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Pinhole projection of a camera-frame point to a pixel.

    ``z <= 0`` still yields finite pixels; callers must check depth.
    """
    x, y, z = (float(c) for c in np.asarray(point_cam, dtype=float).reshape(3))
    safe_z = z if abs(z) > 1e-9 else 1e-9
    return np.array(
        [intr.fx * x / safe_z + intr.cx, intr.fy * y / safe_z + intr.cy], dtype=float
    )


@dataclass(frozen=True)
class FrameProjection:
    """Where a 3D point falls in the prepared wrist frame, and how far away."""

    uv_px: np.ndarray
    depth_m: float
    intrinsics: CameraIntrinsics

    @classmethod
    def of(cls, point_cam: np.ndarray, intr: CameraIntrinsics) -> "FrameProjection":
        point_cam = np.asarray(point_cam, dtype=float).reshape(3)
        return cls(project_point(point_cam, intr), float(point_cam[2]), intr)

    def inside(self, margin_px: float = 0.0) -> bool:
        """Whether the pixel sits inside the frame, clear of a border.

        Ignores depth; callers gate on :attr:`depth_m` separately.
        """
        u, v = float(self.uv_px[0]), float(self.uv_px[1])
        margin = float(margin_px)
        return (
            margin <= u <= self.intrinsics.width - 1 - margin
            and margin <= v <= self.intrinsics.height - 1 - margin
        )


def _rotation_from_x_axis(x_axis: np.ndarray) -> np.ndarray:
    """Orthonormal rotation whose x-axis is ``x_axis`` (the drill/wall normal).

    Roll is free: y is Gram-Schmidt from camera +y (or +x if nearly parallel).
    The finalize planner uses only the x column.
    """
    x = np.asarray(x_axis, dtype=float)
    n = np.linalg.norm(x)
    x = x / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    reference = np.array([0.0, 1.0, 0.0])  # camera +y (down)
    if abs(float(x @ reference)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    y = reference - float(x @ reference) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def target_pose_in_camera(
    detection: PerceptionDetection, intr: CameraIntrinsics
) -> np.ndarray:
    """4x4 target pose in the camera optical frame from one detection.

    Position: back-projected pixel at the predicted depth; x axis: wall normal.
    """
    mat = np.eye(4)
    mat[:3, :3] = _rotation_from_x_axis(detection.wall_normal_cam)
    mat[:3, 3] = backproject_uv(detection.uv_px, detection.depth_m, intr)
    return mat


def target_pose_9d_in_camera(
    detection: PerceptionDetection, intr: CameraIntrinsics
) -> np.ndarray:
    """9D (xyz + rot6d) target pose in the camera optical frame."""
    return mat_to_pose_9d(target_pose_in_camera(detection, intr))
