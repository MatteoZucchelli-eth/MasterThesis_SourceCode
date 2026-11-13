# MIT License
"""Camera model for offline label projection.

``wrist``
    The hand camera, 720x720, carrying the See3CAM intrinsics the sim was
    matched to. The only view labels are projected into.

Projection is in the optical frame (+Z forward, +X right, +Y down); the wrist
URDF frame is already optical. Validated by overlay_check.py (~1 px).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """The camera's geometry, as it reaches the dataset (post-crop)."""

    name: str
    video_key: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame: str


SEE3CAM_CROP = Calibration(
    name="see3cam_crop",
    video_key="observation.images.wrist_view",
    # 1280x720 See3CAM intrinsics centre-cropped to 720x720: cx shifts by 280, cy
    # unchanged.
    fx=690.1127251872208,
    fy=690.1127251872208,
    cx=359.83226287869536,
    cy=369.7137298773904,
    width=720,
    height=720,
    frame="hand_camera_optical_frame",
)

CAMERA = SEE3CAM_CROP

IMAGE_WIDTH = CAMERA.width
IMAGE_HEIGHT = CAMERA.height
FX = CAMERA.fx
FY = CAMERA.fy
CX = CAMERA.cx
CY = CAMERA.cy
CAMERA_FRAME = CAMERA.frame
"""URDF/TF frame the projection happens in (optical convention)."""

VIDEO_KEY = CAMERA.video_key
"""Which of the dataset's videos this calibration describes."""

K = np.array(
    [
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ]
)


def describe() -> str:
    """One line naming the camera — echoed by every CLI at startup."""
    return (
        f"wrist camera: {CAMERA.name} (fx={CAMERA.fx:.2f} fy={CAMERA.fy:.2f} "
        f"cx={CAMERA.cx:.2f} cy={CAMERA.cy:.2f}, {CAMERA.width}x{CAMERA.height}, "
        f"frame {CAMERA.frame}, video {CAMERA.video_key})"
    )


def project(points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project Nx3 optical-frame points to Nx2 pixel coords.

    Returns ``(uv, in_front)``; rows with non-positive depth get NaN pixels
    and ``in_front=False`` instead of a bogus mirror-image projection.
    """
    points_cam = np.atleast_2d(points_cam)
    z = points_cam[:, 2]
    in_front = z > 1e-6
    uv = np.full((points_cam.shape[0], 2), np.nan)
    uv[in_front, 0] = FX * points_cam[in_front, 0] / z[in_front] + CX
    uv[in_front, 1] = FY * points_cam[in_front, 1] / z[in_front] + CY
    return uv, in_front


def in_image(uv: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
    """Boolean mask of pixel rows lying inside the image with a margin."""
    uv = np.atleast_2d(uv)
    ok = ~np.isnan(uv).any(axis=1)
    ok &= (uv[:, 0] >= margin_px) & (uv[:, 0] <= IMAGE_WIDTH - 1 - margin_px)
    ok &= (uv[:, 1] >= margin_px) & (uv[:, 1] <= IMAGE_HEIGHT - 1 - margin_px)
    return ok
