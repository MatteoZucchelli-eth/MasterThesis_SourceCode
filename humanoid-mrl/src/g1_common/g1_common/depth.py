# MIT License
"""Depth image to point cloud deprojection, pure numpy.

Intrinsics come from :mod:`g1_common.camera`; output feeds :mod:`g1_common.plane`.
No rectification: every depth producer here (Isaac, realsense
``depth/image_rect_raw``) is already rectified.
"""

from __future__ import annotations

import numpy as np

from .camera import PinholeIntrinsics

MILLIMETRES_PER_METRE = 1000.0

#: Metres per unit for each depth encoding: 16UC1 is realsense Z16 (mm),
#: 32FC1 is Isaac / depth_image_proc (m).
DEPTH_SCALES: dict[str, float] = {
    "32FC1": 1.0,
    "16UC1": 1.0 / MILLIMETRES_PER_METRE,
}


def depth_scale(encoding: str) -> float:
    """Metres per unit for a depth image ``encoding``.

    Raises on unknown encodings rather than assuming metres.
    """
    scale = DEPTH_SCALES.get(str(encoding))
    if scale is None:
        raise ValueError(
            f"unsupported depth encoding {encoding!r}; "
            f"supported: {sorted(DEPTH_SCALES)}"
        )
    return scale


def deproject(
    depth: np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    stride: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """Deproject a depth image to an (N, 3) cloud in the camera optical frame.

    ``depth`` is in metres, shaped (height, width). Points outside the depth band
    (including zero "no reading" pixels) are dropped. ``stride`` keeps every
    n-th row and column; intrinsics are not rescaled, since original pixel
    coordinates are used.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D (h, w), got shape {depth.shape}")
    height, width = depth.shape
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"depth image is {width}x{height} but its intrinsics describe "
            f"{intrinsics.width}x{intrinsics.height} -- the pair does not "
            "describe one image, so every deprojected point would be wrong"
        )
    stride = max(1, int(stride))

    us = np.arange(0, width, stride, dtype=float)
    vs = np.arange(0, height, stride, dtype=float)
    z = depth[::stride, ::stride]
    # Strict bounds also drop NaNs.
    keep = (z > min_depth_m) & (z < max_depth_m)
    if not keep.any():
        return np.empty((0, 3), dtype=float)

    u_grid, v_grid = np.meshgrid(us, vs)
    z = z[keep]
    u_grid = u_grid[keep]
    v_grid = v_grid[keep]
    return np.column_stack(
        (
            (u_grid - intrinsics.cx) * z / intrinsics.fx,
            (v_grid - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """``points`` (N, 3) through a 4x4 rigid ``transform``, as (N, 3)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]
