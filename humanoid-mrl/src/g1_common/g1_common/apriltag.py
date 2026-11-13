# MIT License
"""Shared AprilTag generation for printing and simulation."""

import cv2
import numpy as np

DEFAULT_FAMILY = "tag36h11"
QUIET_CELLS = 2
_FAMILIES = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


def _dictionary(family: str):
    if family not in _FAMILIES:
        raise ValueError(f"unknown tag family {family!r}")
    getter = getattr(cv2.aruco, "getPredefinedDictionary", None)
    return (getter or cv2.aruco.Dictionary_get)(_FAMILIES[family])


def cells_per_side(family: str = DEFAULT_FAMILY) -> int:
    dictionary = _dictionary(family)
    size = getattr(dictionary, "markerSize", None)
    return int(size if size is not None else dictionary.getMarkerSize()) + 2


def tag_bitmap(
    tag_id: int, family: str = DEFAULT_FAMILY, cell_px: int = 40
) -> np.ndarray:
    n = cells_per_side(family)
    draw = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
    return np.ascontiguousarray(
        np.rot90(draw(_dictionary(family), int(tag_id), n * cell_px), 2)
    )


def tag_cells(tag_id: int, family: str = DEFAULT_FAMILY) -> np.ndarray:
    n = cells_per_side(family)
    bitmap = tag_bitmap(tag_id, family, 1)
    if bitmap.shape != (n, n):
        step = bitmap.shape[0] // n
        bitmap = bitmap[step // 2 :: step, step // 2 :: step][:n, :n]
    return bitmap < 128


def printable_tag(
    tag_id: int,
    family: str = DEFAULT_FAMILY,
    size_m: float = 0.12,
    dpi: int = 300,
    quiet_cells: int = QUIET_CELLS,
) -> tuple[np.ndarray, float]:
    n = cells_per_side(family)
    cell_px = max(1, round(size_m / n * dpi / 0.0254))
    pitch_m = cell_px * 0.0254 / dpi
    tag = tag_bitmap(tag_id, family, cell_px)
    margin = quiet_cells * cell_px
    image = np.pad(tag, margin, constant_values=255)
    return image, pitch_m
