# MIT License
"""Episode-level label series: the projection of the recorded target per frame.

Row t is the projection of state row t, with no lag correction: the recorder
interpolates state at the wrist image stamp (``recorder_node._recording_loop``).
Verify new datasets with sync_check.py; fix any offset in the recorder.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from . import camera
from .projector import TargetProjector

_GEOMETRY_COLS = [
    "u",
    "v",
    "depth_z",
    "tip_target_dist",
] + [f"wall_normal_cam_{i}" for i in range(3)]
"""Per-frame geometry channels, all NaN on rows without a usable label."""


def project_episode(
    df: pd.DataFrame, projector: TargetProjector, target_shape: str = "square"
) -> pd.DataFrame:
    """Raw per-state-row projection, one output row per parquet row.

    Rows with an unresolved target label (NaN ``next.tip_target_9d``) or a
    behind-the-camera target keep their slot with NaN geometry so the frame
    index stays aligned with the video.
    """
    n = len(df)
    out = {c: np.full(n, np.nan) for c in _GEOMETRY_COLS}
    out["in_front"] = np.zeros(n, dtype=bool)
    out["fk_tip_err_m"] = np.full(n, np.nan)
    outlines: list[np.ndarray | None] = [None] * n

    for i in range(n):
        tt9d = np.asarray(df["next.tip_target_9d"].iloc[i], dtype=float)
        if np.isnan(tt9d).any():
            continue
        state = np.asarray(df["observation.state"].iloc[i], dtype=float)
        p = projector.project_frame(state, tt9d, target_shape)
        out["fk_tip_err_m"][i] = p.fk_tip_err_m
        out["tip_target_dist"][i] = p.tip_target_dist
        out["in_front"][i] = p.in_front
        if not p.in_front:
            continue
        out["u"][i], out["v"][i] = p.uv
        out["depth_z"][i] = p.depth_z
        outlines[i] = p.uv_outline
        for j in range(3):
            out[f"wall_normal_cam_{j}"][i] = p.wall_normal_cam[j]

    res = pd.DataFrame(out)
    res.insert(0, "frame_index", np.arange(n))
    # Private, variable-length geometry used by visibility/overlay checks only.
    # Callers remove it before writing the sidecar parquet.
    res["_outline_uv"] = outlines
    return res


def compute_episode_labels(
    df: pd.DataFrame,
    projector: TargetProjector,
    fov_margin_px: float = 12.0,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label series for one episode, one row per video frame.

    ``label_ok`` marks the rows a trainer may use: the target resolved and
    projected in front of the camera. It is exactly the set of rows with
    non-NaN geometry, so a loader can filter on either.

    ``raw`` short-circuits the projection pass when the caller already ran
    :func:`project_episode`."""
    if raw is None:
        raw = project_episode(df, projector)
    labels = raw.copy()
    labels["label_ok"] = labels["in_front"].to_numpy()

    uv = labels[["u", "v"]].to_numpy()
    labels["in_fov"] = camera.in_image(uv, margin_px=fov_margin_px)
    return labels


TARGET_COLOURS = ("red", "green", "blue")
"""Target colours the scene renders, and the words the task prompts use."""

TARGET_SHAPES = ("square", "circle", "triangle")
"""Target outlines the scene renders, and the words the task prompts use."""


def target_colour_from_task(task: str) -> str:
    """Which square a task instruction refers to ("drill in the green target").

    Defaults to red.
    """
    lowered = task.lower()
    for colour in TARGET_COLOURS:
        if colour in lowered:
            return colour
    return "red"


def target_shape_from_task(task: str) -> str:
    """Which outline a task instruction refers to ("place a hole in the red circle").

    Defaults to square.
    """
    lowered = task.lower()
    for shape in TARGET_SHAPES:
        if shape in lowered:
            return shape
    return "square"


# (min-area-rect fill, circularity) prototype per outline: measured medians on
# g1_dataset_negatives (below ideal due to antialiasing at ~25 px).
_SHAPE_PROTOTYPES = {
    "square": (0.87, 0.64),
    "circle": (0.76, 0.70),
    "triangle": (0.50, 0.45),
}
_SHAPE_TOLERANCE = 0.10
"""Descriptor spread treated as one unit of distance when matching a prototype."""


def classify_blob_shape(contour: np.ndarray) -> str | None:
    """Which of :data:`TARGET_SHAPES` a colour blob's outline looks like.

    Nearest prototype in (fill, circularity), with the polygon vertex count as
    a prior to break the square/circle overlap.

    Returns None for a blob too small or too degenerate to describe.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return None
    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    if rw <= 0 or rh <= 0:
        return None
    fill = area / (rw * rh)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    n_verts = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
    prior = {3: "triangle", 4: "square"}.get(
        n_verts, "circle" if n_verts >= 5 else None
    )

    scores = {
        name: np.hypot(
            (fill - f) / _SHAPE_TOLERANCE, (circularity - c) / _SHAPE_TOLERANCE
        )
        for name, (f, c) in _SHAPE_PROTOTYPES.items()
    }
    if prior is not None:
        scores[prior] -= 0.75
    return min(scores, key=scores.get)


_COLOUR_GATES = {
    "red": (80, 40),
    "green": (50, 25),
    "blue": (80, 40),
}
"""(own-channel floor, dominance margin) per target colour, in 8-bit BGR.

Green renders at roughly half the luminance of red/blue, hence its lower gate.
"""


def colour_mask(frame_bgr: np.ndarray, colour: str) -> np.ndarray:
    """Boolean mask of pixels dominated by ``colour``.

    Targets render saturated, so channel dominance suffices; thresholds are
    per colour (:data:`_COLOUR_GATES`).
    """
    if colour not in _COLOUR_GATES:
        raise ValueError(
            f"unknown target colour '{colour}' (expected {TARGET_COLOURS})"
        )
    floor, margin = _COLOUR_GATES[colour]
    b, g, r = cv2.split(frame_bgr.astype(np.int16))
    own, (o1, o2) = {
        "red": (r, (g, b)),
        "green": (g, (r, b)),
        "blue": (b, (r, g)),
    }[colour]
    return (own > floor) & (own > o1 + margin) & (own > o2 + margin)


def detect_target_centroid(
    frame_bgr: np.ndarray,
    colour: str = "red",
    shape: str | None = None,
    min_area: int = 20,
) -> np.ndarray | None:
    """Centroid of the largest ``colour``-dominant blob, or None.

    Used only to check labels against the render (sync_check.py), not as a
    label (biased by occlusion). Pass the episode's own colour, and ``shape``
    on datasets with same-colour distractors; with ``shape`` it returns None
    rather than a non-matching blob.
    """
    mask = colour_mask(frame_bgr, colour).astype(np.uint8)
    n, labelled, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None

    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    for idx in order:
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            return None
        if shape is None:
            return centroids[idx]
        contours, _ = cv2.findContours(
            (labelled == idx).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        if classify_blob_shape(max(contours, key=cv2.contourArea)) == shape:
            return centroids[idx]
    return None
