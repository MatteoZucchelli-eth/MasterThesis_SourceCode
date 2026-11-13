# MIT License
"""Produce 2D target labels for every frame of a recorded dataset.

Writes one sidecar parquet per episode (same chunking as the source data,
under ``<out>/chunk-XXX/``) with the projected target geometry and
visibility flags, plus a ``labels_info.json`` capturing the camera model,
thresholds and aggregate stats. The source dataset is not modified; a
training loader joins the sidecar to the original videos by episode index
+ ``frame_index``.

Label row t is the projection of state row t (see labels.py): the recorder
pairs each frame with the state interpolated at that frame's own image
stamp, so no resampling stands between them.

Labels are projected into the wrist camera. Every source video frame gets one
sidecar row and is eligible for training.

Per-frame columns:
    frame_index          video frame number (= source parquet row)
    u, v                 target-centre pixel (NaN if no valid label)
    depth_z              optical-axis depth of the centre [m]
    wall_normal_cam_0..2 target x-axis (wall normal) in the optical frame
    tip_target_dist      ||tip->target|| [m]
    label_ok             a label exists: the target resolved and projected
                         in front of the camera
    in_front             centre depth > 0
    in_fov               centre inside the image with --margin px
    target_fraction      fraction of the projected target-shape polygon actually
                         showing the target's colour in the video frame (NaN
                         if skipped); low = occluded (e.g. by the drill bit)
                         or mislabel
    visible              label_ok & in_fov & target_fraction >= --target-threshold

Every sidecar row is a training sample. ``visible`` controls which samples
supervise localization and depth; all rows supervise the visibility logit, so
out-of-view and occluded frames are explicit negatives.

Usage:
    python -m dataset_tools.heatmap_labels.process_dataset \
        --dataset data/small_targets/g1_dataset_small_synthetic_1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import camera
from .labels import colour_mask, compute_episode_labels, project_episode
from .overlay_check import episode_paths, episode_target_colour, episode_target_shape
from .projector import TargetProjector

_DEFAULT_OUT_DIR = "heatmap_labels"
"""Where the sidecars land by default, under the dataset root."""


def target_fraction(
    frame_bgr: np.ndarray, outline: np.ndarray, colour: str = "red"
) -> float:
    """Fraction of the projected target-shape polygon showing ``colour``.

    Targets render saturated and the drill/wall do not, so a channel-dominance
    test suffices. NaN when the polygon has no in-image area.
    """
    if outline is None or np.isnan(outline).any():
        return float("nan")
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outline.round().astype(np.int32)], 1)
    area = int(mask.sum())
    if area == 0:
        return float("nan")
    return float((colour_mask(frame_bgr, colour) & (mask == 1)).sum() / area)


def process_episode(
    projector: TargetProjector,
    parquet: Path,
    video: Path,
    margin_px: float,
    target_threshold: float,
    color_check: bool,
    dataset_root: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    """(sidecar rows, source frame count) for one episode.

    The frame count is returned separately for the aggregate source/label check.
    """
    df = pd.read_parquet(parquet)
    shape = (
        episode_target_shape(dataset_root, df) if dataset_root is not None else "square"
    )
    raw = project_episode(df, projector, target_shape=shape)
    outlines = raw.pop("_outline_uv")
    colour = (
        episode_target_colour(dataset_root, df) if dataset_root is not None else "red"
    )

    labels = compute_episode_labels(
        df,
        projector,
        fov_margin_px=margin_px,
        raw=raw,
    )
    n_frames = len(labels)

    # One sequential video pass measures the visible colour inside each outline.
    tf = np.full(len(labels), np.nan)
    if color_check:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video}")
        pos = -1
        for i, target_idx in enumerate(labels["frame_index"].to_numpy()):
            while pos < target_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                pos += 1
            if pos != target_idx:
                break
            row = labels.iloc[i]
            if row["label_ok"]:
                tf[i] = target_fraction(frame, outlines.iloc[int(target_idx)], colour)
        cap.release()
    labels["target_fraction"] = tf

    labels["visible"] = (
        labels["label_ok"]
        & labels["in_fov"]
        & (np.isnan(tf) | (tf >= target_threshold))
    )
    return labels, n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output root (default <dataset>/{_DEFAULT_OUT_DIR})",
    )
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--target-threshold", type=float, default=0.30)
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument(
        "--no-color-check",
        action="store_true",
        help="skip video decoding; target_fraction becomes NaN and visibility "
        "falls back to the geometric flags alone",
    )
    args = ap.parse_args()
    print(camera.describe(), flush=True)

    out_root = args.out or (args.dataset / _DEFAULT_OUT_DIR)

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    episodes = (
        args.episodes
        if args.episodes is not None
        else list(range(info["total_episodes"]))
    )

    projector = TargetProjector()
    agg = {
        "episodes": 0,
        "source_frames": 0,
        "frames": 0,
        "visible": 0,
        "fk_err_max_mm": 0.0,
    }
    t0 = time.time()
    # Live status line on a TTY; every 50 episodes otherwise.
    live = sys.stdout.isatty()
    width = 0
    for n, ep in enumerate(episodes):
        parquet, video = episode_paths(args.dataset, ep, chunks_size)
        labels, n_frames = process_episode(
            projector,
            parquet,
            video,
            margin_px=args.margin,
            target_threshold=args.target_threshold,
            color_check=not args.no_color_check,
            dataset_root=args.dataset,
        )
        out_dir = out_root / f"chunk-{ep // chunks_size:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(out_dir / f"episode_{ep:06d}.parquet")

        fk_max = np.nanmax(labels["fk_tip_err_m"].to_numpy()) * 1000
        agg["episodes"] += 1
        agg["source_frames"] += n_frames
        agg["frames"] += len(labels)
        agg["visible"] += int(labels["visible"].sum())
        agg["fk_err_max_mm"] = max(agg["fk_err_max_mm"], float(fk_max))
        rate = (n + 1) / max(time.time() - t0, 1e-9)
        status = (
            f"[{n + 1}/{len(episodes)}] ep {ep} | {rate:.2f} eps/s | "
            f"eta {(len(episodes) - n - 1) / rate / 60:5.1f} min | "
            f"rows {agg['frames']}/{agg['source_frames']} | "
            f"visible {agg['visible'] / max(agg['frames'], 1):.0%}"
        )
        if live:
            # Pad to clear the previous, longer status.
            width = max(width, len(status))
            print(f"\r{status:<{width}}", end="", flush=True)
        elif (n + 1) % 50 == 0 or n + 1 == len(episodes):
            print(status, flush=True)
    if live:
        print()

    meta = {
        "source_dataset": str(args.dataset),
        "video_key": camera.VIDEO_KEY,
        "camera": {
            "name": camera.CAMERA.name,
            "frame": camera.CAMERA_FRAME,
            "width": camera.IMAGE_WIDTH,
            "height": camera.IMAGE_HEIGHT,
            "fx": camera.FX,
            "fy": camera.FY,
            "cx": camera.CX,
            "cy": camera.CY,
            "distortion": [0.0] * 5,
        },
        "params": {
            "target_threshold": args.target_threshold,
            "fov_margin_px": args.margin,
            "color_check": not args.no_color_check,
        },
        "stats": agg,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "labels_info.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {agg} -> {out_root}")


if __name__ == "__main__":
    main()
