# MIT License
"""Overlay projected target labels on wrist-view frames to validate geometry.

Validates intrinsics, camera extrinsics, joint mapping, 9D conventions and
frame composition at once. Timing is checked by sync_check.py instead.

Usage:
    python -m dataset_tools.heatmap_labels.overlay_check \
        --dataset data/small_targets/g1_dataset_small_synthetic_1500 \
        --episodes 0 5 100 --stride 60 --out /tmp/overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import camera
from .labels import (
    compute_episode_labels,
    project_episode,
    target_colour_from_task,
    target_shape_from_task,
)
from .projector import TargetProjector


def episode_task(root: Path, df: pd.DataFrame) -> str:
    """This episode's task string, or "" if the dataset does not carry one."""
    tasks_file = root / "meta" / "tasks.jsonl"
    if not tasks_file.exists() or "task_index" not in df:
        return ""
    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in tasks_file.read_text().strip().splitlines()
    }
    return tasks.get(int(df["task_index"].iloc[0]), "")


def episode_target_colour(root: Path, df: pd.DataFrame) -> str:
    """The colour of the target this episode's task names.

    Read per episode, since randomised datasets mix colours.
    """
    return target_colour_from_task(episode_task(root, df))


def episode_target_shape(root: Path, df: pd.DataFrame) -> str:
    """The outline of the target this episode's task names.

    Defaults to "square" when the task names no shape.
    """
    return target_shape_from_task(episode_task(root, df))


def episode_paths(root: Path, ep: int, chunks_size: int = 1000) -> tuple[Path, Path]:
    """(parquet, video) for one episode; the video is the wrist view."""
    chunk = ep // chunks_size
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
    video = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / camera.VIDEO_KEY
        / f"episode_{ep:06d}.mp4"
    )
    return parquet, video


def draw_overlay(
    frame: np.ndarray, row: pd.Series, outline: np.ndarray | None = None
) -> np.ndarray:
    out = frame.copy()
    if row["label_ok"] and not np.isnan(row["u"]):
        if outline is not None and not np.isnan(outline).any():
            cv2.polylines(
                out,
                [outline.round().astype(np.int32)],
                isClosed=True,
                color=(0, 255, 255),
                thickness=1,
            )
        # Cross size scaled to frame width.
        cv2.drawMarker(
            out,
            (round(row["u"]), round(row["v"])),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            markerSize=max(8, round(camera.IMAGE_WIDTH / 30)),
            thickness=1,
        )
    txt = (
        f"f{int(row['frame_index'])}  "
        f"tip-tgt {row['tip_target_dist'] * 1000:.0f}mm  "
        f"cam z {row['depth_z'] * 1000:.0f}mm  "
        f"fov {bool(row['in_fov'])}  ok {bool(row['label_ok'])}"
    )
    cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--stride", type=int, default=60, help="save every Nth frame")
    ap.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help="explicit frame indices instead of the stride",
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/overlay"))
    args = ap.parse_args()
    print(camera.describe(), flush=True)

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    projector = TargetProjector()

    for ep in args.episodes:
        parquet, video = episode_paths(args.dataset, ep, chunks_size)
        df = pd.read_parquet(parquet)
        colour = episode_target_colour(args.dataset, df)
        shape = episode_target_shape(args.dataset, df)
        raw = project_episode(df, projector, target_shape=shape)
        outlines = raw.pop("_outline_uv")
        labels = compute_episode_labels(df, projector, raw=raw)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video}")

        n_saved = 0
        for i in range(len(labels)):
            ok, frame = cap.read()
            if not ok:
                print(f"ep {ep}: video ended early at frame {i}/{len(labels)}")
                break
            wanted = (
                i in args.frames if args.frames is not None else i % args.stride == 0
            )
            if wanted:
                cv2.imwrite(
                    str(args.out / f"ep{ep:06d}_f{i:04d}.png"),
                    draw_overlay(frame, labels.iloc[i], outlines.iloc[i]),
                )
                n_saved += 1
        cap.release()

        fk = labels["fk_tip_err_m"].to_numpy() * 1000
        fk = fk[~np.isnan(fk)]
        ok_rows = labels["label_ok"]
        print(
            f"ep {ep} [{colour} {shape}]: {len(labels)} frames | "
            f"label_ok {ok_rows.mean():.0%} | "
            f"in-FOV {labels['in_fov'][ok_rows].mean():.0%} | "
            f"FK-vs-state tip err mm: median {np.median(fk):.2f} "
            f"max {fk.max():.2f} | {n_saved} overlays -> {args.out}"
        )


if __name__ == "__main__":
    main()
