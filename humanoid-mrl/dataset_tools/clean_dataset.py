#!/usr/bin/env python3
"""Clean slip and all-black-camera episodes from a dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

DIST_COL = "next.tip_target_dist"
SEG_MM = 0.5  # excursion boundary; the frame-to-frame noise floor is ~0.11 mm


# ── dataset layout ────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _data_path(root: Path, info: dict, idx: int) -> Path:
    chunk = idx // info["chunks_size"]
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=idx)


def _aux_path(root: Path, info: dict, tree: str, idx: int) -> Path:
    """Per-episode parquet in an auxiliary tree laid out like ``data/``."""
    chunk = idx // info["chunks_size"]
    return root / tree / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"


def _video_paths(root: Path, info: dict, cams: list[str], idx: int) -> list[Path]:
    chunk = idx // info["chunks_size"]
    return [
        root / info["video_path"].format(
            episode_chunk=chunk, video_key=cam, episode_index=idx
        )
        for cam in cams
    ]


def _discover_cameras(root: Path) -> list[str]:
    chunk_dir = root / "videos" / "chunk-000"
    if not chunk_dir.is_dir():
        return []
    return sorted(p.name for p in chunk_dir.iterdir() if p.is_dir())


def _discover_aux_trees(root: Path) -> list[str]:
    """Sibling trees of per-episode parquet keyed on the episode index."""
    return [
        d.name
        for d in sorted(root.iterdir())
        if d.is_dir()
        and d.name not in ("data", "videos", "meta")
        and (d / "chunk-000").is_dir()
    ]


# ── detection ─────────────────────────────────────────────────────────────────


def find_slip(path: Path, near_mm: float, min_drawup_mm: float) -> dict | None:
    """The worst qualifying slip in one episode, or None if it is clean."""
    col = pq.read_table(path, columns=[DIST_COL])[DIST_COL]
    d = np.array(col, dtype=float) * 1000.0  # copy (arrow buffers are read-only), m -> mm
    n = len(d)
    if n == 0:
        return None
    if np.isnan(d).any():  # interpolate over unlabelled frames
        ok = np.where(~np.isnan(d))[0]
        if len(ok) == 0:
            return None
        d = np.interp(np.arange(n), ok, d[ok])

    runmin = np.minimum.accumulate(d)
    rise = d - runmin

    worst = None
    i = 0
    while i < n:
        if rise[i] <= SEG_MM:
            i += 1
            continue
        j = i
        while j < n and rise[j] > SEG_MM:
            j += 1
        peak = i + int(np.argmax(rise[i:j]))
        if runmin[i] <= near_mm and rise[peak] >= min_drawup_mm:
            hit = {
                "drawup_mm": float(rise[peak]),
                "dist_at_start_mm": float(runmin[i]),
                "peak_dist_mm": float(d[peak]),
                "start_frame": i,
                "peak_frame": peak,
                "peak_frac": peak / n,
                "recovered": bool(j < n),
                "final_dist_mm": float(d[-1]),
                "n_frames": n,
            }
            if worst is None or hit["drawup_mm"] > worst["drawup_mm"]:
                worst = hit
        i = j
    return worst


def is_black_video(path: Path, threshold: float) -> bool:
    """Return true when the start, middle, and end frames are black."""
    video = cv2.VideoCapture(str(path))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        video.release()
        return False

    for frame_index in (0, count // 2, count - 1):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = video.read()
        if not ok or float(frame.mean()) > threshold:
            video.release()
            return False
    video.release()
    return True


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", help="Dataset root (the directory holding meta/)")
    ap.add_argument(
        "--min-drawup-mm",
        type=float,
        default=3.0,
        help="Flag when the distance rises this far above its best-so-far "
        "(default: %(default)s; 3-15 all select the same episodes)",
    )
    ap.add_argument(
        "--near-mm",
        type=float,
        default=15.0,
        help="Only count excursions starting within this distance of the "
        "target, which excludes the transit phase (default: %(default)s)",
    )
    ap.add_argument(
        "--min-peak-frac",
        type=float,
        default=0.0,
        help="Also require the excursion to peak this far into the episode, "
        "as a fraction (default: %(default)s = anywhere)",
    )
    ap.add_argument(
        "--black-threshold",
        type=float,
        default=2.0,
        help="Maximum mean pixel value for a black frame (default: %(default)s)",
    )
    ap.add_argument("--report", help="Write per-episode removal details to this CSV")
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the plan and change nothing"
    )
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = ap.parse_args(argv)

    root = Path(args.dataset)
    info_path, episodes_path = root / "meta/info.json", root / "meta/episodes.jsonl"
    for p in (info_path, episodes_path):
        if not p.exists():
            print(f"error: {p} not found — is {root} a dataset?", file=sys.stderr)
            return 1

    info = json.load(open(info_path))
    episodes = sorted(_load_jsonl(episodes_path), key=lambda e: e["episode_index"])
    cams = _discover_cameras(root)
    aux = _discover_aux_trees(root)
    print(f"Dataset: {root}")
    print(f"Episodes: {len(episodes)}   cameras: {cams or '(none)'}   aux: {aux or '(none)'}")

    # ── scan ──────────────────────────────────────────────────────────────
    rejected: dict[int, dict] = {}
    for k, e in enumerate(episodes):
        idx = e["episode_index"]
        pqf = _data_path(root, info, idx)
        if not pqf.exists():
            print(f"warning: {pqf} missing, skipping", file=sys.stderr)
            continue
        hit = find_slip(pqf, args.near_mm, args.min_drawup_mm)
        slip = hit if hit and hit["peak_frac"] >= args.min_peak_frac else None
        black_cameras = [
            cam
            for cam, path in zip(cams, _video_paths(root, info, cams, idx))
            if path.exists() and is_black_video(path, args.black_threshold)
        ]
        if slip or black_cameras:
            rejected[idx] = {
                "slip": slip,
                "black_cameras": black_cameras,
                "n_frames": e["length"],
            }
        if (k + 1) % 500 == 0:
            print(f"  scanned {k + 1}/{len(episodes)} …", flush=True)

    if not rejected:
        print("\nNo slip or black-camera episodes found. Nothing to do.")
        return 0

    slips = {idx: finding["slip"] for idx, finding in rejected.items() if finding["slip"]}
    black = {
        idx: finding["black_cameras"]
        for idx, finding in rejected.items()
        if finding["black_cameras"]
    }
    print(
        f"\nFound {len(rejected)} episodes to remove: "
        f"{len(slips)} slips, {len(black)} with black cameras"
    )
    if slips:
        draw = np.array([h["drawup_mm"] for h in slips.values()])
        print(
            f"  slip drawup mm: p50 {np.median(draw):.1f}  "
            f"p90 {np.percentile(draw, 90):.1f}  max {draw.max():.1f}"
        )
    if black:
        counts = ", ".join(
            f"{cam}: {sum(cam in names for names in black.values())}" for cam in cams
        )
        print(f"  black cameras: {counts}")

    if args.report:
        with open(args.report, "w", newline="") as fh:
            cols = [
                "episode_index", "reason", "black_cameras", "n_frames",
                "drawup_mm", "dist_at_start_mm", "peak_dist_mm", "start_frame",
                "peak_frame", "peak_frac", "recovered", "final_dist_mm",
            ]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for idx, finding in sorted(rejected.items()):
                slip = finding["slip"] or {}
                reasons = ["slip"] if finding["slip"] else []
                if finding["black_cameras"]:
                    reasons.append("black_camera")
                w.writerow(
                    {
                        "episode_index": idx,
                        "reason": "+".join(reasons),
                        "black_cameras": ";".join(finding["black_cameras"]),
                        "n_frames": finding["n_frames"],
                        **{c: slip.get(c, "") for c in cols[4:]},
                    }
                )
        print(f"  report written to {args.report}")

    if len(rejected) >= len(episodes):
        print("error: refusing to delete every episode", file=sys.stderr)
        return 1

    # ── plan: survivors slide down to close the gaps ──────────────────────
    old_start, cursor = {}, 0
    for e in episodes:
        old_start[e["episode_index"]] = cursor
        cursor += e["length"]

    plan, new_cursor = [], 0
    for new_idx, e in enumerate(
        e for e in episodes if e["episode_index"] not in rejected
    ):
        old_idx = e["episode_index"]
        plan.append(
            {
                "old_idx": old_idx,
                "new_idx": new_idx,
                "old_gstart": old_start[old_idx],
                "new_gstart": new_cursor,
                "length": e["length"],
                "tasks": e.get("tasks", []),
            }
        )
        new_cursor += e["length"]

    moved = [p for p in plan if p["old_idx"] != p["new_idx"]]
    rewrite = [p for p in plan if p["old_idx"] != p["new_idx"] or p["old_gstart"] != p["new_gstart"]]
    print(
        f"\nPlan: {len(episodes)} -> {len(plan)} episodes, "
        f"{info['total_frames']} -> {new_cursor} frames "
        f"(-{info['total_frames'] - new_cursor}); renumber {len(moved)}"
    )

    if args.dry_run:
        print("--dry-run: no files changed.")
        return 0
    if not args.yes:
        if input("\nProceed? This permanently deletes files [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    # ── execute ───────────────────────────────────────────────────────────
    # 1. Delete rejected episodes.
    for idx in sorted(rejected):
        victims = [
            _data_path(root, info, idx),
            *_video_paths(root, info, cams, idx),
            *(_aux_path(root, info, t, idx) for t in aux),
        ]
        for f in victims:
            if f.exists():
                f.unlink()

    # 2. Slide survivors down. Ascending old-index order keeps every target
    #    slot free, because new_idx <= old_idx and lower slots vacate first.
    for p in rewrite:
        old_idx, new_idx = p["old_idx"], p["new_idx"]

        # data/: rewrite episode_index + the global frame counter, then move.
        src = _data_path(root, info, old_idx)
        dst = _data_path(root, info, new_idx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(src)
        n = table.num_rows
        for col, vals in (
            ("episode_index", np.full(n, new_idx, dtype=np.int64)),
            ("index", np.arange(p["new_gstart"], p["new_gstart"] + n, dtype=np.int64)),
        ):
            if col in table.column_names:
                table = table.set_column(
                    table.column_names.index(col), col, [vals]
                )
        pq.write_table(table, dst)
        if dst != src:
            src.unlink()

        if new_idx == old_idx:
            continue
        # videos/ and the aux trees are keyed only on the file name: rename.
        pairs = list(
            zip(
                _video_paths(root, info, cams, old_idx),
                _video_paths(root, info, cams, new_idx),
            )
        ) + [
            (_aux_path(root, info, t, old_idx), _aux_path(root, info, t, new_idx))
            for t in aux
        ]
        for s, d in pairs:
            if s.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                s.rename(d)

    # 3. Drop chunk directories the shrunken dataset does not fill. Walk
    #    bottom-up, because videos nest a camera level under each chunk.
    for tree in ["data", "videos", *aux]:
        base = root / tree
        if not base.is_dir():
            continue
        for d in sorted((p for p in base.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()

    # 4. Rewrite the metadata.
    with open(episodes_path, "w") as fh:
        for p in plan:
            fh.write(
                json.dumps(
                    {
                        "episode_index": p["new_idx"],
                        "tasks": p["tasks"],
                        "length": p["length"],
                    }
                )
                + "\n"
            )

    info["total_episodes"] = len(plan)
    info["total_frames"] = new_cursor
    info["total_videos"] = len(plan) * len(cams)
    info["total_chunks"] = (len(plan) - 1) // info["chunks_size"] + 1
    info["splits"] = {"train": f"0:{len(plan)}"}
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"\nDone. Dataset now has {len(plan)} episodes, {new_cursor} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
