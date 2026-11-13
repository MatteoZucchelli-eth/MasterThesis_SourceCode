#!/usr/bin/env python3
"""Delete episode(s) from the G1 LeRobot/GR00T-style dataset and reindex the rest.

LeRobot needs contiguous episode indices, so survivors are renumbered: file
names, the ``episode_index`` and global ``index`` columns, meta/episodes.jsonl
and meta/info.json. Validates and prints the plan before touching disk.

Run with the recorder stopped.

Usage
-----
    python3 delete_episode.py --episode 7                 # delete one
    python3 delete_episode.py --episode 7 12 3            # delete several
    python3 delete_episode.py --episode 7 --dry-run       # show plan only
    python3 delete_episode.py --episode 7 --yes           # skip confirmation
    python3 delete_episode.py --dataset /path --episode 7 # non-default dataset

``tasks.jsonl`` and ``modality.json`` are left untouched (task indices are stable
and shared across episodes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DATASET = "/workspace/g1_dataset"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _episode_chunk(idx: int, chunks_size: int) -> int:
    return idx // chunks_size


def _data_path(root: Path, info: dict, idx: int) -> Path:
    chunk = _episode_chunk(idx, info["chunks_size"])
    return root / info["data_path"].format(episode_chunk=chunk, episode_index=idx)


def _video_paths(root: Path, info: dict, cameras: list[str], idx: int) -> list[Path]:
    chunk = _episode_chunk(idx, info["chunks_size"])
    return [
        root
        / info["video_path"].format(
            episode_chunk=chunk, video_key=cam, episode_index=idx
        )
        for cam in cameras
    ]


def _discover_cameras(root: Path, info: dict) -> list[str]:
    """Camera video keys, read from the videos/chunk-000 directory on disk."""
    chunk_dir = root / f"videos/chunk-{0:03d}"
    if not chunk_dir.is_dir():
        return []
    return sorted(p.name for p in chunk_dir.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset root")
    ap.add_argument(
        "--episode",
        type=int,
        nargs="+",
        required=True,
        help="Episode index/indices to delete",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the plan and change nothing"
    )
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = ap.parse_args(argv)

    root = Path(args.dataset)
    meta = root / "meta"
    info_path = meta / "info.json"
    episodes_path = meta / "episodes.jsonl"

    for p in (info_path, episodes_path):
        if not p.exists():
            print(f"error: {p} not found — is {root} a dataset?", file=sys.stderr)
            return 1

    info = json.load(open(info_path))
    episodes = _load_jsonl(episodes_path)
    episodes.sort(key=lambda e: e["episode_index"])
    cameras = _discover_cameras(root, info)

    existing = {e["episode_index"] for e in episodes}
    to_delete = set(args.episode)
    unknown = sorted(to_delete - existing)
    if unknown:
        print(
            f"error: episode(s) {unknown} not in dataset " f"(have 0..{max(existing)})",
            file=sys.stderr,
        )
        return 1
    if len(to_delete) >= len(episodes):
        print("error: refusing to delete every episode", file=sys.stderr)
        return 1

    # ── Build the reindexing plan ──────────────────────────────────────────
    survivors = [e for e in episodes if e["episode_index"] not in to_delete]
    plan: list[dict] = []  # per survivor: old/new episode index + old/new global start
    old_start = {}
    cursor = 0
    for e in episodes:
        old_start[e["episode_index"]] = cursor
        cursor += e["length"]

    new_cursor = 0
    for new_idx, e in enumerate(survivors):
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

    total_frames_new = new_cursor
    frames_removed = info["total_frames"] - total_frames_new
    changed = [
        p
        for p in plan
        if p["old_idx"] != p["new_idx"] or p["old_gstart"] != p["new_gstart"]
    ]

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"Dataset: {root}")
    print(f"Cameras: {', '.join(cameras) if cameras else '(none found)'}")
    print(
        f"Delete {sorted(to_delete)}: "
        f"{len(episodes)} -> {len(survivors)} episodes, "
        f"{info['total_frames']} -> {total_frames_new} frames "
        f"(-{frames_removed})"
    )
    renamed = [p for p in plan if p["old_idx"] != p["new_idx"]]
    if renamed:
        preview = ", ".join(f"{p['old_idx']}->{p['new_idx']}" for p in renamed[:12])
        more = "" if len(renamed) <= 12 else f", … (+{len(renamed) - 12})"
        print(f"Renumber {len(renamed)} episode(s): {preview}{more}")
    else:
        print("No renumbering needed (deleting the tail).")

    if args.dry_run:
        print("\n--dry-run: no files changed.")
        return 0

    if not args.yes:
        reply = (
            input("\nProceed? This permanently deletes files [y/N]: ").strip().lower()
        )
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    # ── Execute ────────────────────────────────────────────────────────────
    # 1. Remove the deleted episodes' files, freeing their slots.
    for idx in sorted(to_delete):
        for f in [_data_path(root, info, idx), *_video_paths(root, info, cameras, idx)]:
            if f.exists():
                f.unlink()

    # 2. Slide survivors down; ascending order guarantees each target slot is free.
    for p in changed:
        old_idx, new_idx = p["old_idx"], p["new_idx"]

        # Parquet: rewrite episode_index + global index.
        src_pq = _data_path(root, info, old_idx)
        dst_pq = _data_path(root, info, new_idx)
        df = pd.read_parquet(src_pq)
        df["episode_index"] = new_idx
        df["index"] = range(p["new_gstart"], p["new_gstart"] + len(df))
        df.to_parquet(dst_pq, index=False)
        if dst_pq != src_pq:
            src_pq.unlink()

        # Videos: pure rename, one per camera.
        if new_idx != old_idx:
            src_vids = _video_paths(root, info, cameras, old_idx)
            dst_vids = _video_paths(root, info, cameras, new_idx)
            for src, dst in zip(src_vids, dst_vids):
                if src.exists():
                    src.rename(dst)

    # 3. Rewrite episodes.jsonl.
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

    # 4. Update info.json counters.
    info["total_episodes"] = len(survivors)
    info["total_frames"] = total_frames_new
    info["total_videos"] = len(survivors) * len(cameras)
    info["splits"] = {"train": f"0:{len(survivors)}"}
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)

    print(
        f"\nDone. Dataset now has {len(survivors)} episodes, {total_frames_new} frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
