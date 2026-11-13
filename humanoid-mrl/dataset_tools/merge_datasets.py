#!/usr/bin/env python3
# MIT License
"""Merge per-worker LeRobot datasets from parallel AUTO_COLLECT runs into one.

Each headless collection worker writes its own dataset root
(``g1_dataset_worker_<i>``) because the recorder is a single-writer design.
This script re-indexes those roots into a single dataset the fine-tuning
pipeline can load: episode and frame indices become globally consecutive, task
indices are remapped onto a unified ``tasks.jsonl``, videos are copied under
their new episode names, and ``info.json`` totals are rebuilt. Episodes are
placed into ``chunk-<n>`` directories according to ``chunks_size``, so merges
past 1000 episodes stay loadable.

Run it through ``dataset_tools/scripts/merge_datasets.sh``, which supplies the venv and the
repo root this module has to be imported from::

    dataset_tools/scripts/merge_datasets.sh --out g1_dataset_merged \
        g1_dataset_worker_1 g1_dataset_worker_2 ...

Source datasets are read-only; the output directory must not already exist
(no silent appends — re-run the merge from scratch instead).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Parquet columns holding a task index (as written by RecorderNode).
_TASK_INDEX_COLUMNS = (
    "task_index",
    "annotation.human.action.task_description",
    "annotation.human.validity",
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append_jsonl(path: Path, entry: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _video_columns(info: dict) -> list[str]:
    return [
        key
        for key, feat in info.get("features", {}).items()
        if feat.get("dtype") == "video"
    ]


def _chunk_dir(episode_index: int, chunks_size: int) -> str:
    return f"chunk-{episode_index // chunks_size:03d}"


def _find_source_file(
    top: Path, episode_index: int, chunks_size: int, tail: str
) -> Path | None:
    """Locate ``<top>/chunk-<n>/<tail>``, falling back to scanning all chunk dirs.

    The recorder may write every episode into ``chunk-000`` regardless of
    ``chunks_size``.
    """
    expected = top / _chunk_dir(episode_index, chunks_size) / tail
    if expected.exists():
        return expected
    matches = sorted(top.glob(f"chunk-*/{tail}"))
    return matches[0] if matches else None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when source and destination share a filesystem, else copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_source(root: Path) -> dict:
    """Read one worker dataset's metadata, validating the pieces the merge needs."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"error: {root} has no meta/info.json — not a dataset root?")
    with open(info_path) as fh:
        info = json.load(fh)

    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    if not episodes:
        print(f"warning: {root} contains no episodes; skipping.")
    return {"root": root, "info": info, "episodes": episodes, "tasks": tasks}


def _check_compatible(reference: dict, other: dict) -> None:
    """Datasets must agree on everything that describes a frame's layout."""
    ref_info, info = reference["info"], other["info"]
    for key in ("fps", "state_dim", "action_dim", "robot_type", "codebase_version"):
        if ref_info.get(key) != info.get(key):
            sys.exit(
                f"error: {other['root']} is incompatible with {reference['root']}: "
                f"info.json '{key}' differs ({info.get(key)} vs {ref_info.get(key)})."
            )
    if set(_video_columns(ref_info)) != set(_video_columns(info)):
        sys.exit(
            f"error: {other['root']} has different camera streams than "
            f"{reference['root']}."
        )


def merge(source_roots: list[Path], out_root: Path) -> None:
    sources = [_load_source(root) for root in source_roots]
    for other in sources[1:]:
        _check_compatible(sources[0], other)

    info = json.loads(json.dumps(sources[0]["info"]))  # deep copy
    chunks_size = int(info.get("chunks_size", 1000))
    video_cols = _video_columns(info)

    (out_root / "meta").mkdir(parents=True)
    # modality.json is recorder-static; take it from the first source.
    src_modality = sources[0]["root"] / "meta" / "modality.json"
    if src_modality.exists():
        shutil.copy2(src_modality, out_root / "meta" / "modality.json")

    # ── Unified task table ────────────────────────────────────────────────────
    task_to_index: dict[str, int] = {}
    for source in sources:
        for entry in source["tasks"]:
            task = entry["task"]
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
    for task, idx in task_to_index.items():
        _append_jsonl(
            out_root / "meta" / "tasks.jsonl", {"task_index": idx, "task": task}
        )

    # ── Episodes ──────────────────────────────────────────────────────────────
    new_episode = 0
    global_frame = 0
    for source in sources:
        root = source["root"]
        old_to_new_task = {
            int(entry["task_index"]): task_to_index[entry["task"]]
            for entry in source["tasks"]
        }
        for entry in source["episodes"]:
            old_episode = int(entry["episode_index"])
            new_chunk = _chunk_dir(new_episode, chunks_size)

            parquet_in = _find_source_file(
                root / "data",
                old_episode,
                chunks_size,
                f"episode_{old_episode:06d}.parquet",
            )
            if parquet_in is None:
                sys.exit(
                    f"error: episode {old_episode} of {root} is listed in "
                    f"episodes.jsonl but has no parquet under data/chunk-*/."
                )

            df = pd.read_parquet(parquet_in)
            df["episode_index"] = new_episode
            df["index"] = range(global_frame, global_frame + len(df))
            for col in _TASK_INDEX_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].map(old_to_new_task)
                    if df[col].isna().any():
                        sys.exit(
                            f"error: {parquet_in} column '{col}' holds a task index "
                            f"missing from {root}/meta/tasks.jsonl."
                        )

            parquet_out_dir = out_root / "data" / new_chunk
            parquet_out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(
                parquet_out_dir / f"episode_{new_episode:06d}.parquet", index=False
            )

            for col in video_cols:
                video_in = _find_source_file(
                    root / "videos",
                    old_episode,
                    chunks_size,
                    f"{col}/episode_{old_episode:06d}.mp4",
                )
                if video_in is None:
                    sys.exit(
                        f"error: episode {old_episode} of {root} has no "
                        f"{col} video under videos/chunk-*/."
                    )
                video_out_dir = out_root / "videos" / new_chunk / col
                video_out_dir.mkdir(parents=True, exist_ok=True)
                _link_or_copy(
                    video_in, video_out_dir / f"episode_{new_episode:06d}.mp4"
                )

            _append_jsonl(
                out_root / "meta" / "episodes.jsonl",
                {
                    "episode_index": new_episode,
                    "tasks": [old_to_new_task[t] for t in entry["tasks"]],
                    "length": int(entry["length"]),
                },
            )
            global_frame += len(df)
            new_episode += 1
        print(f"{root}: merged {len(source['episodes'])} episodes.")

    # ── info.json ─────────────────────────────────────────────────────────────
    info["total_episodes"] = new_episode
    info["total_frames"] = global_frame
    info["total_tasks"] = len(task_to_index)
    info["total_videos"] = new_episode * len(video_cols)
    info["total_chunks"] = (new_episode - 1) // chunks_size + 1 if new_episode else 0
    info["splits"] = {"train": f"0:{new_episode}"}
    with open(out_root / "meta" / "info.json", "w") as fh:
        json.dump(info, fh, indent=2)

    print(
        f"\nMerged {new_episode} episodes / {global_frame} frames from "
        f"{len(sources)} dataset(s) into {out_root}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sources", nargs="+", type=Path, help="Worker dataset roots, in merge order."
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output root (must not exist yet)."
    )
    args = parser.parse_args()

    if args.out.exists():
        sys.exit(f"error: {args.out} already exists — merge to a fresh directory.")
    for source in args.sources:
        if args.out.resolve() == source.resolve():
            sys.exit("error: --out must differ from every source root.")

    merge(args.sources, args.out)


if __name__ == "__main__":
    main()
