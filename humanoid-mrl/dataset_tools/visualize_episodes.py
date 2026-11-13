#!/usr/bin/env python3
"""Plot a random sample of a dataset's episodes with visualize_episode's panels.

Spot-checks a whole dataset (task mix, truncated approaches, missing settle
plateaus). A dataset smaller than the sample size is plotted in full.

Usage:
    python3 visualize_episodes.py --dataset data/small_targets/g1_dataset_autocollect_v2
    python3 visualize_episodes.py --dataset <dataset> --count 4 --seed 7 --out /tmp/plots
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

# Headless backend, set before visualize_episode imports pyplot.
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from visualize_episode import (  # noqa: E402
    _load_meta,
    _parse_vector_col,
    plot_episode,
    print_summary,
)

DEFAULT_COUNT = 10


def episode_files(dataset: Path) -> list[Path]:
    """Every episode parquet in the dataset, in chunk then episode order.

    Globs across all chunks.
    """
    files = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        sys.exit(f"No episode parquets under {dataset / 'data'}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="plot exactly these episode indices instead of sampling — how a "
        "caller drives this and the video checks over one shared sample",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"how many episodes to plot (default {DEFAULT_COUNT}); a smaller "
        "dataset is plotted in full. Ignored with --episodes",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the sample; a fresh one is drawn and printed if omitted, "
        "so any run can be reproduced from its own output",
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/episode_plots"))
    args = ap.parse_args()

    files = episode_files(args.dataset)
    if args.episodes is not None:
        by_index = {int(f.stem.split("_")[1]): f for f in files}
        missing = [e for e in args.episodes if e not in by_index]
        if missing:
            sys.exit(f"no such episode(s) in {args.dataset}: {missing}")
        chosen, seed = sorted(by_index[e] for e in args.episodes), None
    else:
        seed = random.randrange(2**32) if args.seed is None else args.seed
        # Random sample, plotted in episode order.
        chosen = sorted(random.Random(seed).sample(files, min(args.count, len(files))))

    args.out.mkdir(parents=True, exist_ok=True)
    origin = "given" if seed is None else f"seed {seed}"
    print(
        f"{len(files)} episodes in {args.dataset}; "
        f"plotting {len(chosen)} ({origin}) into {args.out}"
    )

    for path in chosen:
        df = pd.read_parquet(path)
        if df.empty:
            print(f"skipping {path.name}: empty")
            continue
        state = _parse_vector_col(df, "observation.state")
        action = _parse_vector_col(df, "action")
        ep_idx = (
            int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else 0
        )
        meta = _load_meta(args.dataset, ep_idx)
        print_summary(df, state, action, meta, path)
        plot_episode(
            df,
            state,
            action,
            meta,
            path,
            args.out / f"{path.stem}.png",
            show=False,
        )


if __name__ == "__main__":
    main()
