#!/usr/bin/env python3
"""Visualize a single recorded episode from the G1 LeRobot-style dataset.

Usage
-----
    python3 visualize_episode.py                            # latest episode
    python3 visualize_episode.py --episode 3               # episode index 3
    python3 visualize_episode.py --path /path/to/episode.parquet
    python3 visualize_episode.py --dataset /my/dataset --episode 0 --no-show

Prints a summary table and saves/shows a multi-panel figure: drill-tip
position (state vs action), drill-tip rot6d, right-arm joints, reward/done,
and tip -> target distance/xyz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from g1_definitions import definitions
from g1_definitions.definitions import ACTION_DIM, STATE_DIM

# key → slice for numpy indexing, from the shared dataset schema.
STATE_SLICES = {k: slice(s, e) for k, (s, e) in definitions.STATE_SLICES.items()}
ACTION_SLICES = {k: slice(s, e) for k, (s, e) in definitions.ACTION_SLICES.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_vector_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return (n_frames, dim) float32 array from a list-of-lists column."""
    return np.array(df[col].tolist(), dtype=np.float32)


def _resolve_parquet(args) -> Path:
    if args.path:
        return Path(args.path)

    data_dir = Path(args.dataset) / "data"
    if not data_dir.exists():
        sys.exit(f"Dataset data directory not found: {data_dir}")

    # Search all chunk-* directories.
    by_index = {
        int(f.stem.split("_")[1]): f
        for f in data_dir.glob("chunk-*/episode_*.parquet")
    }
    if not by_index:
        sys.exit(f"No parquet files found under {data_dir}")

    if args.episode is None:
        return by_index[max(by_index)]  # latest

    if args.episode not in by_index:
        sys.exit(f"Episode {args.episode} not found in {args.dataset}")
    return by_index[args.episode]


def _load_meta(dataset: Path, episode_idx: int) -> dict:
    """Load tasks.jsonl and episodes.jsonl for the given episode."""
    meta = {}
    tasks_path = dataset / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = {}
        with open(tasks_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    tasks[t["task_index"]] = t["task"]
        meta["tasks"] = tasks

    ep_path = dataset / "meta" / "episodes.jsonl"
    if ep_path.exists():
        with open(ep_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if e["episode_index"] == episode_idx:
                        meta["episode"] = e
                        break
    return meta


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(
    df: pd.DataFrame,
    state: np.ndarray,
    action: np.ndarray,
    meta: dict,
    parquet_path: Path,
) -> None:
    n = len(df)
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0] if n > 1 else 0.0
    fps_actual = (n - 1) / duration if duration > 0 else float("nan")

    ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else "?"
    task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else "?"
    task_name = meta.get("tasks", {}).get(task_idx, "?")

    print("=" * 60)
    print(f"  File        : {parquet_path}")
    print(f"  Episode     : {ep_idx}")
    print(f"  Task        : [{task_idx}] {task_name}")
    print(f"  Frames      : {n}")
    print(f"  Duration    : {duration:.2f} s  ({fps_actual:.1f} fps)")
    print(f"  State dim   : {state.shape[1]}")
    print(f"  Action dim  : {action.shape[1]}")
    print("-" * 60)
    print("  NaN check:")
    print(f"    state  NaNs : {np.isnan(state).sum()}")
    print(f"    action NaNs : {np.isnan(action).sum()}")
    print("-" * 60)

    if "next.tip_target_dist" in df.columns:
        dist = df["next.tip_target_dist"].to_numpy(dtype=np.float32)
        n_nan = int(np.isnan(dist).sum())
        print(f"    label  NaNs : {n_nan}")
        print("-" * 60)
        finite = dist[~np.isnan(dist)]
        if finite.size:
            closest = int(np.nanargmin(dist))
            print(f"  Tip→target  : start {finite[0] * 1e3:.1f} mm")
            print(
                f"                closest {finite.min() * 1e3:.1f} mm "
                f"at frame {closest}/{n - 1}"
            )
            print(f"                final {finite[-1] * 1e3:.1f} mm")
        else:
            print("  Tip→target  : no labelled frames (all NaN)")
        print("-" * 60)

    print("=" * 60)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _t(df: pd.DataFrame) -> np.ndarray:
    return df["timestamp"].to_numpy(dtype=np.float32)


def plot_episode(
    df: pd.DataFrame,
    state: np.ndarray,
    action: np.ndarray,
    meta: dict,
    parquet_path: Path,
    save_path: Path | None,
    show: bool,
) -> None:
    t = _t(df)
    ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else "?"
    task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else "?"
    task_name = meta.get("tasks", {}).get(task_idx, "")

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        f"Episode {ep_idx} — {task_name}\n{parquet_path.name}",
        fontsize=13,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

    def ax(row, col=None, colspan=1):
        if col is None:
            return fig.add_subplot(gs[row, :])
        if colspan == 2:
            return fig.add_subplot(gs[row, :])
        return fig.add_subplot(gs[row, col])

    jnames_arm = ["sh_pit", "sh_rol", "sh_yaw", "elbow", "wr_rol", "wr_pit", "wr_yaw"]
    tip = state[:, STATE_SLICES[definitions.RIGHT_EEF_KEY]]

    # ── 1. Drill-tip pose: measured position vs commanded position ────────
    ax_rp = ax(0, 0)
    ax_arp = ax(0, 1)
    arw_pos = action[:, ACTION_SLICES[definitions.RIGHT_EEF_KEY]][:, :3]
    for i, lbl in enumerate(["x", "y", "z"]):
        ax_rp.plot(t, tip[:, i], label=lbl)
        ax_arp.plot(t, arw_pos[:, i], label=lbl)
    ax_rp.set_title("State: right DRILL TIP position (m)")
    ax_arp.set_title("Action: right drill tip target position (m)")
    for a in (ax_rp, ax_arp):
        a.legend(fontsize=7, ncol=3)
        a.set_xlabel("time (s)")
        a.set_ylabel("m")
        a.grid(True, alpha=0.3)

    # ── 2. Drill-tip rot6d ────────────────────────────────────────────────
    ax_rr = ax(1, colspan=2)
    for i in range(6):
        ax_rr.plot(t, tip[:, 3 + i], label=f"r{i}", alpha=0.8)
    ax_rr.set_title("State: right drill tip rot6d")
    ax_rr.legend(fontsize=7, ncol=6)
    ax_rr.set_xlabel("time (s)")
    ax_rr.grid(True, alpha=0.3)

    # ── 3. Right arm joint angles ─────────────────────────────────────────
    ax_ra = ax(2, colspan=2)
    ra = state[:, STATE_SLICES["right_arm"]]
    for i, jn in enumerate(jnames_arm):
        ax_ra.plot(t, ra[:, i], label=jn, alpha=0.85)
    ax_ra.set_title("State: right arm joints (rad)")
    ax_ra.set_ylabel("rad")
    ax_ra.legend(fontsize=7, ncol=7)
    ax_ra.set_xlabel("time (s)")
    ax_ra.grid(True, alpha=0.3)

    # ── 4. Reward / done ──────────────────────────────────────────────────
    ax_d = ax(3, colspan=2)
    if "next.reward" in df.columns:
        ax_d.plot(t, df["next.reward"].to_numpy(), label="reward", color="green")
    if "next.done" in df.columns:
        done = df["next.done"].to_numpy().astype(float)
        ax_d2 = ax_d.twinx()
        ax_d2.fill_between(t, done, alpha=0.2, color="red", label="done")
        ax_d2.set_ylim(-0.05, 1.3)
        ax_d2.set_yticks([0, 1])
        ax_d2.legend(fontsize=8, loc="upper right")
    ax_d.set_title("Reward / done flag")
    ax_d.legend(fontsize=8, loc="upper left")
    ax_d.set_xlabel("time (s)")
    ax_d.grid(True, alpha=0.3)

    # ── 5. Tip → target task label ─────────────────────────────────────────
    ax_p = ax(4, colspan=2)
    if "next.tip_target_dist" in df.columns:
        dist = df["next.tip_target_dist"].to_numpy(dtype=np.float32)
        ax_p.plot(t, dist * 1e3, label="‖xyz‖", color="purple", linewidth=2)
        if "next.tip_target_9d" in df.columns:
            pose9d = np.stack(df["next.tip_target_9d"].to_numpy()).astype(np.float32)
            for i, (axis, colour) in enumerate(
                zip("xyz", ("tab:red", "tab:green", "tab:blue"))
            ):
                ax_p.plot(
                    t, pose9d[:, i] * 1e3, label=f"{axis}", color=colour, alpha=0.6
                )
        # Shade unlabelled (NaN) frames.
        gaps = np.isnan(dist)
        if gaps.any():
            ax_p.fill_between(
                t,
                0,
                1,
                where=gaps,
                transform=ax_p.get_xaxis_transform(),
                color="red",
                alpha=0.15,
                label="unlabelled (NaN)",
            )
        ax_p.axhline(0.0, color="gray", linewidth=0.8)
    else:
        ax_p.text(
            0.5,
            0.5,
            "no next.tip_target_dist column",
            ha="center",
            va="center",
            transform=ax_p.transAxes,
        )
    ax_p.set_title("Red target in the drill-tip frame (xyz of next.tip_target_9d)")
    ax_p.legend(fontsize=8, loc="upper right", ncol=2)
    ax_p.set_xlabel("time (s)")
    ax_p.set_ylabel("mm")
    ax_p.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a G1 dataset episode.")
    parser.add_argument(
        "--dataset",
        default="/workspace/g1_dataset",
        help="Root directory of the LeRobot-style dataset (default: /workspace/g1_dataset)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to visualize (default: latest)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Direct path to a parquet file (overrides --dataset / --episode)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save figure to this path (e.g. episode_0.png). "
        "Defaults to <parquet_stem>.png next to the parquet file.",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open an interactive plot window"
    )
    args = parser.parse_args()

    parquet_path = _resolve_parquet(args)
    print(f"Loading {parquet_path} …")

    df = pd.read_parquet(parquet_path)

    if df.empty:
        sys.exit("Parquet file is empty.")

    state = _parse_vector_col(df, "observation.state")
    action = _parse_vector_col(df, "action")

    if state.shape[1] != STATE_DIM:
        print(f"Warning: expected state dim {STATE_DIM}, got {state.shape[1]}")
    if action.shape[1] != ACTION_DIM:
        print(f"Warning: expected action dim {ACTION_DIM}, got {action.shape[1]}")

    dataset_root = (
        Path(args.dataset) if not args.path else parquet_path.parent.parent.parent
    )
    ep_idx = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else 0
    meta = _load_meta(dataset_root, ep_idx)

    print_summary(df, state, action, meta, parquet_path)

    show = not args.no_show
    save_path = None
    if args.save:
        save_path = Path(args.save)
    elif args.no_show:
        save_path = parquet_path.with_suffix(".png")

    if not show:
        matplotlib.use("Agg")

    plot_episode(df, state, action, meta, parquet_path, save_path, show)


if __name__ == "__main__":
    main()
