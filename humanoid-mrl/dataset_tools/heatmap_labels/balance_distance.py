# MIT License
"""Rebalance a heatmap-label sidecar over tip-target distance.

Most visible frames sit within 1 cm of the target, where it lands near a fixed
wrist pixel, so a language-blind constant predictor scores well. This tool drops
rows from the sidecar parquets (the training-sample mask under
``perception_labels_only``) to flatten that distribution; labels are untouched.

- Rows are binned by ``tip_target_dist``; each bin is capped at a quota and
  sampled round-robin across episodes.
- Visible and non-visible rows are capped separately, preserving the negative
  share for the visibility BCE.

Usage:
    # look first -- prints the before/after distribution, writes nothing
    python -m dataset_tools.heatmap_labels.balance_distance \
        --dataset data/small_targets/g1_dataset_autocollect_v2_rightfreq --report-only

    # write the pruned sidecar next to the original
    python -m dataset_tools.heatmap_labels.balance_distance \
        --dataset data/small_targets/g1_dataset_autocollect_v2_rightfreq

    # ... and swap it in (the loader only reads `heatmap_labels/`)
    python -m dataset_tools.heatmap_labels.balance_distance \
        --dataset data/small_targets/g1_dataset_autocollect_v2_rightfreq --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR_NAME = "heatmap_labels"
_DEFAULT_OUT_DIR = "heatmap_labels_balanced"
_BACKUP_DIR_NAME = "heatmap_labels_full"

_INDEX_COLS = ["frame_index", "tip_target_dist", "visible", "u", "v"]

_CONTACT_DIST_M = 0.002
"""Rows this close to the target define the drill-tip pixel (report only)."""


def sidecar_path(labels_root: Path, episode: int, chunks_size: int) -> Path:
    return labels_root / f"chunk-{episode // chunks_size:03d}" / f"episode_{episode:06d}.parquet"


def load_index(labels_root: Path, episodes: list[int], chunks_size: int) -> pd.DataFrame:
    """One row per sidecar row: episode, frame_index, distance, visibility, uv."""
    frames: list[pd.DataFrame] = []
    missing = 0
    t0 = time.time()
    live = sys.stdout.isatty()
    for n, ep in enumerate(episodes):
        path = sidecar_path(labels_root, ep, chunks_size)
        if not path.exists():
            missing += 1
            continue
        df = pd.read_parquet(path, columns=_INDEX_COLS)
        df["episode"] = ep
        frames.append(df)
        if live and (n + 1) % 100 == 0:
            rate = (n + 1) / max(time.time() - t0, 1e-9)
            print(
                f"\rscanning [{n + 1}/{len(episodes)}] {rate:.0f} eps/s", end="", flush=True
            )
    if live:
        print()
    if missing:
        print(f"warning: {missing} episodes have no sidecar and are ignored", flush=True)
    if not frames:
        raise SystemExit(f"no sidecar parquets found under {labels_root}")
    index = pd.concat(frames, ignore_index=True)
    index["visible"] = index["visible"].fillna(False).astype(bool)
    return index


def make_edges(bin_width: float, max_dist: float, explicit: list[float] | None) -> np.ndarray:
    """Bin edges in metres, closed by an overflow bin at +inf past ``max_dist``."""
    if explicit:
        edges = np.asarray(sorted(explicit), dtype=float)
    else:
        n = int(round(max_dist / bin_width))
        edges = np.linspace(0.0, n * bin_width, n + 1)
    return np.concatenate((edges, [np.inf]))


def bin_of(dist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per row; NaN and out-of-range distances fall in the overflow bin."""
    d = np.nan_to_num(np.asarray(dist, dtype=float), nan=np.inf)
    return np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)


def pick_quota(counts: np.ndarray, quantile: float, explicit: int | None) -> int:
    """Per-bin cap. Default: the given quantile of the non-empty bin sizes."""
    if explicit is not None:
        return int(explicit)
    non_empty = counts[counts > 0]
    if non_empty.size == 0:
        return 0
    return int(max(1, round(float(np.quantile(non_empty, quantile)))))


def round_robin_mask(episode_ids: np.ndarray, quota: int, rng: np.random.Generator) -> np.ndarray:
    """Keep ``quota`` rows, taken one per episode per pass."""
    n = episode_ids.size
    if n <= quota:
        return np.ones(n, dtype=bool)
    perm = rng.permutation(n)
    shuffled = episode_ids[perm]
    order = np.argsort(shuffled, kind="stable")
    grouped = shuffled[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(grouped)) + 1))
    lengths = np.diff(np.concatenate((starts, [n])))
    ranks_sorted = np.arange(n) - np.repeat(starts, lengths)
    ranks = np.empty(n, dtype=np.int64)
    ranks[perm[order]] = ranks_sorted
    # Random tiebreak within a rank so the mid-pass cut is not biased by episode id.
    keep = np.lexsort((rng.random(n), ranks))[:quota]
    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    return mask


def select(
    index: pd.DataFrame, edges: np.ndarray, quota: int, neg_quota: int, seed: int
) -> np.ndarray:
    """Boolean keep-mask over ``index`` rows, capping each (bin, visibility) cell."""
    rng = np.random.default_rng(seed)
    bins = bin_of(index["tip_target_dist"].to_numpy(), edges)
    episodes = index["episode"].to_numpy()
    visible = index["visible"].to_numpy()
    keep = np.zeros(len(index), dtype=bool)
    for is_visible, cap in ((True, quota), (False, neg_quota)):
        for b in range(len(edges) - 1):
            rows = np.flatnonzero((bins == b) & (visible == is_visible))
            if rows.size == 0 or cap == 0:
                continue
            keep[rows[round_robin_mask(episodes[rows], cap, rng)]] = True
    return keep


def histogram_table(index: pd.DataFrame, edges: np.ndarray, keep: np.ndarray) -> str:
    """Per-bin visible/negative counts before and after, as a printable table."""
    bins = bin_of(index["tip_target_dist"].to_numpy(), edges)
    visible = index["visible"].to_numpy()
    lines = [
        f"{'distance bin':>16}  {'visible':>9} {'->':^4} {'kept':>8} {'share':>7}   "
        f"{'neg':>9} {'->':^4} {'kept':>8}"
    ]
    kept_vis_total = int((keep & visible).sum())
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        label = f"{lo * 100:5.1f}-{hi * 100:5.1f}cm" if np.isfinite(hi) else f">{lo * 100:5.1f}cm  "
        in_bin = bins == b
        v0 = int((in_bin & visible).sum())
        v1 = int((in_bin & visible & keep).sum())
        n0 = int((in_bin & ~visible).sum())
        n1 = int((in_bin & ~visible & keep).sum())
        share = 100.0 * v1 / max(kept_vis_total, 1)
        lines.append(
            f"{label:>16}  {v0:9d} {'->':^4} {v1:8d} {share:6.1f}%   {n0:9d} {'->':^4} {n1:8d}"
        )
    return "\n".join(lines)


def shortcut_report(index: pd.DataFrame, keep: np.ndarray) -> str:
    """How well a fixed-pixel, language-blind predictor does, before vs after.

    The drill-tip pixel is the median uv of contact rows.
    """
    visible = index["visible"].to_numpy()
    dist = index["tip_target_dist"].to_numpy()
    u = index["u"].to_numpy()
    v = index["v"].to_numpy()
    contact = visible & (dist < _CONTACT_DIST_M) & np.isfinite(u)
    if contact.sum() == 0:
        return "shortcut report: no contact frames found, skipped"
    cu, cv = float(np.median(u[contact])), float(np.median(v[contact]))
    out = [f"constant-predictor baseline at the drill-tip pixel ({cu:.0f}, {cv:.0f}):"]
    for name, sel in (("before", visible), ("after ", visible & keep)):
        rows = sel & np.isfinite(u)
        if rows.sum() == 0:
            continue
        err = np.hypot(u[rows] - cu, v[rows] - cv)
        out.append(
            f"  {name}  n={rows.sum():8d}  median {np.median(err):6.1f} px  "
            f"mean {err.mean():6.1f} px  within 50 px: {100 * (err < 50).mean():5.1f}%"
        )
    out.append("  (higher is better here: it is the error the head CANNOT get for free)")
    return "\n".join(out)


def write_sidecar(
    src_root: Path, out_root: Path, index: pd.DataFrame, keep: np.ndarray, chunks_size: int
) -> dict:
    """Copy each episode's sidecar with only the kept rows."""
    kept = index.loc[keep, ["episode", "frame_index"]]
    by_episode = {ep: set(g) for ep, g in kept.groupby("episode")["frame_index"]}
    stats = {"episodes": 0, "frames": 0, "visible": 0, "episodes_dropped": 0}
    live = sys.stdout.isatty()
    episodes = sorted(index["episode"].unique())
    t0 = time.time()
    for n, ep in enumerate(episodes):
        frames = by_episode.get(ep)
        if not frames:
            stats["episodes_dropped"] += 1
            continue
        df = pd.read_parquet(sidecar_path(src_root, ep, chunks_size))
        df = df[df["frame_index"].isin(frames)].reset_index(drop=True)
        out_dir = out_root / f"chunk-{ep // chunks_size:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"episode_{ep:06d}.parquet")
        stats["episodes"] += 1
        stats["frames"] += len(df)
        stats["visible"] += int(df["visible"].sum())
        if live and (n + 1) % 100 == 0:
            rate = (n + 1) / max(time.time() - t0, 1e-9)
            print(f"\rwriting [{n + 1}/{len(episodes)}] {rate:.0f} eps/s", end="", flush=True)
    if live:
        print()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=f"sidecar to rebalance (default <dataset>/{_SRC_DIR_NAME})",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help=f"output root (default <dataset>/{_DEFAULT_OUT_DIR})"
    )
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--bin-width", type=float, default=0.01, help="distance bin width [m]")
    ap.add_argument(
        "--max-dist", type=float, default=0.15, help="last binned distance; past it is one bin [m]"
    )
    ap.add_argument(
        "--edges",
        type=float,
        nargs="*",
        default=None,
        help="explicit bin edges in metres, overriding --bin-width/--max-dist",
    )
    ap.add_argument(
        "--per-bin",
        type=int,
        default=None,
        help="visible rows to keep per bin (default: --cap-quantile of the bin sizes)",
    )
    ap.add_argument(
        "--cap-quantile",
        type=float,
        default=0.5,
        help="quantile of non-empty bin sizes used as the cap; lower = flatter and smaller",
    )
    ap.add_argument(
        "--neg-ratio",
        type=float,
        default=None,
        help="non-visible cap as a fraction of the visible cap "
        "(default: the source's own negative:visible ratio)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true", help="print the plan, write nothing")
    ap.add_argument(
        "--overwrite", action="store_true", help="replace an existing --out directory"
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help=f"after writing, move the original to {_BACKUP_DIR_NAME}/ and the balanced "
        f"sidecar into {_SRC_DIR_NAME}/, which is the only name the loader reads",
    )
    args = ap.parse_args()

    src_root = args.labels or (args.dataset / _SRC_DIR_NAME)
    out_root = args.out or (args.dataset / _DEFAULT_OUT_DIR)
    if out_root.resolve() == src_root.resolve():
        raise SystemExit("--out must differ from the source sidecar")

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    episodes = (
        args.episodes if args.episodes is not None else list(range(info["total_episodes"]))
    )

    index = load_index(src_root, episodes, chunks_size)
    edges = make_edges(args.bin_width, args.max_dist, args.edges)
    bins = bin_of(index["tip_target_dist"].to_numpy(), edges)
    visible = index["visible"].to_numpy()

    counts = np.array([int(((bins == b) & visible).sum()) for b in range(len(edges) - 1)])
    quota = pick_quota(counts, args.cap_quantile, args.per_bin)
    neg_ratio = (
        args.neg_ratio
        if args.neg_ratio is not None
        else (~visible).sum() / max(visible.sum(), 1)
    )
    neg_quota = int(round(quota * neg_ratio))

    keep = select(index, edges, quota, neg_quota, args.seed)

    n_eps = index["episode"].nunique()
    kept_eps = index.loc[keep, "episode"].nunique()
    print(f"\nsource: {src_root}")
    print(
        f"bins: {len(edges) - 1} over 0-{args.max_dist:.2f} m + overflow | "
        f"cap {quota} visible / {neg_quota} negative per bin (neg ratio {neg_ratio:.2f})"
    )
    print()
    print(histogram_table(index, edges, keep))
    print()
    print(
        f"rows      {len(index):9d} -> {int(keep.sum()):9d}  ({100 * keep.mean():.1f}%)\n"
        f"visible   {int(visible.sum()):9d} -> {int((keep & visible).sum()):9d}\n"
        f"episodes  {n_eps:9d} -> {kept_eps:9d}  "
        f"({int(keep.sum()) / max(kept_eps, 1):.0f} rows/episode)"
    )
    print()
    print(shortcut_report(index, keep))

    if args.report_only:
        print("\n--report-only: nothing written")
        return

    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"{out_root} already exists; pass --overwrite to replace it "
                "(re-running with a different --cap-quantile is the usual reason)"
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    stats = write_sidecar(src_root, out_root, index, keep, chunks_size)

    src_info_path = src_root / "labels_info.json"
    meta = json.loads(src_info_path.read_text()) if src_info_path.exists() else {}
    meta["balanced_from"] = str(src_root)
    meta["balance"] = {
        "bin_edges_m": [None if np.isinf(e) else float(e) for e in edges],
        "per_bin_visible": quota,
        "per_bin_negative": neg_quota,
        "neg_ratio": float(neg_ratio),
        "cap_quantile": args.cap_quantile,
        "seed": args.seed,
    }
    meta["stats"] = stats
    (out_root / "labels_info.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {stats} -> {out_root}")

    if args.apply:
        backup = args.dataset / _BACKUP_DIR_NAME
        if backup.exists():
            raise SystemExit(f"{backup} already exists; not overwriting it")
        shutil.move(str(src_root), str(backup))
        shutil.move(str(out_root), str(src_root))
        print(
            f"applied: {backup.name}/ holds the full sidecar, {src_root.name}/ is now balanced\n"
            f"  undo with: rm -rf {src_root} && mv {backup} {src_root}"
        )


if __name__ == "__main__":
    main()
