# MIT License
"""Measure the video<->state time offset by matching projection to pixels.

The target is detected per video frame (largest blob of the task's colour
matching its shape) and compared with the state projection at frame t+k over a
range of shifts k, both over the whole episode (default: full-length scan) and
in sliding windows to catch clock drift.

Healthy: best shift within a frame or two of zero, windows agreeing or
unresolved (``?``, e.g. while the arm holds still). A few px residual no shift
removes is a static geometric bias, not a timing fault.

Default output is one line per episode plus a run verdict; --verbose prints
the scan and window tables for every episode.

Usage:
    python -m dataset_tools.heatmap_labels.sync_check \
        --dataset <dataset> --episodes 0 1 2 3 4 5
    python -m dataset_tools.heatmap_labels.sync_check \
        --dataset <dataset> --episodes 4 --verbose   # tables for a suspect episode
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import camera
from .labels import detect_target_centroid, project_episode
from .overlay_check import episode_paths, episode_target_colour, episode_target_shape
from .projector import TargetProjector

WINDOW = 150
"""Frames per sliding window (5 s at 30 fps)."""

MAX_BASIN_FRAMES = 6
"""Max width of a window's set of equally-good shifts for it to count as resolved."""


def shift_error(uv_proj: np.ndarray, uv_det: np.ndarray, k: int) -> tuple[float, int]:
    """Median |proj(t+k) - detected(t)| in px, and how many frames it used."""
    n = len(uv_proj)
    if k >= 0:
        err = uv_proj[k:n] - uv_det[: n - k]
    else:
        err = uv_proj[: n + k] - uv_det[-k:]
    d = np.linalg.norm(err, axis=1)
    d = d[~np.isnan(d)]
    return (float(np.median(d)) if len(d) else float("nan")), len(d)


def best_shift(
    uv_proj: np.ndarray, uv_det: np.ndarray, shifts: range, min_frames: int
) -> tuple[int | None, float, int]:
    best: tuple[int | None, float, int] = (None, float("inf"), 0)
    for k in shifts:
        med, count = shift_error(uv_proj, uv_det, k)
        if count >= min_frames and med < best[1]:
            best = (k, med, count)
    return best


@dataclass
class EpisodeResult:
    """One episode's measurement; :attr:`problem` holds the verdict."""

    ep: int
    colour: str
    shape: str
    n: int
    n_det: int
    k: int | None
    med: float
    med0: float
    count: int
    count0: int
    rows: list[tuple[int, int | None, float]]

    @property
    def sparse(self) -> bool:
        """Fewer than 1.5 windows of detections: scored but marked as thin."""
        return self.n_det < 3 * WINDOW // 2

    @property
    def resolved(self) -> list[tuple[int, int, float]]:
        """Windows that resolved an offset with a best match <= 5 px."""
        return [(a, k, m) for a, k, m in self.rows if k is not None and m <= 5.0]

    @property
    def drift_checked(self) -> bool:
        """Whether a drift profile was built (False under --no-windows or short
        episodes)."""
        return bool(self.rows)

    @property
    def drift_frames(self) -> int:
        """Spread of resolved window offsets in frames (0 if fewer than two)."""
        known = [k for _, k, _ in self.resolved]
        return max(known) - min(known) if len(known) >= 2 else 0

    @property
    def problem(self) -> str | None:
        """Why this episode needs attention, or None if it is simply healthy."""
        if self.k is None:
            return f"not checked (need {WINDOW} detections)"
        if abs(self.k) > 3:
            side = "behind" if self.k > 0 else "ahead of"
            return f"pairing fault — video {abs(self.k)} frames {side} the states"
        if self.drift_frames > 3:
            return f"drift — offset moves {self.drift_frames} frames across the episode"
        return None


def measure(
    dataset: Path,
    ep: int,
    chunks_size: int,
    projector,
    shifts_arg: int,
    step: int,
    windows: bool,
) -> EpisodeResult:
    """Scan one episode for the shift that best aligns projection to pixels."""
    parquet, video = episode_paths(dataset, ep, chunks_size)
    df = pd.read_parquet(parquet)
    colour = episode_target_colour(dataset, df)
    shape = episode_target_shape(dataset, df)
    n = len(df)

    # Projected centre per state row (raw, no lag correction).
    uv_proj = project_episode(df, projector, target_shape=shape)[["u", "v"]].to_numpy()

    # Detected centre per video frame.
    uv_det = np.full((n, 2), np.nan)
    cap = cv2.VideoCapture(str(video))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        c = detect_target_centroid(frame, colour, shape)
        if c is not None:
            uv_det[i] = c
    cap.release()

    n_det = int(np.isfinite(uv_det).all(axis=1).sum())
    span = shifts_arg if shifts_arg > 0 else n - WINDOW
    shifts = range(-span, span + 1, max(1, step))
    k, med, count = best_shift(uv_proj, uv_det, shifts, WINDOW)
    med0, count0 = shift_error(uv_proj, uv_det, 0)

    rows: list[tuple[int, int | None, float]] = []
    if windows and k is not None:
        for a in range(0, n - WINDOW, WINDOW // 2):
            idx = np.arange(a, a + WINDOW)
            curve: list[tuple[int, float]] = []
            for s in shifts:
                src = idx + s
                ok = (src >= 0) & (src < n)
                d = np.linalg.norm(uv_proj[src[ok]] - uv_det[idx[ok]], axis=1)
                d = d[~np.isnan(d)]
                if len(d) >= WINDOW * 0.4:
                    curve.append((s, float(np.median(d))))
            if not curve:
                rows.append((a, None, float("nan")))
                continue
            s_best, m_best = min(curve, key=lambda c: c[1])
            # Identifiability: resolved only if shifts away from the best fit
            # visibly worse. Tolerance scales with the window's own residual.
            tol = max(1.0, m_best)
            basin = [s for s, m in curve if m <= m_best + tol]
            resolved = max(basin) - min(basin) <= MAX_BASIN_FRAMES
            rows.append((a, s_best if resolved else None, m_best))

    return EpisodeResult(
        ep=ep,
        colour=colour,
        shape=shape,
        n=n,
        n_det=n_det,
        k=k,
        med=med,
        med0=med0,
        count=count,
        count0=count0,
        rows=rows,
    )


HEADER = f"{'ep':>5}  {'target':<15} {'detected':>9} {'shift':>6} {'error':>7}  result"


def compact_row(r: EpisodeResult) -> str:
    det = f"{r.n_det}/{r.n}"
    k_s = "-" if r.k is None else f"{r.k:+d}"
    px_s = "-" if r.k is None else f"{r.med:.1f}px"
    note = r.problem
    if note is None:
        note = "aligned (limited detections)" if r.sparse else "aligned"
    return (
        f"{r.ep:>5}  {r.colour + ' ' + r.shape:<15} {det:>9} {k_s:>6} {px_s:>7}  {note}"
    )


def print_profile(r: EpisodeResult, indent: str = "    ") -> None:
    """Print the per-window offset table (``?`` = unresolved window)."""
    if not r.rows:
        return
    print(f"{indent}drift profile ({WINDOW}-frame windows):")
    print(f"{indent}  frame :" + "".join(f"{a:>7d}" for a, _, _ in r.rows))
    print(
        f"{indent}  shift :"
        + "".join(
            f"{k:>+7d}" if (k is not None and m <= 5.0) else "      ?"
            for _, k, m in r.rows
        )
    )
    print(
        f"{indent}  px    :"
        + "".join(f"{m:>7.1f}" if np.isfinite(m) else "      -" for _, _, m in r.rows)
    )
    n_unresolved = len(r.rows) - len(r.resolved)
    if n_unresolved:
        print(
            f"{indent}  ({n_unresolved} window(s) marked ? did not resolve — the "
            "projection holds too still there to tell one shift from another)"
        )


def print_verbose(r: EpisodeResult, fps: float) -> None:
    """Print the full scan and window table for one episode."""
    print(
        f"ep {r.ep} [{r.colour} {r.shape}]: {r.n} frames, {r.n_det} with a detected target"
    )
    if r.sparse:
        print(f"  limited detections: {r.n_det}/{r.n}")
    if r.k is None:
        print(f"  not checked: at least {WINDOW} detections are required")
        return
    print(f"  k=0      : {r.med0:7.2f} px  (n={r.count0})")
    print(f"  best k={r.k:+5d}: {r.med:7.2f} px  (n={r.count})  [{r.k / fps:+.2f} s]")
    if abs(r.k) > 3:
        print(
            f"  ^ the video is {abs(r.k)} frames ({abs(r.k) / fps:.1f} s) "
            f"{'behind' if r.k > 0 else 'ahead of'} the states. That is a "
            "recorder pairing fault, not render lag — re-record rather than "
            "shifting the labels."
        )
    print_profile(r, indent="  ")
    if r.drift_frames > 3:
        first, last = r.resolved[0], r.resolved[-1]
        drift = (last[1] - first[1]) / max(1, last[0] - first[0])
        print(
            f"    ^ the offset moves by {r.drift_frames} frames across the "
            f"episode ({drift * 100:+.1f}% of elapsed time) — the two clocks "
            "are running at different rates."
        )


def print_summary(results: list[EpisodeResult]) -> None:
    """Print the alignment result for the run."""
    scored = [r for r in results if r.k is not None]
    unchecked = [r for r in results if r.k is None]
    print()
    if not scored:
        print(f"Summary: 0/{len(results)} checked; not enough target detections.")
        return

    offsets = sorted({r.k for r in scored})
    span = (
        f"{offsets[0]:+d}"
        if len(offsets) == 1
        else f"{offsets[0]:+d}..{offsets[-1]:+d}"
    )
    drifting = [r for r in scored if r.drift_frames > 3]
    profiled = [r for r in scored if r.drift_checked]
    pairing = [r for r in scored if abs(r.k) > 3]

    print(
        f"Summary: {len(scored)}/{len(results)} checked, "
        f"{len(unchecked)} not checked; offset {span} frame(s)."
    )
    if drifting or pairing:
        failed = sorted({r.ep for r in drifting + pairing})
        print(f"Result: re-record episodes {failed}.")
    elif max(abs(o) for o in offsets) > 2:
        print(f"Result: consistent {span}-frame offset; investigate.")
    elif not profiled:
        print("Result: small offset; drift was not checked.")
    elif unchecked:
        print("Result: checked episodes are aligned; others need more detections.")
    else:
        print("Result: video and states are aligned.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="test k in [-N, N]; 0 (default) scans the whole episode",
    )
    ap.add_argument(
        "--step", type=int, default=1, help="shift granularity for the coarse scan"
    )
    ap.add_argument(
        "--no-windows",
        action="store_true",
        help="skip the drift profile; the verdict then reports drift as "
        "unmeasured rather than clean",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="full scan and window table for every episode, healthy ones included",
    )
    args = ap.parse_args()
    print(f"Camera: {camera.VIDEO_KEY}", flush=True)

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    fps = info.get("fps", 30)
    projector = TargetProjector()

    if not args.verbose:
        print(HEADER)
    results = []
    for ep in args.episodes:
        r = measure(
            args.dataset,
            ep,
            chunks_size,
            projector,
            args.shifts,
            args.step,
            not args.no_windows,
        )
        results.append(r)
        if args.verbose:
            print_verbose(r, fps)
        else:
            print(compact_row(r), flush=True)
            # Tables only for problematic episodes.
            if r.problem and r.k is not None:
                print_profile(r)
    print_summary(results)


if __name__ == "__main__":
    main()
