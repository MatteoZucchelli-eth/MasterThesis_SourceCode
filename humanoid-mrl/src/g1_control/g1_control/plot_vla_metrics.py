#!/usr/bin/env python3
# MIT License
"""Render the drill-tip tracking PNGs for a MetricRecorder CSV.

Reads a ``vla_metrics_*.csv`` / ``progress_test_*.csv`` and saves alongside it:

  * ``*_tracking.png``        commanded vs. measured tip position (x/y/z);
  * ``*_tracking_error.png``  per-axis + L2 tracking error over time;
  * ``*_target_distance.png`` wall-target distance to the measured tip and to
    the commanded target, each minimum L2 clearly annotated;
  * ``*_relative_pose.png``   predicted vs. measured target pose relative to
    the drill tip (x/y/z);
  * ``*_relative_error.png``  the perception error over time — predicted vs.
    actual relative target pose (x/y/z, L2 and x-axis angle), with the mean ± std
    of each segment drawn over it;
  * ``*_perception_stats.csv`` those means and standard deviations as numbers,
    overall and per scripted-sweep waypoint.

Runs automatically on VLA exit (spawned by ``MetricRecorder.finalize``) or by
hand::

    python -m g1_control.plot_vla_metrics vla_metrics/vla_metrics_<stamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g1_control.utils.perception_stats import (
    format_stats,
    perception_error_stats,
    pick_group_column,
    stats_rows,
)

_AXIS_COLORS = ("tab:green", "tab:orange", "tab:purple")


def _load(csv_path: str) -> dict[str, np.ndarray]:
    """Read the CSV into a dict of float columns (NaN where unparseable)."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    def column(name: str) -> np.ndarray:
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return np.asarray(out)

    return {c: column(c) for c in fields}


def _decorate(ax, bounds: list[float]) -> None:
    """Grid + dashed vertical lines at the scripted-sweep waypoint changes."""
    ax.grid(True, alpha=0.3)
    for b in bounds:
        ax.axvline(b, color="gray", ls="--", lw=0.6, alpha=0.6)


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_pair_xyz(
    d, t, bounds, out, fmt_a, fmt_b, label_a, label_b, scale, unit, title
):
    """Three stacked panels comparing two x/y/z series (solid vs dashed)."""
    if any(fmt.format(a) not in d for fmt in (fmt_a, fmt_b) for a in "xyz"):
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, axis in zip(axes, "xyz"):
        ax.plot(t, d[fmt_a.format(axis)] * scale, color="tab:blue", label=label_a)
        ax.plot(
            t, d[fmt_b.format(axis)] * scale, color="tab:orange", ls="--", label=label_b
        )
        ax.set_ylabel(f"{axis} ({unit})")
        _decorate(ax, bounds)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    _save(fig, out)


def _plot_error_xyz(
    d, t, bounds, out, err_fmt, bottom_col, bottom_scale, bottom_label, title
):
    """Per-axis error (mm) on top, a scalar error norm on the bottom."""
    if bottom_col not in d or any(err_fmt.format(a) not in d for a in "xyz"):
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, color in zip("xyz", _AXIS_COLORS):
        ax0.plot(t, d[err_fmt.format(axis)] * 1000.0, color=color, label=f"err {axis}")
    ax0.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax0.set_ylabel("per-axis error (mm)")
    ax0.legend(loc="upper right", fontsize=8)
    _decorate(ax0, bounds)

    ax1.plot(t, d[bottom_col] * bottom_scale, color="tab:red")
    ax1.set_ylabel(bottom_label)
    ax1.set_ylim(bottom=0)
    _decorate(ax1, bounds)
    ax1.set_xlabel("time (s)")
    fig.suptitle(title)
    _save(fig, out)


def _plot_target_distance(d, t, bounds, out):
    """Wall-target distances with the minimum of each series annotated."""
    series = [
        ("dist_target_actual", "target → actual tip", "tab:blue"),
        ("dist_target_cmd", "target → commanded tip", "tab:orange"),
    ]
    series = [s for s in series if s[0] in d and np.any(np.isfinite(d[s[0]]))]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = []
    for col, label, color in series:
        y = d[col]
        ax.plot(t, y, color=color, label=label)
        i = int(np.nanargmin(y))
        ax.axhline(y[i], color=color, ls=":", lw=1.0, alpha=0.7)
        ax.scatter([t[i]], [y[i]], color=color, s=40, zorder=5, edgecolor="k", lw=0.5)
        summary.append(f"{label} min: {y[i] * 1000:.1f} mm @ t={t[i]:.1f}s")
    ax.set_ylabel("distance to wall target (m)")
    ax.set_xlabel("time (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    _decorate(ax, bounds)
    ax.text(
        0.01,
        0.98,
        "\n".join(summary),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    fig.suptitle("Drill-tip distance to wall target (min L2)")
    _save(fig, out)


def _segments(t: np.ndarray, keys: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Split the session into (t_start, t_end, mask) runs of constant ``keys``.

    No finite keys yields one whole-session run; NaN keys compare equal.
    """
    if len(t) == 0:
        return []
    keys = None if keys is None else np.asarray(keys, dtype=float)
    if keys is None or len(keys) != len(t) or not np.isfinite(keys).any():
        return [(float(t[0]), float(t[-1]), np.ones(len(t), dtype=bool))]
    # NaN -> sentinel so NaN runs compare equal.
    keys = np.where(np.isfinite(keys), keys, -np.inf)

    out = []
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or keys[i] != keys[start]:
            mask = np.zeros(len(t), dtype=bool)
            mask[start:i] = True
            out.append((float(t[start]), float(t[i - 1]), mask))
            start = i
    return out


def _band(ax, t0: float, t1: float, values: np.ndarray, color: str) -> None:
    """Draw ``mean`` as a line and ``mean ± std`` as a shaded band over [t0, t1]."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return
    mean, std = float(np.mean(v)), float(np.std(v))
    ax.hlines(mean, t0, t1, color=color, lw=1.6, alpha=0.9, zorder=4)
    ax.fill_between(
        [t0, t1], mean - std, mean + std, color=color, alpha=0.15, lw=0, zorder=1
    )


def _plot_perception_error(d, t, bounds, out, stats, group_column) -> None:
    """Perception error over time with per-segment mean ± std bands.

    Panels: per-axis error (mm), L2 (mm), x-axis angle error (deg).
    """
    axes_cols = [f"rel_err_{a}" for a in "xyz"]
    if any(c not in d for c in axes_cols):
        return

    l2 = np.linalg.norm(np.stack([d[c] for c in axes_cols]), axis=0) * 1000.0
    x_ang = d.get("rel_err_xaxis_deg")
    segments = _segments(t, d.get(group_column))

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for axis, col, color in zip("xyz", axes_cols, _AXIS_COLORS):
        y = d[col] * 1000.0
        ax0.plot(t, y, color=color, lw=0.9, alpha=0.55, label=f"err {axis}")
        for t0, t1, mask in segments:
            _band(ax0, t0, t1, y[mask], color)
    ax0.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax0.set_ylabel("per-axis error (mm)")
    ax0.legend(loc="upper right", fontsize=8, title="line: mean, band: ±1 std")
    _decorate(ax0, bounds)

    ax1.plot(t, l2, color="tab:red", lw=0.9, alpha=0.55)
    for t0, t1, mask in segments:
        _band(ax1, t0, t1, l2[mask], "tab:red")
    ax1.set_ylabel("L2 error (mm)")
    ax1.set_ylim(bottom=0)
    _decorate(ax1, bounds)

    if x_ang is not None:
        ax2.plot(t, x_ang, color="tab:blue", lw=0.9, alpha=0.55)
        for t0, t1, mask in segments:
            _band(ax2, t0, t1, x_ang[mask], "tab:blue")
    ax2.set_ylabel("x-axis error (deg)")
    ax2.set_ylim(bottom=0)
    _decorate(ax2, bounds)
    ax2.set_xlabel("time (s)")

    overall = next((s for s in stats if s.group == "all"), None)
    if overall is not None and overall.channels:
        ax0.text(
            0.01,
            0.98,
            "overall\n"
            + "\n".join(
                f"{c.label}: {c.mean:+.2f} ± {c.std:.2f} {c.unit}"
                for c in overall.channels.values()
            ),
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    fig.suptitle("Perception error over time (predicted − actual, drill-tip frame)")
    _save(fig, out)


def _write_stats_csv(stats, path: str) -> None:
    """Write the perception-error mean/std table beside the session CSV."""
    if not stats:
        return
    header, rows = stats_rows(stats)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="Path to a vla_metrics_*/progress_test_* CSV")
    args = ap.parse_args()

    d = _load(args.csv_path)
    if "t" not in d or len(d["t"]) == 0:
        print(f"No samples in {args.csv_path}; nothing to plot.")
        return
    t = d["t"]
    # Segment boundaries: waypoint changes, else planner phase transitions.
    wp = d.get("waypoint_idx", np.array([]))
    bounds = [float(t[i]) for i in range(1, len(wp)) if wp[i] != wp[i - 1]]
    phase = d.get("phase", np.array([]))
    if not bounds and np.isfinite(phase).any():
        bounds = [float(t[i]) for i in range(1, len(phase)) if phase[i] != phase[i - 1]]
    base = os.path.splitext(args.csv_path)[0]

    _plot_pair_xyz(
        d,
        t,
        bounds,
        f"{base}_tracking.png",
        "cmd_{}",
        "act_{}",
        "commanded",
        "actual",
        1.0,
        "m",
        "Drill-tip pose: commanded vs. actual",
    )
    _plot_error_xyz(
        d,
        t,
        bounds,
        f"{base}_tracking_error.png",
        "track_err_{}",
        "track_err_l2",
        1000.0,
        "L2 error (mm)",
        "Drill-tip tracking error (commanded − actual)",
    )
    _plot_target_distance(d, t, bounds, f"{base}_target_distance.png")
    _plot_pair_xyz(
        d,
        t,
        bounds,
        f"{base}_relative_pose.png",
        "pred_rel_{}",
        "act_rel_{}",
        "predicted",
        "actual",
        1000.0,
        "mm",
        "Target pose relative to drill tip: predicted vs. actual",
    )
    group_column = pick_group_column(d)
    stats = perception_error_stats(d, group_by=group_column)
    _plot_perception_error(
        d, t, bounds, f"{base}_relative_error.png", stats, group_column
    )
    _write_stats_csv(stats, f"{base}_perception_stats.csv")
    print(format_stats(stats))


if __name__ == "__main__":
    main()
