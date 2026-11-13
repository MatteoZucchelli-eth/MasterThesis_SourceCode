# MIT License
"""Aggregate statistics for the perception (tip->target estimate) error.

Summarises the MetricRecorder's ``rel_err_*`` columns (predicted minus measured
target pose, drill-tip frame) overall and per group. Shared by
:mod:`g1_control.states.metric_recorder` and :mod:`g1_control.plot_vla_metrics`;
no matplotlib/ROS imports (runs in the realtime control process).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Error channels summarised, as (column, label, scale, unit). The x/y/z columns
# are metres and the x-axis orientation error is already degrees.
_CHANNELS = (
    ("rel_err_x", "x", 1000.0, "mm"),
    ("rel_err_y", "y", 1000.0, "mm"),
    ("rel_err_z", "z", 1000.0, "mm"),
    ("rel_err_l2", "L2", 1000.0, "mm"),
    ("rel_err_xaxis_deg", "x-axis", 1.0, "deg"),
)

# Tip->target distance column, reported per group.
_DISTANCE_COLUMN = "dist_target_actual"


@dataclass
class ChannelStats:
    """Mean/std (and count) of one error channel, in the channel's unit."""

    label: str
    unit: str
    mean: float
    std: float
    rms: float
    abs_mean: float
    max_abs: float
    count: int

    def __str__(self) -> str:
        return (
            f"{self.label}: {self.mean:+7.2f} ± {self.std:5.2f} {self.unit} "
            f"(|mean| {self.abs_mean:5.2f}, rms {self.rms:5.2f}, "
            f"max {self.max_abs:5.2f}, n={self.count})"
        )


@dataclass
class GroupStats:
    """Perception-error statistics for one group of samples."""

    group: str  # "all", or "waypoint <i>" / "phase <p>"
    samples: int  # samples in the group (before per-channel NaN drops)
    distance_mean: float  # mean tip->target distance over the group (m), NaN if absent
    channels: dict[str, ChannelStats] = field(default_factory=dict)

    def format(self) -> str:
        """Multi-line ``mean ± std`` block for the logger / stdout."""
        dist = (
            f", tip→target {self.distance_mean * 1000.0:.0f} mm"
            if np.isfinite(self.distance_mean)
            else ""
        )
        head = f"{self.group} (n={self.samples}{dist})"
        if not self.channels:
            return f"{head}: no finite perception-error samples"
        body = "\n    ".join(str(c) for c in self.channels.values())
        return f"{head}\n    {body}"


def _channel_stats(label: str, unit: str, values: np.ndarray) -> ChannelStats | None:
    """Summarise one channel, dropping NaNs; None when nothing is finite.

    ``std`` is the population standard deviation (ddof=0).
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    return ChannelStats(
        label=label,
        unit=unit,
        mean=float(np.mean(v)),
        std=float(np.std(v)),
        rms=float(np.sqrt(np.mean(np.square(v)))),
        abs_mean=float(np.mean(np.abs(v))),
        max_abs=float(np.max(np.abs(v))),
        count=int(v.size),
    )


def _group_stats(group: str, columns: dict[str, np.ndarray]) -> GroupStats:
    """Build the per-channel stats for one already-selected group of samples."""
    n = len(next(iter(columns.values()))) if columns else 0
    dist = columns.get(_DISTANCE_COLUMN)
    if dist is not None and np.isfinite(dist).any():
        distance_mean = float(np.mean(dist[np.isfinite(dist)]))
    else:
        distance_mean = float("nan")

    stats = GroupStats(group=group, samples=n, distance_mean=distance_mean)
    for col, label, scale, unit in _CHANNELS:
        values = columns.get(col)
        if values is None:
            continue
        cs = _channel_stats(label, unit, np.asarray(values, dtype=float) * scale)
        if cs is not None:
            stats.channels[label] = cs
    return stats


def _with_l2(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add the derived ``rel_err_l2`` column (norm of the x/y/z error;
    NaN-propagating)."""
    axes = [f"rel_err_{a}" for a in "xyz"]
    if any(a not in columns for a in axes):
        return columns
    stacked = np.stack([np.asarray(columns[a], dtype=float) for a in axes])
    return {**columns, "rel_err_l2": np.linalg.norm(stacked, axis=0)}


def pick_group_column(columns: dict[str, np.ndarray]) -> str:
    """Column that splits a session into segments.

    ``waypoint_idx`` when it varies (scripted sweep), otherwise ``phase``.
    """
    wp = columns.get("waypoint_idx")
    if wp is not None:
        distinct = {
            float(v) for v in np.asarray(wp, float) if np.isfinite(v) and v >= 0
        }
        if len(distinct) > 1:
            return "waypoint_idx"
    return "phase"


def perception_error_stats(
    columns: dict[str, np.ndarray],
    group_by: str = "waypoint_idx",
) -> list[GroupStats]:
    """Mean/std of the perception error, overall and per group.

    Parameters
    ----------
    columns:
        Column-name -> float array, as produced by ``plot_vla_metrics._load``
        or ``MetricRecorder`` (all arrays the same length).
    group_by:
        Column whose distinct finite values split the session into groups
        (``waypoint_idx`` or ``phase``). Absent or constant -> overall row only.

    Returns
    -------
    The ``"all"`` group first, then one group per distinct value in ascending
    order. Empty when the CSV carries no ``rel_err_*`` columns at all.
    """
    columns = _with_l2(columns)
    if not any(col in columns for col, *_ in _CHANNELS):
        return []

    out = [_group_stats("all", columns)]

    keys = columns.get(group_by)
    if keys is None:
        return out
    keys = np.asarray(keys, dtype=float)
    values = sorted({float(k) for k in keys if np.isfinite(k) and k >= 0})
    if len(values) < 2:
        return out

    label = "waypoint" if group_by == "waypoint_idx" else group_by
    for value in values:
        mask = keys == value
        out.append(
            _group_stats(f"{label} {value:g}", {c: v[mask] for c, v in columns.items()})
        )
    return out


def format_stats(stats: list[GroupStats]) -> str:
    """Render ``perception_error_stats`` output as a console block."""
    if not stats:
        return "perception error: no rel_err_* samples recorded"
    body = "\n  ".join(s.format() for s in stats)
    return (
        "perception error (predicted − actual target pose, drill-tip frame)\n  " + body
    )


def stats_rows(stats: list[GroupStats]) -> tuple[list[str], list[list]]:
    """Flatten the stats into a (header, rows) pair for a CSV writer."""
    header = [
        "group",
        "samples",
        "dist_target_actual_mean_m",
        "channel",
        "unit",
        "mean",
        "std",
        "rms",
        "abs_mean",
        "max_abs",
        "count",
    ]
    rows = []
    for g in stats:
        for c in g.channels.values():
            rows.append(
                [
                    g.group,
                    g.samples,
                    g.distance_mean,
                    c.label,
                    c.unit,
                    c.mean,
                    c.std,
                    c.rms,
                    c.abs_mean,
                    c.max_abs,
                    c.count,
                ]
            )
    return header, rows
