#!/usr/bin/env python3
# MIT License
"""Which VLA sessions placed the drill tip on the target well enough to count.

One line per ``vla_metrics_*.csv``: success means all criteria held for an
unbroken stretch of at least the hold time, read from the target TF:

* ``miss_lateral`` — offset across the wall normal (where the hole lands); tight.
* ``miss_normal`` — offset along the normal (remaining push); looser.
* ``perp_err_deg`` — angle between drill axis and wall normal.

Stdlib only, no ROS needed::

    python3 src/g1_control/g1_control/vla_success.py vla_metrics/2700_vanilla
    python3 -m g1_control.vla_success vla_metrics --lateral-mm 3 --hold-s 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os

_COLUMNS = ("miss_lateral", "miss_normal", "perp_err_deg")


def longest_hold(
    times: list[float],
    inside: list[bool],
    lateral: list[float],
    normal: list[float] | None = None,
) -> tuple:
    """Longest unbroken stretch with every criterion met.

    Returns ``(duration_s, start_t, min_lateral, worst_abs_normal)``. A NaN
    (missing target TF) ends a stretch. Duration is summed from the ``t`` column,
    so irregular sampling is handled. Lateral is best-case, normal worst-case.
    """
    best = (0.0, math.nan, math.nan, math.nan)
    run_start: float | None = None
    run_min = math.inf
    run_worst = -math.inf
    prev_t = math.nan

    padded = list(zip(times, inside, lateral, normal or [math.nan] * len(times)))
    for t, ok, lat, nrm in padded + [(math.nan, False, math.nan, math.nan)]:
        if ok:
            if run_start is None:
                run_start = t
            run_min = min(run_min, lat)
            if not math.isnan(nrm):
                run_worst = max(run_worst, abs(nrm))
            prev_t = t
            continue
        if run_start is not None:
            duration = prev_t - run_start
            if duration > best[0]:
                best = (
                    duration,
                    run_start,
                    run_min,
                    math.nan if run_worst == -math.inf else run_worst,
                )
            run_start = None
            run_min = math.inf
            run_worst = -math.inf
    return best


def _episode(path: str) -> tuple[list[float], dict[str, list[float]], list[str]]:
    """``t`` plus the three criterion columns, NaN where unparseable.

    Also returns the criterion columns missing from the file, so they are
    reported instead of passing silently.
    """

    def value(row: dict, name: str) -> float:
        try:
            return float(row[name])
        except (KeyError, TypeError, ValueError):
            return math.nan

    with open(path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    missing = [c for c in _COLUMNS if c not in header]
    times = [value(r, "t") for r in rows]
    cols = {c: [value(r, c) for r in rows] for c in _COLUMNS}
    return times, cols, missing


def _csv_paths(targets: list[str]) -> list[str]:
    """The session CSVs under ``targets``, minus the companion stats files."""
    found = []
    for target in targets:
        if os.path.isdir(target):
            found += [
                os.path.join(target, n)
                for n in sorted(os.listdir(target))
                if n.endswith(".csv")
            ]
        else:
            found.append(target)
    return [p for p in found if not p.endswith("perception_stats.csv")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "paths",
        nargs="*",
        default=["vla_metrics"],
        help="CSV files and/or directories of them (default: vla_metrics)",
    )
    ap.add_argument(
        "--lateral-mm",
        type=float,
        default=5.0,
        help="max offset across the wall normal (default: 5)",
    )
    ap.add_argument(
        "--depth-mm",
        type=float,
        default=10.0,
        help="max offset along the wall normal (default: 10)",
    )
    ap.add_argument(
        "--perp-deg",
        type=float,
        default=10.0,
        help="max drill-axis tilt off the wall normal (default: 10)",
    )
    ap.add_argument("--hold-s", type=float, default=2.0)
    args = ap.parse_args()

    lateral_m = args.lateral_mm / 1000.0
    depth_m = args.depth_mm / 1000.0
    paths = _csv_paths(args.paths)
    if not paths:
        print("no CSVs found")
        return

    print(
        f"lateral < {args.lateral_mm:g} mm, |normal| < {args.depth_mm:g} mm, "
        f"perp < {args.perp_deg:g} deg, held >= {args.hold_s:g} s "
        f"({len(paths)} sessions)\n"
    )
    width = max(len(os.path.basename(p)) for p in paths)
    passed = 0
    for path in paths:
        times, cols, missing = _episode(path)
        name = os.path.basename(path)
        if missing:
            print(f"SKIP  {name:{width}}  no {', '.join(missing)} column")
            continue
        lat, nrm, perp = (cols[c] for c in _COLUMNS)
        # NaN fails every comparison, which ends the run.
        inside = [
            (a < lateral_m) and (abs(b) < depth_m) and (c < args.perp_deg)
            for a, b, c in zip(lat, nrm, perp)
        ]
        duration, start, run_min, run_worst = longest_hold(times, inside, lat, nrm)
        ok = duration >= args.hold_s
        passed += ok
        measured = [d for d in lat if not math.isnan(d)]
        depths = [d for d in nrm if not math.isnan(d)]
        line = f"{'PASS' if ok else 'fail'}  {name:{width}}  best hold {duration:5.1f} s"
        if measured:
            line += f"  |  min lateral {min(measured) * 1000.0:6.2f} mm"
        # Signed final offset along the normal: positive is short, negative past.
        if depths:
            line += f"  |  final normal {depths[-1] * 1000.0:+6.2f} mm"
        if ok:
            line += (
                f"  |  from t={start:.1f} s at {run_min * 1000.0:.2f} mm lateral, "
                f"|normal| <= {run_worst * 1000.0:.2f} mm"
            )
        print(line)

    print(f"\n{passed}/{len(paths)} sessions succeeded")


if __name__ == "__main__":
    main()
