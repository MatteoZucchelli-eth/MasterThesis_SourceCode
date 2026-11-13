#!/usr/bin/env python3
"""Write a compact Markdown summary for a directory of VLA evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re


def _finite(values):
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


# MetricRecorder CSVs: "<prefix>_<tag>_<date>_<time>.csv"; the prefix (state
# name) is ignored so one report covers every state.
_SESSION_CSV = re.compile(r"_(s\d+-[a-z0-9-]+)_\d{8}_\d{6}\.csv$")


def _session_tag(basename: str) -> str:
    """The scenario/target tag MetricRecorder put in the filename, or "?".

    ``vla_metrics_s04-colour-green-circle_20260909_162230.csv`` ->
    ``s04-colour-green-circle``.
    """
    match = _SESSION_CSV.search(basename)
    return match.group(1) if match else "?"


def generate_report(output_dir: str, since: float | None = None) -> str:
    """Summarise the evaluation CSVs in ``output_dir``.

    ``since`` (a POSIX timestamp) keeps only sessions written after it; older
    CSVs are counted in the header but excluded.
    """
    rows = []
    earlier = 0
    # Any rollout CSV, whichever state wrote it; see _SESSION_CSV.
    pattern = os.path.join(output_dir, "*.csv")
    for path in sorted(glob.glob(pattern)):
        if not _SESSION_CSV.search(os.path.basename(path)):
            continue
        if since is not None and os.path.getmtime(path) < since:
            earlier += 1
            continue
        with open(path, newline="") as stream:
            samples = list(csv.DictReader(stream))
        if not samples:
            continue
        scenarios = _finite(row.get("scenario") for row in samples)
        distances = _finite(row.get("dist_target_actual") for row in samples)
        times = _finite(row.get("t") for row in samples)
        rows.append(
            (
                int(scenarios[0]) if scenarios else "?",
                _session_tag(os.path.basename(path)),
                os.path.basename(path),
                max(times) if times else 0.0,
                min(distances) * 1000.0 if distances else float("nan"),
                distances[-1] * 1000.0 if distances else float("nan"),
            )
        )

    report_path = os.path.join(output_dir, "evaluation_report.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as stream:
        stream.write("# VLA evaluation report\n\n")
        stream.write(f"Sessions: {len(rows)}\n\n")
        if earlier:
            stream.write(
                f"{earlier} older session(s) in this folder predate the run and "
                "are not included.\n\n"
            )
        stream.write(
            "| Scenario | Target | Duration (s) | Min tip distance (mm) | "
            "Final tip distance (mm) | CSV |\n"
        )
        stream.write("|---:|---|---:|---:|---:|---|\n")
        for scenario, tag, name, duration, minimum, final in rows:
            stream.write(
                f"| {scenario} | {tag} | {duration:.1f} | {minimum:.1f} | "
                f"{final:.1f} | `{name}` |\n"
            )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(generate_report(args.output_dir))


if __name__ == "__main__":
    main()
