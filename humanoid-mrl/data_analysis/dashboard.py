#!/usr/bin/env python3
"""Render dashboard.html from dq.py's metrics.json. Stdlib only.

Usage: dashboard.py metrics.json dashboard_template.html dashboard.html
"""
import json
import sys

KEYS = [
    "n_episodes",
    "n_frames",
    "total_hours",
    "fps",
    "episode_duration_median_s",
    "n_tasks_declared",
    "n_tasks_referenced",
    "integrity_issue_count",
    "integrity_warning_count",
    "state_nonfinite",
    "action_nonfinite",
    "timestamp_nonmonotonic_steps",
    "timestamp_dt_cv",
    "missing_video_files",
    "target_mm",
    "episodes_per_target_cell",
    "target_cells_total",
    "goal_nn_median_mm",
    "target_labelled_episode_frac",
    "target_label_nonfinite_frac",
    "demo_target_hit_frac",
    "demo_min_target_dist_p95_mm",
    "demo_final_target_dist_p95_mm",
    "action_repeat_frac",
    "state_dim",
    "action_dim",
]


def short_label(r):
    label = r["name"]
    if label.startswith("g1_dataset_"):
        label = label[len("g1_dataset_") :]
    return label.replace("_", " ")


def render_dashboard(results, template_path, out_path):
    results.sort(key=lambda x: (x.get("group") or "", x["name"]))

    data = []
    for x in results:
        tasks = []
        for e in x.get("per_task", {}).values():
            if "goals_2d_mm" in e:
                tasks.append(
                    {
                        "label": e.get("task_label", "task"),
                        "episodes": e["episodes"],
                        "hours": e.get("hours"),
                        "density": e.get("episodes_per_target_cell"),
                        "goals": e["goals_2d_mm"],
                    }
                )
        data.append(
            {
                "name": x["name"],
                "label": short_label(x),
                "group": x.get("group"),
                "duplicate_of": x.get("duplicate_of"),
                "metrics": {k: x.get(k) for k in KEYS},
                "tasks": tasks,
                "issues": x.get("integrity_issues", []),
                "warnings": x.get("integrity_warnings", []),
                "goal_source": x.get("goal_source"),
            }
        )

    html = open(template_path).read()
    marker = "/*__DATA__*/"
    if marker not in html:
        sys.exit(f"template {template_path} has no {marker} marker")
    open(out_path, "w").write(
        html.replace(marker, json.dumps(data, separators=(",", ":")))
    )
    print(
        f"[dashboard] {out_path}: {len(data)} datasets, "
        f"{sum(len(d['tasks']) for d in data)} coverage panels",
        file=sys.stderr,
    )


def main():
    metrics_path, template_path, out_path = sys.argv[1:4]
    with open(metrics_path) as f:
        results = json.load(f)
    render_dashboard(results, template_path, out_path)


if __name__ == "__main__":
    main()
