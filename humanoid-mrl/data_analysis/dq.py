#!/usr/bin/env python3
"""Minimal preflight report for local LeRobot v2 datasets."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EPS = 1e-9


def find_root(path):
    if os.path.isdir(os.path.join(path, "meta")):
        return path
    found = [p for p in glob.glob(os.path.join(path, "*"))
             if os.path.isdir(os.path.join(p, "meta"))]
    if len(found) == 1:
        return found[0]
    raise ValueError(f"expected one LeRobot dataset under {path}, found {len(found)}")


def list_column(table, name):
    column = table.column(name).combine_chunks()
    flat = column.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(len(column), -1).astype(float)


def eef_slot(modality):
    for key, bounds in modality.get("state", {}).items():
        if key.startswith("right") and key.endswith("eef_9d"):
            return slice(bounds["start"], bounds["end"]), key
    return None, None


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def smooth(points, width=5):
    if len(points) < width:
        return points.copy()
    kernel, pad = np.ones(width) / width, width // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid")[:len(points)]
                     for i in range(points.shape[1])], axis=1)


def rotation_matrices(rotation_6d):
    first, second = rotation_6d[:, :3], rotation_6d[:, 3:]
    first = first / (np.linalg.norm(first, axis=1, keepdims=True) + EPS)
    second = second - (first * second).sum(1, keepdims=True) * first
    second = second / (np.linalg.norm(second, axis=1, keepdims=True) + EPS)
    return np.stack((first, second, np.cross(first, second)), axis=2)


class Dataset:
    def __init__(self, path):
        self.root = find_root(path)
        self.name = os.path.basename(self.root.rstrip("/"))
        if self.name == "g1_dataset":
            self.name = os.path.basename(os.path.dirname(self.root.rstrip("/")))
        meta = os.path.join(self.root, "meta")
        with open(os.path.join(meta, "info.json")) as f:
            self.info = json.load(f)
        with open(os.path.join(meta, "modality.json")) as f:
            self.modality = json.load(f)
        self.fps = float(self.info["fps"])

        self.task_names = {}
        path = os.path.join(meta, "tasks.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    self.task_names[int(row["task_index"])] = row["task"]
        episode_meta = {}
        path = os.path.join(meta, "episodes.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    episode_meta[int(row["episode_index"])] = row

        files = sorted(glob.glob(os.path.join(
            self.root, "data", "chunk-*", "episode_*.parquet")))
        if not files:
            raise ValueError("no episode parquet files")
        self.episodes, first_schema, self.schema_mismatches = [], None, 0
        for path in files:
            index = int(Path(path).stem.split("_")[1])
            table = pq.read_table(path)
            required = ("observation.state", "action", "timestamp")
            missing = set(required) - set(table.column_names)
            if missing:
                raise ValueError(f"episode {index} lacks {sorted(missing)}")
            schema = tuple((key, str(table.schema.field(key).type)) for key in required)
            first_schema = first_schema or schema
            self.schema_mismatches += schema != first_schema
            meta_row = episode_meta.get(index, {})
            episode = {
                "index": index,
                "state": list_column(table, "observation.state"),
                "action": list_column(table, "action"),
                "timestamp": table.column("timestamp").to_numpy(
                    zero_copy_only=False).astype(float),
                "tasks": tuple(int(x) for x in meta_row.get("tasks", ())),
                "declared_length": meta_row.get("length"),
            }
            if "next.tip_target_dist" in table.column_names:
                episode["tip_dist"] = table.column("next.tip_target_dist").to_numpy(
                    zero_copy_only=False).astype(float)
            if "next.tip_target_9d" in table.column_names:
                episode["tip_target"] = list_column(table, "next.tip_target_9d")
            self.episodes.append(episode)

        self.state_slot, self.state_slot_name = eef_slot(self.modality)
        self.scale_mm = 1000.0
        if self.state_slot:
            sample = np.concatenate([e["state"][:10] for e in self.episodes])
            pos = sample[:, self.state_slot.start:self.state_slot.start + 3]
            self.scale_mm = 1000.0 if np.percentile(np.abs(pos), 99) < 5 else 1.0

    def positions(self, episode):
        if self.state_slot is None:
            return None
        return (episode["state"][:, self.state_slot.start:self.state_slot.start + 3]
                * self.scale_mm)


def label(ds, tasks):
    return ", ".join(ds.task_names.get(i, f"unknown task {i}") for i in tasks) or "untasked"


def goal_points(ds):
    goals = []
    exact = all("tip_dist" in e and "tip_target" in e for e in ds.episodes)
    for episode in ds.episodes:
        pos = ds.positions(episode)
        if pos is None:
            return np.empty((0, 3)), "unavailable"
        if exact:
            target = episode["tip_target"][:, :3] * ds.scale_mm
            valid = np.isfinite(target).all(1) & np.isfinite(episode["tip_dist"])
            if valid.any():
                rotation_6d = episode["state"][:, ds.state_slot.start + 3:ds.state_slot.start + 9]
                world_target = pos + np.einsum("nij,nj->ni", rotation_matrices(rotation_6d), target)
                goals.append(np.median(world_target[valid], axis=0))
                continue
        filtered = smooth(pos)
        speed = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
        start = len(speed) // 2
        i = start + int(np.argmin(speed[start:])) if len(speed) else len(pos) - 1
        goals.append(pos[i])
    return np.asarray(goals), "target labels" if exact else "rest-point proxy"


def pca(points):
    centered = points - points.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    extent = np.percentile(projected, 95, axis=0) - np.percentile(projected, 5, axis=0)
    return projected[:, :2], extent


def coverage(ds, goals, target_mm):
    groups = {}
    for i, episode in enumerate(ds.episodes):
        groups.setdefault(episode["tasks"], []).append(i)
    per_task, densities, cells, nearest = {}, [], [], []
    for tasks, indices in groups.items():
        points = goals[indices]
        entry = {
            "task_label": label(ds, tasks),
            "episodes": len(indices),
            "hours": sum(len(ds.episodes[i]["state"]) for i in indices) / ds.fps / 3600,
        }
        if len(points) >= 3:
            projected, extent = pca(points)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            nn = distances.min(1)
            nearest.extend(nn)
            entry.update({
                "pca_extent_1_mm": float(extent[0]),
                "pca_extent_2_mm": float(extent[1]),
                "goal_nn_median_mm": float(np.median(nn)),
                "goals_2d_mm": np.round(projected, 1).tolist(),
            })
            if target_mm:
                count = max(1.0, extent[0] * extent[1] / target_mm**2)
                density = len(indices) / count
                entry.update(target_cells=float(count), episodes_per_target_cell=float(density))
                densities.append(density)
                cells.append(count)
        per_task[str(list(tasks))] = entry
    result = {"per_task": per_task, "n_tasks_referenced": len(groups)}
    if nearest:
        result["goal_nn_median_mm"] = float(np.median(nearest))
    if densities:
        result.update(episodes_per_target_cell=float(min(densities)),
                      target_cells_total=float(sum(cells)))
    return result


def motion_consistency(ds):
    repeats = []
    for episode in ds.episodes:
        if len(episode["action"]) < 2:
            continue
        delta = np.diff(episode["action"], axis=0)
        repeats.append(float((np.max(np.abs(delta), axis=1) == 0).mean()))
    return {"action_repeat_frac": float(np.median(repeats)) if repeats else float("nan")}


def target_accuracy(ds, target_mm):
    labelled = [e for e in ds.episodes if "tip_dist" in e]
    result = {"target_labelled_episode_frac": len(labelled) / len(ds.episodes)}
    if not labelled:
        return result
    minima, finals, hits, bad, count = [], [], [], 0, 0
    for episode in labelled:
        distance = episode["tip_dist"] * ds.scale_mm
        bad += int((~np.isfinite(distance)).sum())
        count += len(distance)
        valid = distance[np.isfinite(distance)]
        if len(valid):
            minima.append(float(valid.min()))
            finals.append(float(valid[-1]))
            if target_mm:
                hits.append(valid.min() <= target_mm / 2)
    result.update(target_label_nonfinite_frac=bad / max(count, 1),
                  demo_min_target_dist_p95_mm=pct(minima, 95),
                  demo_final_target_dist_p95_mm=pct(finals, 95))
    if hits:
        result["demo_target_hit_frac"] = float(np.mean(hits))
    return result


def integrity(ds):
    issues, warnings = [], []
    frames = sum(len(e["state"]) for e in ds.episodes)
    if int(ds.info.get("total_frames", frames)) != frames:
        issues.append(f"info.json frames differ from the {frames} rows on disk")
    if int(ds.info.get("total_episodes", len(ds.episodes))) != len(ds.episodes):
        issues.append(f"info.json episode count differs from the {len(ds.episodes)} files")
    length_bad = sum(e["declared_length"] is not None
                     and e["declared_length"] != len(e["state"]) for e in ds.episodes)
    if length_bad:
        issues.append(f"{length_bad} episode length mismatches")
    if ds.schema_mismatches:
        issues.append(f"{ds.schema_mismatches} core schema mismatches")

    referenced = {task for e in ds.episodes for task in e["tasks"]}
    declared = set(ds.task_names)
    if referenced - declared:
        issues.append(f"unknown task ids referenced: {sorted(referenced - declared)}")
    if int(ds.info.get("total_tasks", len(declared))) != len(declared):
        warnings.append("info.json task count differs from tasks.jsonl")

    state_bad = sum(int((~np.isfinite(e["state"])).sum()) for e in ds.episodes)
    action_bad = sum(int((~np.isfinite(e["action"])).sum()) for e in ds.episodes)
    if state_bad:
        issues.append(f"{state_bad} non-finite state values")
    if action_bad:
        issues.append(f"{action_bad} non-finite action values")
    positive_dt, nonmonotonic = [], 0
    for episode in ds.episodes:
        dt = np.diff(episode["timestamp"])
        nonmonotonic += int((dt <= 0).sum())
        positive_dt.extend(dt[dt > 0])
    if nonmonotonic:
        issues.append(f"{nonmonotonic} non-monotonic timestamp steps")
    positive_dt = np.asarray(positive_dt)

    cameras = {key: "×".join(str(x) for x in feature.get("shape", [])[1:])
               for key, feature in ds.info.get("features", {}).items()
               if feature.get("dtype") == "video"}
    missing_videos = 0
    for episode in ds.episodes:
        for key in cameras:
            pattern = os.path.join(ds.root, "videos", "chunk-*", key,
                                   f"episode_{episode['index']:06d}.mp4")
            missing_videos += not glob.glob(pattern)
    if missing_videos:
        issues.append(f"{missing_videos} expected video files missing")
    return {
        "integrity_issue_count": len(issues), "integrity_issues": issues,
        "integrity_warning_count": len(warnings), "integrity_warnings": warnings,
        "state_nonfinite": state_bad, "action_nonfinite": action_bad,
        "timestamp_nonmonotonic_steps": nonmonotonic,
        "timestamp_dt_cv": (float(positive_dt.std() / positive_dt.mean())
                            if len(positive_dt) else float("nan")),
        "missing_video_files": missing_videos, "cameras": cameras,
    }


def analyze(ds, target_mm):
    lengths = np.asarray([len(e["state"]) for e in ds.episodes])
    frames = int(lengths.sum())
    result = {
        "name": ds.name, "root": ds.root, "target_mm": target_mm,
        "n_episodes": len(ds.episodes), "n_frames": frames,
        "total_hours": frames / ds.fps / 3600, "fps": ds.fps,
        "episode_duration_p05_s": pct(lengths / ds.fps, 5),
        "episode_duration_median_s": pct(lengths / ds.fps, 50),
        "episode_duration_p95_s": pct(lengths / ds.fps, 95),
        "n_tasks_declared": len(ds.task_names),
        "state_dim": int(ds.episodes[0]["state"].shape[1]),
        "action_dim": int(ds.episodes[0]["action"].shape[1]),
        "right_eef_slot": ds.state_slot_name,
    }
    result.update(integrity(ds))
    result.update(target_accuracy(ds, target_mm))
    result.update(motion_consistency(ds))
    goals, source = goal_points(ds)
    result["goal_source"] = source
    if len(goals):
        result.update(coverage(ds, goals, target_mm))
    return result


ROWS = [
    ("n_episodes", "episodes", "{:.0f}"), ("n_frames", "frames", "{:.0f}"),
    ("total_hours", "demonstration hours", "{:.2f}"), ("fps", "fps", "{:.1f}"),
    ("episode_duration_median_s", "median episode duration (s)", "{:.1f}"),
    ("n_tasks_declared", "declared tasks", "{:.0f}"),
    ("n_tasks_referenced", "referenced task groups", "{:.0f}"),
    ("integrity_issue_count", "integrity issues", "{:.0f}"),
    ("integrity_warning_count", "integrity warnings", "{:.0f}"),
    ("state_nonfinite", "non-finite state values", "{:.0f}"),
    ("action_nonfinite", "non-finite action values", "{:.0f}"),
    ("timestamp_nonmonotonic_steps", "non-monotonic timestamp steps", "{:.0f}"),
    ("missing_video_files", "missing video files", "{:.0f}"),
    ("demo_target_hit_frac", "demo target-hit fraction", "{:.2f}"),
    ("demo_min_target_dist_p95_mm", "demo minimum target error p95 (mm)", "{:.1f}"),
    ("episodes_per_target_cell", "episodes / PCA target cell (heuristic)", "{:.2f}"),
    ("goal_nn_median_mm", "goal nearest-neighbour distance (mm)", "{:.1f}"),
    ("action_repeat_frac", "repeated-action fraction", "{:.2f}"),
]


def print_report(results):
    width = max(len(label) for _, label, _ in ROWS)
    sizes = [max(12, len(r["name"])) for r in results]
    print("# Dataset preflight\n")
    print("| " + "metric".ljust(width) + " | " + " | ".join(
        r["name"].ljust(w) for r, w in zip(results, sizes)) + " |")
    print("|-" + "-" * width + "-|" + "|".join("-" * (w + 2) for w in sizes) + "|")
    for key, label_, spec in ROWS:
        if not any(key in r for r in results):
            continue
        values = []
        for result, width_ in zip(results, sizes):
            value = result.get(key)
            shown = "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else spec.format(value)
            values.append(shown.ljust(width_))
        print("| " + label_.ljust(width) + " | " + " | ".join(values) + " |")
    print("\nPCA density and the repeated-action fraction are descriptive heuristics, not pass/fail tests.")
    for result in results:
        if result["integrity_issues"]:
            print(f"\n## {result['name']} integrity issues\n")
            for issue in result["integrity_issues"]:
                print(f"- {issue}")
        if result["integrity_warnings"]:
            print(f"\n## {result['name']} integrity warnings\n")
            for warning in result["integrity_warnings"]:
                print(f"- {warning}")


def discover(root):
    found = []
    for group in ("big_targets", "small_targets"):
        for path in sorted(glob.glob(os.path.join(root, group, "*"))):
            if os.path.isdir(path):
                try:
                    find_root(path)
                    found.append((path, group))
                except ValueError:
                    pass
    return found


def main():
    parser = argparse.ArgumentParser(description="Minimal LeRobot dataset preflight")
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--scan", metavar="DIR")
    parser.add_argument("--big-mm", type=float, default=100)
    parser.add_argument("--small-mm", type=float, default=10)
    parser.add_argument("--target-mm", type=float)
    parser.add_argument("--target", action="append", default=[], metavar="NAME=MM")
    parser.add_argument("--out", help="write full JSON")
    parser.add_argument("--dashboard", help="write self-contained HTML")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.target)
    paths = [(path, None) for path in args.datasets]
    if args.scan:
        paths += discover(args.scan)
    if not paths:
        parser.error("pass dataset paths and/or --scan DIR")
    group_sizes = {"big_targets": args.big_mm, "small_targets": args.small_mm}
    results = []
    for path, group in paths:
        try:
            ds = Dataset(path)
            raw_size = overrides.get(ds.name, group_sizes.get(group, args.target_mm))
            result = analyze(ds, float(raw_size) if raw_size is not None else None)
            result["group"] = group
            results.append(result)
            print(f"[dq] {ds.name}: {result['n_episodes']} episodes", file=sys.stderr)
        except Exception as error:
            print(f"[dq] skipping {path}: {error}", file=sys.stderr)
    print_report(results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    if args.dashboard:
        from dashboard import render_dashboard
        template = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
        render_dashboard(results, template, args.dashboard)


if __name__ == "__main__":
    main()
