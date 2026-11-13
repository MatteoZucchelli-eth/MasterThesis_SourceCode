# MIT License
"""Drill-tip tracking metric recorder for the controller manager.

Samples at ~10 Hz during a VLA session:

  * commanded vs. measured drill-tip position and tracking error;
  * wall-target distance to the measured and commanded tip, split into lateral
    and along-normal components;
  * predicted vs. measured target pose in the drill-tip frame.

Writes a timestamped CSV to the output folder and spawns
``g1_control.plot_vla_metrics`` to render PNGs.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin
from g1_definitions import definitions
from std_msgs.msg import String

from g1_control.states.base import ControlContext
from g1_control.utils.perception_stats import (
    format_stats,
    perception_error_stats,
    pick_group_column,
    stats_rows,
)
from g1_control.utils.ros_helpers import LATCHED_QOS, lookup_se3

_CONSOLE_LOG_PERIOD_S = 10.0
"""Seconds between console tip→target lines (0.1 Hz)."""

_NAN = float("nan")

# start() sentinel: keep the constructor source (None disables it).
_KEEP = object()


def _set_xyz(row: dict, prefix: str, v) -> None:
    """Fill ``<prefix>_x/y/z`` from the first three entries of ``v``."""
    row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"] = (
        float(x) for x in v[:3]
    )


class MetricRecorder:
    """Samples drill-tip tracking + target-relative metrics at ~10 Hz to a CSV."""

    _CSV_HEADER = [
        "t",
        "waypoint_idx",
        "scenario",  # eval scenario id (sim/eval_scenario), NaN outside eval
        "cmd_x",
        "cmd_y",
        "cmd_z",  # commanded tip target (pelvis frame)
        "act_x",
        "act_y",
        "act_z",  # measured tip pose (pelvis frame)
        "track_err_x",
        "track_err_y",
        "track_err_z",
        "track_err_l2",  # cmd - act
        "target_x",
        "target_y",
        "target_z",  # wall target (pelvis frame)
        "dist_target_actual",  # |target - measured tip|
        "dist_target_cmd",  # |target - commanded tip|
        "pred_rel_x",
        "pred_rel_y",
        "pred_rel_z",  # VLA-predicted target, tip frame
        "act_rel_x",
        "act_rel_y",
        "act_rel_z",  # measured target, tip frame
        "rel_err_x",
        "rel_err_y",
        "rel_err_z",  # predicted - actual (tip frame)
        "rel_err_rx",
        "rel_err_ry",
        "rel_err_rz",
        "rel_err_xaxis_deg",  # angle between predicted and measured target x axis
        # Same groups for the driving (second) source, against the same truth.
        "drv_pred_rel_x",
        "drv_pred_rel_y",
        "drv_pred_rel_z",  # driving model's predicted target, tip frame
        "drv_rel_err_x",
        "drv_rel_err_y",
        "drv_rel_err_z",  # its error against the same measured target
        "drv_rel_err_xaxis_deg",
        "fused_x",
        "fused_y",
        "fused_z",  # fused target estimate (pelvis frame)
        # Placement error about the wall normal, against the true target TF.
        "miss_lateral",  # |component of target - tip across the wall normal|
        "miss_normal",  # signed component along it; + = tip short of the wall
        "perp_err_deg",  # angle between the drill axis and the wall normal
        # Same split for the commanded tip (separates model vs. controller error).
        "cmd_miss_lateral",  # |component of target - commanded tip across the normal|
        "cmd_miss_normal",  # signed component along it; + = command short of the wall
        "phase",  # VLA_PLANNER substate (VlaPlannerPhase value); NaN elsewhere
        # Sequence number of the pred_rel_* prediction (rows repeat between
        # predictions).
        "pred_seq",
        # Detector confidence from the newest raw (ungated) detection; NaN = none yet.
        "visibility",  # primary source's confidence, 0..1
        "visible",  # its own boolean gate; 1.0 on a model with no `visible` head
        "drv_visibility",  # the driving model's confidence, same instant
    ]

    def __init__(
        self,
        node,
        ctx: ControlContext,
        vla_client,
        tip_estimator=None,
        driver_client=None,
    ) -> None:
        self._ctx = ctx
        self._logger = node.get_logger()
        self._vla_client = vla_client
        # Optional second source; its columns stay NaN when None.
        self._driver_client = driver_client
        self._tip_estimator = tip_estimator
        # Per-session telemetry source; start() may override it.
        self._session_estimator = tip_estimator

        self._output_dir = (
            node.get_parameter("progress_test_output_dir").value
            if node.has_parameter("progress_test_output_dir")
            else ""
        ) or "vla_metrics"
        self._csv_enabled = (
            node.get_parameter("vla_metrics_csv_enabled").value
            if node.has_parameter("vla_metrics_csv_enabled")
            else True
        )

        self._records: list[list[float]] = []
        self._start_time: float | None = None
        self._counter = 0
        self._finalized = False
        self._prefix = "vla_metrics"

        # Latched eval scenario from the sim ({"scenario": -1} clears it); frozen
        # into _session_scenario at start().
        self._live_scenario: dict | None = None
        self._session_scenario: dict | None = None
        node.create_subscription(
            String,
            definitions.TOPIC_SIM_EVAL_SCENARIO,
            self._scenario_cb,
            LATCHED_QOS,
        )

    def _scenario_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self._logger.warning(f"vla-metrics: bad eval scenario payload: {msg.data}")
            return
        self._live_scenario = payload if payload.get("scenario", -1) >= 0 else None

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def set_estimate_source(self, vla_client) -> None:
        """Point the ``pred_rel_*`` / ``rel_err_*`` columns at another client."""
        self._vla_client = vla_client

    def start(
        self,
        prefix: str = "vla_metrics",
        tip_estimator=_KEEP,
    ) -> None:
        """Reset the recorder at the start of a VLA session.

        ``prefix`` names the output file (``vla_metrics``, ``progress_test``,
        ``vla_planner``). ``tip_estimator`` overrides the fused-estimate source
        for this session (``None`` blanks it); leave unset to keep the default.
        """
        self._prefix = prefix
        self._session_estimator = (
            self._tip_estimator if tip_estimator is _KEEP else tip_estimator
        )
        self._records = []
        self._start_time = None
        self._counter = 0
        self._finalized = False
        self._session_scenario = self._live_scenario
        if self._session_scenario is not None:
            self._logger.info(
                f"VLA metrics: session tagged with eval scenario "
                f"{self._session_scenario}"
            )

    def record(self, now: float, waypoint_idx: int, phase: float = _NAN) -> None:
        """Append one tracking sample at ~10 Hz.

        ``waypoint_idx`` is the scripted-sweep index, or -1 during a real VLA
        run. ``phase`` is the VLA_PLANNER substate value (NaN elsewhere).
        ``now`` comes from the task clock (sim seconds under ``use_sim_time``).
        """
        if self._start_time is None:
            self._start_time = now
        self._counter += 1
        stride = max(1, round(0.1 / self._ctx.control_dt))
        if self._counter % stride != 0:
            return

        row = self._sample(now - self._start_time, waypoint_idx, phase)
        self._records.append([row[c] for c in self._CSV_HEADER])
        # Throttled console headline; full detail goes to the CSV.
        dist = row["dist_target_actual"]
        line = (
            f"[vla] tip→target {dist * 1000.0:6.1f} mm"
            if np.isfinite(dist)
            else "[vla] tip→target   n/a  "
        )
        lat, nrm = row["miss_lateral"], row["perp_err_deg"]
        if np.isfinite(lat):
            line += (
                f" | lateral {lat * 1000.0:5.2f} mm"
                f" | normal {row['miss_normal'] * 1000.0:+6.2f} mm"
                f" | perp {nrm:4.1f}°"
            )
        self._logger.info(line, throttle_duration_sec=_CONSOLE_LOG_PERIOD_S)

    def _score_prediction(self, row, client, act_rel, pred_key, err_key) -> None:
        """Write one source's prediction and its error against ``act_rel``.

        Rotation components are written only for the primary source.
        """
        if client is None:
            return
        pred_9d = client.get_tip_target()
        if pred_9d is None:
            return
        _set_xyz(row, pred_key, pred_9d)
        if act_rel is None:
            return

        ctx = self._ctx
        pred = ctx.robot_model.action_9d_to_se3(np.asarray(pred_9d, dtype=float))
        _set_xyz(row, err_key, pred.translation - act_rel.translation)
        if f"{err_key}_rx" in row:
            rot_err = pin.log3(pred.rotation.T @ act_rel.rotation)
            row[f"{err_key}_rx"], row[f"{err_key}_ry"], row[f"{err_key}_rz"] = map(
                float, rot_err
            )
        # Angle between predicted and measured target x axis (roll is irrelevant).
        cos_x = float(np.dot(pred.rotation[:, 0], act_rel.rotation[:, 0]))
        row[f"{err_key}_xaxis_deg"] = float(
            np.degrees(np.arccos(np.clip(cos_x, -1.0, 1.0)))
        )

    @staticmethod
    def _write_stats(path: str, stats) -> None:
        """Write one perception stats table to CSV."""
        header, rows = stats_rows(stats)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def perception_stats(self, source_prefix: str = ""):
        """Perception-error mean/std for the samples recorded so far.

        Grouped by waypoint (sweep), planner phase (VLA_PLANNER) or ``"all"``.
        ``source_prefix`` (e.g. ``"drv_"``) scores that source by renaming its
        columns over the primary's.
        """
        columns = {
            name: np.asarray([r[i] for r in self._records], dtype=float)
            for i, name in enumerate(self._CSV_HEADER)
        }
        if source_prefix:
            columns = {
                **columns,
                **{
                    name.removeprefix(source_prefix): values
                    for name, values in columns.items()
                    if name.startswith(source_prefix)
                },
            }
        return perception_error_stats(columns, group_by=pick_group_column(columns))

    def _filename_tag(self) -> str:
        """Filename tag naming the scored target.

        The scored ``ctx.drill_target_frame`` decides; the latched scenario only
        adds its id/probe when consistent: ``_s01-colour-red-square``,
        ``_red-square``, or ``""`` when no target resolved. Scenario keys are
        read defensively.
        """
        frame = self._ctx.drill_target_frame
        cell = frame.removesuffix("_target").replace("_", "-") if frame else ""
        sc = self._session_scenario
        if sc is not None and sc.get("scenario") is not None:
            sc_cell = "-".join(sc.get("target_cell", ())) or sc.get(
                "target_frame", ""
            ).removesuffix("_target").replace("_", "-")
            visible_cells = {
                "-".join(visible) for visible in sc.get("visible_cells", ())
            }
            if cell and cell in visible_cells:
                return f"_s{int(sc['scenario']):02d}-{sc.get('probe', '?')}-{cell}"
            if sc_cell and (not cell or sc_cell == cell):
                return f"_s{int(sc['scenario']):02d}-{sc.get('probe', '?')}-{sc_cell}"
            if sc_cell:
                self._logger.warning(
                    f"VLA metrics: scored target '{cell}' is not scenario "
                    f"{sc.get('scenario')}'s '{sc_cell}' — the instruction "
                    "outranks the latched scenario, naming the file after the "
                    "target the rows were actually measured against."
                )
        return f"_{cell}" if cell else ""

    def finalize(self) -> None:
        """Log the perception-error summary, write the CSVs, render the PNGs.

        Idempotent; no-op when there is nothing to write.
        """
        if self._finalized or not self._records:
            return
        self._finalized = True

        # Console summary first, so it survives a disabled/failed CSV write.
        stats = self.perception_stats()
        self._logger.info(f"VLA metrics: {format_stats(stats)}")
        driver_stats = (
            self.perception_stats("drv_") if self._driver_client is not None else None
        )
        if driver_stats is not None:
            self._logger.info(
                f"VLA metrics, driving model: {format_stats(driver_stats)}"
            )

        if not self._csv_enabled:
            return
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self._output_dir, f"{self._prefix}{self._filename_tag()}_{stamp}.csv"
            )
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._CSV_HEADER)
                writer.writerows(self._records)
            self._logger.info(
                f"VLA metrics: {len(self._records)} samples written to {path}"
            )
            base = os.path.splitext(path)[0]
            self._write_stats(f"{base}_perception_stats.csv", stats)
            if driver_stats is not None:
                self._write_stats(f"{base}_driver_perception_stats.csv", driver_stats)
        except OSError as e:
            self._logger.error(f"VLA metrics: failed to write CSV: {e}")
            return

        # Detached subprocess keeps matplotlib out of the control process.
        try:
            subprocess.Popen(
                [sys.executable, "-m", "g1_control.plot_vla_metrics", path],
                start_new_session=True,
            )
        except OSError as e:
            self._logger.error(f"VLA metrics: failed to launch plotter: {e}")

    # ── Sampling ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_about_normal(delta, normal) -> tuple[float, float]:
        """``(lateral, along)`` of ``delta`` about the unit ``normal`` (same frame)."""
        along = float(delta @ normal)
        return float(np.linalg.norm(delta - along * normal)), along

    @staticmethod
    def _set_placement_error(row: dict, act_rel) -> None:
        """Split the tip->target offset about the wall normal; measure drill
        perpendicularity.

        ``act_rel`` is the target pose in the drill-tip frame: its translation
        is tip->target, its x axis the wall normal; the drill axis is tip x.
        """
        delta = np.asarray(act_rel.translation, dtype=float)
        normal = np.asarray(act_rel.rotation[:, 0], dtype=float)
        row["miss_lateral"], row["miss_normal"] = MetricRecorder._split_about_normal(
            delta, normal
        )
        # normal[0] = cos(drill axis, wall normal); clipped for arccos.
        row["perp_err_deg"] = float(
            np.degrees(np.arccos(np.clip(normal[0], -1.0, 1.0)))
        )

    def _sample(self, t: float, waypoint_idx: int, phase: float) -> dict[str, float]:
        """One CSV row keyed by header name (NaN where a source is unavailable)."""
        ctx = self._ctx
        row = dict.fromkeys(self._CSV_HEADER, _NAN)
        row["t"], row["waypoint_idx"] = t, waypoint_idx
        row["phase"] = float(phase)
        if self._session_scenario is not None:
            row["scenario"] = float(self._session_scenario["scenario"])

        # Commanded vs measured tip (pelvis frame); measured stands in for no command.
        act = ctx.current_right_eef_9d
        cmd = ctx.targets.right_9d if ctx.targets.right_9d is not None else act
        act = None if act is None else np.asarray(act[:3], dtype=float)
        cmd = None if cmd is None else np.asarray(cmd[:3], dtype=float)
        if act is not None:
            _set_xyz(row, "act", act)
        if cmd is not None:
            _set_xyz(row, "cmd", cmd)
        if act is not None and cmd is not None:
            _set_xyz(row, "track_err", cmd - act)
            row["track_err_l2"] = float(np.linalg.norm(cmd - act))

        # Wall target pose (pelvis frame); its x axis is the wall normal.
        target_se3 = lookup_se3(ctx.tf_buffer, ctx.drill_target_frame, "pelvis")
        if target_se3 is not None:
            target = np.asarray(target_se3.translation, dtype=float)
            _set_xyz(row, "target", target)
            if act is not None:
                row["dist_target_actual"] = float(np.linalg.norm(act - target))
            if cmd is not None:
                row["dist_target_cmd"] = float(np.linalg.norm(cmd - target))
                # Same convention as miss_*: + normal = short of the wall.
                (
                    row["cmd_miss_lateral"],
                    row["cmd_miss_normal"],
                ) = self._split_about_normal(
                    target - cmd, np.asarray(target_se3.rotation[:, 0], dtype=float)
                )

        # Predicted vs measured target pose (tip frame), both sources, one lookup.
        act_rel = lookup_se3(ctx.tf_buffer, ctx.drill_target_frame, "drill_tip_link")
        if act_rel is not None:
            _set_xyz(row, "act_rel", act_rel.translation)
            self._set_placement_error(row, act_rel)
        self._score_prediction(row, self._vla_client, act_rel, "pred_rel", "rel_err")
        self._score_prediction(
            row, self._driver_client, act_rel, "drv_pred_rel", "drv_rel_err"
        )

        # Identifies the pred_rel_* reading (not atomic with it).
        anchored = (
            None
            if self._vla_client is None
            else self._vla_client.get_tip_target_anchored()
        )
        if anchored is not None:
            row["pred_seq"] = float(anchored[3])

        # Raw (ungated) detection confidence; NaN before the first detection.
        for client, key in (
            (self._vla_client, "visibility"),
            (self._driver_client, "drv_visibility"),
        ):
            detection = None if client is None else client.get_last_detection()
            if detection is None:
                continue
            row[key] = float(detection.visibility)
            if client is self._vla_client:
                row["visible"] = float(detection.visible)

        # Fused-estimate telemetry from this session's source (the VLA_PLANNER
        # accumulator in planner sessions); NaN when no source is attached.
        if self._session_estimator is not None:
            fused = self._session_estimator.position()
            if fused is not None:
                _set_xyz(row, "fused", fused)

        return row
