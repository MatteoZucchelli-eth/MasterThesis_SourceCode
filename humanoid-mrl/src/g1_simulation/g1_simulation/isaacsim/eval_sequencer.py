# MIT License
"""Deterministic evaluation harness: generated scenarios, V-key to advance."""
from __future__ import annotations

import json
import logging
import os

from g1_simulation.isaacsim import scene
from g1_simulation.isaacsim.carb_keys import is_key_press
from g1_simulation.isaacsim.eval_scenarios import SCENARIOS
from g1_simulation.isaacsim.pose_resetter import RobotPoseResetter


SELECTION_ENV = "EVAL_SCENARIO_FILTER"
"""Environment variable naming a subset of the table to run."""


def select_scenarios(scenarios: list[dict], spec: str):
    """Restrict the run to some scenarios, and optionally to some targets.

    ``spec`` is a comma-separated list of ``<id>`` or ``<id>:<colour>-<shape>``
    entries; empty (or unset) runs the whole table unchanged. Repeating an id
    with different cells unions them, so
    ``"5:red-triangle,6:green-triangle,6:red-triangle"`` runs one target in
    scenario 5 and two in scenario 6.

    Returns ``(scenarios, task_cells)``, where ``task_cells`` maps a scenario
    id to the cells to command in it, or None to command all of its objects.

    Only narrows which targets are commanded; every scenario object stays shown.
    """
    spec = (spec or "").strip()
    if not spec:
        return list(scenarios), {}

    wanted: dict[int, set] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        head, _, cell = entry.partition(":")
        try:
            scenario_id = int(head.strip().lstrip("sS").lstrip("0") or "0")
        except ValueError:
            raise ValueError(f"{SELECTION_ENV}: cannot read a scenario id from {entry!r}")
        cells = wanted.setdefault(scenario_id, set())
        if cell:
            colour, _, shape = cell.strip().partition("-")
            if not colour or not shape:
                raise ValueError(
                    f"{SELECTION_ENV}: expected <id>:<colour>-<shape>, got {entry!r}"
                )
            cells.add((colour, shape))

    known = {sc["id"] for sc in scenarios}
    missing = sorted(set(wanted) - known)
    if missing:
        raise ValueError(f"{SELECTION_ENV}: no such scenario id(s): {missing}")

    selected = [sc for sc in scenarios if sc["id"] in wanted]
    task_cells = {}
    for sc in selected:
        cells = wanted[sc["id"]]
        if not cells:
            continue
        unknown = sorted(cells - {tuple(c) for c in sc["objects"]})
        if unknown:
            raise ValueError(
                f"{SELECTION_ENV}: scenario {sc['id']} has no cell(s) {unknown}; "
                f"it shows {sorted(tuple(c) for c in sc['objects'])}"
            )
        task_cells[sc["id"]] = cells
    return selected, task_cells


class EvaluationSequencer:
    """Deterministic evaluation harness: V-key steps through the scenarios.

    Each V-press teleports the robot to the scenario's base pose, shows its
    three wall objects (hiding the other six, so the probed attribute is the
    only cue) and announces the scenario from the static :mod:`.eval_scenarios`
    table. The announcement is drained by :class:`.ros_bridge.RosBridge`, which
    latches it on ``sim/eval_scenario``; results are recorded by g1_control.
    """

    def __init__(self, pose_resetter: RobotPoseResetter):
        self._pose_resetter = pose_resetter
        self._scenario_idx = -1
        # Optional subset via EVAL_SCENARIO_FILTER (see select_scenarios).
        self._scenarios, self._task_cells = select_scenarios(
            SCENARIOS, os.environ.get(SELECTION_ENV, "")
        )
        if not self._scenarios:
            raise ValueError(f"{SELECTION_ENV} selected no scenarios")

        # Set by _advance() (keyboard thread), drained by the ROS bridge.
        self._pending_announcement: str | None = None

        if len(self._scenarios) != len(SCENARIOS):
            logging.info(
                f"EvaluationSequencer: {SELECTION_ENV} restricts this run to "
                f"{len(self._scenarios)} of {len(SCENARIOS)} scenarios: "
                f"{[sc['id'] for sc in self._scenarios]}."
            )
        logging.info(
            "EvaluationSequencer ready — press V to start/advance scenarios "
            f"({len(self._scenarios)} total)."
        )

    def on_keyboard_input(self, e) -> None:
        if is_key_press(e, "V"):
            self.advance()

    def advance(self) -> None:
        """Place the next scenario (the programmatic equivalent of V)."""
        self._advance()

    def pop_scenario_announcement(self) -> str | None:
        """Consume the pending scenario announcement (JSON), at most once per V."""
        msg = self._pending_announcement
        self._pending_announcement = None
        return msg

    def _advance(self) -> None:
        self._scenario_idx = (self._scenario_idx + 1) % len(self._scenarios)
        sc = self._scenarios[self._scenario_idx]

        logging.info(
            f"EvaluationSequencer: scenario {sc['id']}/{len(self._scenarios)} — "
            f"{sc['probe']} probe ({sc['constant']}), "
            f"instruction \"{sc['instruction']}\""
        )

        self._pose_resetter.teleport_to(sc["robot_position"], sc["robot_yaw_deg"])
        self._place_objects(sc["objects"])

        # The controller resolves its target frame from the instruction.
        self._pending_announcement = json.dumps(
            {
                "scenario": sc["id"],
                # Scenarios in this run (after filtering).
                "total": len(self._scenarios),
                # Total rollouts; the bringup node stops the evaluation on it.
                "total_tasks": sum(
                    len(self._tasks_for(scenario)) for scenario in self._scenarios
                ),
                "probe": sc["probe"],
                "constant": sc["constant"],
                "target_cell": list(sc["target_cell"]),
                "target_frame": sc["target_frame"],
                "instruction": sc["instruction"],
                "visible_cells": [list(cell) for cell in sc["objects"]],
                "tasks": self._tasks_for(sc),
            }
        )

    def _tasks_for(self, sc: dict) -> list:
        """The cells this run commands in ``sc``, as the bringup node's tasks."""
        commanded = self._task_cells.get(sc["id"], sc["objects"])
        return [
            {
                "target_cell": list(cell),
                "instruction": f"drill in the {cell[0]} {cell[1]}",
            }
            for cell in sc["objects"]
            if cell in commanded
        ]

    def _place_objects(self, objects: dict) -> None:
        """Show and place this scenario's cells; hide every other cell.

        ``objects`` is keyed by ``(colour, shape)`` like :data:`scene.WALL_OBJECTS`;
        unknown cells raise.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        unknown = set(objects) - set(scene.WALL_OBJECTS)
        if unknown:
            raise KeyError(
                f"EvaluationSequencer: scenario names cells the scene does not "
                f"build: {sorted(unknown)}. Regenerate eval_scenarios.py."
            )

        for cell, (prim_path, _, _) in scene.WALL_OBJECTS.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logging.warning(f"EvaluationSequencer: prim {prim_path} not found")
                continue
            imageable = UsdGeom.Imageable(prim)
            if cell not in objects:
                imageable.MakeInvisible()
                continue
            imageable.MakeVisible()
            y, z = objects[cell]
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(scene.TARGET_X, y, z))
                    break
