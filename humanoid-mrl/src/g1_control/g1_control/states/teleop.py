# MIT License
"""TELEOP state: map XR palm tracking onto arm IK targets and a base twist."""
from __future__ import annotations

import numpy as np
from g1_definitions import definitions

from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils import mat4_to_se3


class TeleopHandler(StateHandler):
    def __init__(self, ctx: ControlContext, teleoperation_manager) -> None:
        self._ctx = ctx
        self._teleop = teleoperation_manager

    # ── State lifecycle ───────────────────────────────────────────────────────

    def on_enter(self, previous_state: int) -> None:
        ctx = self._ctx
        coming_from_vla = previous_state == definitions.FSMState.VLA.value
        self._teleop.reset_arm_tracking(use_default_pose=not coming_from_vla)
        if coming_from_vla:
            # on_activate() already seeded the interpolator at the current joints;
            # clear the target so the arms do not snap to the default pose.
            ctx.robot_state.upper_body_target = None
        else:
            ctx.robot_state.upper_body_target = (
                ctx.robot_model.get_initial_upper_body_pose()
            )

    def on_exit(self) -> None:
        ctx = self._ctx
        ctx.targets.clear()
        # The teleop joint target is stale once the teleop loop stops refreshing it.
        ctx.robot_state.upper_body_target = None

    # ── Per-tick handler ──────────────────────────────────────────────────────

    def update(self) -> None:
        ctx = self._ctx
        twist_cmd, palm_tracking = self._teleop.get_teleop_commands()
        ctx.robot_state.twist_command = twist_cmd

        if self._teleop.needs_robot_poses:
            target_left_palm, target_right_palm = self._get_target_palm_poses()

            if target_left_palm is not None and target_right_palm is not None:
                self._teleop.set_robot_reference_poses(
                    target_left_palm.homogeneous,
                    target_right_palm.homogeneous,
                )
                # Seed targets from current FK so the recorded action does not
                # jump when tracking activates. Right is the tip, not the palm.
                ctx.targets.left_9d = ctx.robot_model.se3_to_9d_pose(target_left_palm)
                ctx.targets.right_9d = ctx.current_right_eef_9d

        if palm_tracking is not None:
            ctx.robot_state.upper_body_target = self._compute_arm_target(palm_tracking)
        elif self._teleop.use_default_pose:
            ctx.robot_state.upper_body_target = (
                ctx.robot_model.get_initial_upper_body_pose()
            )
            # Clear targets so the recorder falls back to FK (e.g. after Y reset).
            ctx.targets.clear()

    # ── IK / upper-body target helpers ────────────────────────────────────────

    def _compute_arm_target(self, palm_tracking: dict) -> np.ndarray:
        """Compute upper-body joint targets from XR palm-tracking matrices.

        XR drives the palms and IK solves for them; the recorded right-arm action
        is the drill tip, read back by FK from the IK solution (matching
        auto-collect's tip targets).
        """
        ctx = self._ctx
        model = ctx.robot_model
        target_left = mat4_to_se3(palm_tracking["left"])
        target_right = mat4_to_se3(palm_tracking["right"])

        left_arm_q = model.solve_ik(
            definitions.LEFT_PALM_FRAME, target_left, ctx.full_q
        )
        right_arm_q = model.solve_ik(
            definitions.RIGHT_PALM_FRAME, target_right, ctx.full_q
        )

        # Recorder targets. Left: raw XR palm target. Right: FK tip pose of the IK
        # solution.
        ctx.targets.left_9d = model.se3_to_9d_pose(target_left)
        ctx.targets.right_9d = model.frame_pose_9d(right_arm_q, ctx.right_eef_frame)

        return model.assemble_upper_body_target(
            ctx.default_upper_body_pose, left_arm_q, right_arm_q
        )

    def _get_target_palm_poses(self):
        ctx = self._ctx
        if ctx.robot_state.upper_body_target is not None:
            # Use the commanded target rather than measured joints: gravity droop in
            # full_q would otherwise accumulate across re-activations.
            q = ctx.full_q.copy()
            q[ctx.upper_body_indices] = ctx.robot_state.upper_body_target
            return ctx.robot_model.get_palm_poses(q)
        return ctx.robot_model.get_palm_poses(ctx.full_q)
