# MIT License
"""GOAL_POSE state: translate the drill tip in its own frame via IK.

Each ``geometry_msgs/Vector3`` received by GOAL_POSE is an xyz translation in
the *current* ``drill_tip_link`` frame. The state snapshots the measured tip
pose when the command arrives, composes the translation once, and holds the
resulting pelvis-frame goal. The tip orientation is unchanged.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from g1_definitions import definitions
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import MarkerArray

from g1_control import topics
from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils.ros_helpers import (
    delete_all_markers,
    LATCHED_QOS,
    pose_error,
    sphere_marker,
)


def relative_tip_goal(tip_se3: pin.SE3, translation_xyz) -> pin.SE3:
    """Translate ``tip_se3`` by xyz expressed in the tip's local frame.

    The frame convention is ``pelvis_T_goal = pelvis_T_tip * tip_T_goal``.
    The returned pose keeps the tip's orientation.
    """
    translation = np.asarray(translation_xyz, dtype=float)
    if translation.shape != (3,):
        raise ValueError(f"expected xyz translation, got shape {translation.shape}")
    if not np.all(np.isfinite(translation)):
        raise ValueError("xyz translation contains NaN or infinity")
    return pin.SE3(
        tip_se3.rotation.copy(),
        tip_se3.translation + tip_se3.rotation @ translation,
    )


class GoalPoseHandler(StateHandler):
    """Drive the tip by a received local-frame xyz translation."""

    FRAME = definitions.DRILL_TIP_FRAME
    SIDE = "right"

    def __init__(self, node, ctx: ControlContext, controller) -> None:
        self._ctx = ctx
        self._logger = node.get_logger()
        # Active controller; its gravity FF is compared with reported tau_est.
        self._controller = controller

        goal_topic = (
            node.get_parameter("goal_pose_topic").value
            if node.has_parameter("goal_pose_topic")
            else "/goal_pose"
        ) or "/goal_pose"
        self._goal_topic = goal_topic

        # Resolved once on receipt (re-resolving per tick would chase the tip).
        self._goal_se3: pin.SE3 | None = None

        robot_model = ctx.robot_model

        # Right-arm joint indices (full pinocchio q) and names, for tracking logs.
        idx_to_name = {i: n for n, i in robot_model.joint_to_dof_index.items()}
        self._arm_indices = robot_model.get_joint_group_indices(f"{self.SIDE}_arm")
        self._arm_names = [idx_to_name.get(i, f"q[{i}]") for i in self._arm_indices]

        # Waist joints (IK-locked), logged against their nominal hold since waist
        # drift shifts the arm base.
        self._waist_indices = robot_model.get_joint_group_indices("waist")
        self._waist_names = [idx_to_name.get(i, f"q[{i}]") for i in self._waist_indices]
        self._waist_nominal = robot_model.default_body_pose[self._waist_indices].copy()

        self._goal_sub = node.create_subscription(
            Vector3, goal_topic, self._goal_callback, 1
        )
        # Under ik_debug/, not /goal_pose/*, to avoid an RViz type clash.
        self._marker_pub = node.create_publisher(
            MarkerArray, topics.TOPIC_IK_GOAL_MARKERS, LATCHED_QOS
        )

        self._logger.info(
            f"Goal-pose state ready: listening for drill-tip-relative xyz "
            f"translations on '{goal_topic}' (drives {self.FRAME})."
        )

    # ── State lifecycle ───────────────────────────────────────────────────────

    def on_enter(self, previous_state: int) -> None:
        """Wait for a fresh goal; hold the pose the controller seeded on activation
        until one arrives (no stale target/chunk drives the arm)."""
        self._goal_se3 = None
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None

    def on_exit(self) -> None:
        """Drop the last IK target (only meaningful while this state refreshes it)
        and clear the RViz markers so a stale goal isn't left hanging."""
        self._ctx.targets.right_9d = None
        self._ctx.robot_state.upper_body_target = None
        self._marker_pub.publish(delete_all_markers("goal_pose"))

    # ── Per-tick handler ──────────────────────────────────────────────────────

    def update(self) -> None:
        """Drive the drill tip to the resolved relative goal via per-tick IK.

        Uses the single-target path (Path B) with the base held still. Until a
        goal arrives the arm holds the pose seeded on activation. Records the
        commanded joints' 9D tip pose.
        """
        ctx = self._ctx
        rs = ctx.robot_state
        # Hold the base still; never populate the chunk so Path B is used.
        rs.twist_command = np.zeros(3)
        rs.upper_body_target_chunk = None

        goal_se3 = self._goal_se3
        if goal_se3 is None:
            self._logger.info(
                "[goal-pose] GOAL_POSE active; waiting for a drill-tip-relative "
                f"xyz translation on '{self._goal_topic}'...",
                throttle_duration_sec=2.0,
            )
            return

        right_arm_q = ctx.robot_model.solve_ik(self.FRAME, goal_se3, ctx.full_q)
        rs.upper_body_target = ctx.robot_model.assemble_upper_body_target(
            ctx.default_upper_body_pose, ctx.full_q, right_arm_q
        )
        ctx.targets.right_9d = ctx.robot_model.frame_pose_9d(right_arm_q, self.FRAME)

        self._log_residual(goal_se3, right_arm_q)
        self._log_joint_error(right_arm_q)
        self._publish_markers(goal_se3)

    # ── Goal resolution ───────────────────────────────────────────────────────

    def _goal_callback(self, msg: Vector3) -> None:
        """Snapshot the measured tip and resolve one local xyz command."""
        if not self._ctx.has_drill_tip:
            self._logger.error(
                f"Rejected goal translation: '{self.FRAME}' is not in the URDF."
            )
            return

        relative_xyz = np.array([msg.x, msg.y, msg.z], dtype=float)
        try:
            self._ctx.robot_model.cache_forward_kinematics(
                self._ctx.full_q, auto_clip=False
            )
            current_tip = self._ctx.robot_model.frame_placement(self.FRAME)
            goal_se3 = relative_tip_goal(current_tip, relative_xyz)
        except ValueError as exc:
            self._logger.error(f"Rejected goal translation: {exc}")
            return

        self._goal_se3 = goal_se3
        relative_text = ", ".join(f"{value:+.3f}" for value in relative_xyz)
        goal_text = ", ".join(f"{value:.3f}" for value in goal_se3.translation)
        self._logger.info(
            f"Goal translation received in '{self.FRAME}': [{relative_text}] m -> "
            f"pelvis goal [{goal_text}] m"
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _log_residual(self, goal_se3: pin.SE3, right_arm_q: np.ndarray) -> None:
        """Decompose the drill-tip goal error into its sources (~1 Hz).

        Position (mm) and orientation (deg):

        * ``ik``     – FK of the IK solution vs the goal (solver residual).
        * ``track``  – measured FK vs the IK solution's FK (tracking error).
        """
        ctx = self._ctx
        robot_model = ctx.robot_model

        # Pure IK residual: FK of the joint vector the solver returned.
        robot_model.cache_forward_kinematics(right_arm_q, auto_clip=False)
        ik_se3 = robot_model.frame_placement(self.FRAME)
        ik_pos, ik_ang = pose_error(goal_se3, ik_se3)

        # The achieved pose needs FK cached at the live (measured) q.
        robot_model.cache_forward_kinematics(ctx.full_q, auto_clip=False)

        if ctx.current_right_eef_9d is None:
            self._logger.info(
                f"[goal-pose] ik={ik_pos * 1e3:.1f}mm/{np.degrees(ik_ang):.2f}deg "
                "(no measured drill-tip FK yet)",
                throttle_duration_sec=1.0,
            )
            return

        achieved_se3 = robot_model.frame_placement(self.FRAME)
        track_pos, track_ang = pose_error(ik_se3, achieved_se3)
        total_pos, total_ang = pose_error(goal_se3, achieved_se3)

        self._logger.info(
            f"[goal-pose] ik={ik_pos * 1e3:.1f}mm/{np.degrees(ik_ang):.2f}deg | "
            f"track={track_pos * 1e3:.1f}mm/{np.degrees(track_ang):.2f}deg | "
            f"total={total_pos * 1e3:.1f}mm/{np.degrees(total_ang):.2f}deg",
            throttle_duration_sec=1.0,
        )

    def _log_joint_error(self, right_arm_q: np.ndarray) -> None:
        """Log desired (IK) vs actual (measured) right-arm joints, per joint (~1 Hz).

        ``right_arm_q`` is the IK solution (full pinocchio q); ``ctx.full_q`` is
        measured.
        """
        full_q = self._ctx.full_q

        def _rows(names, desired, actual):
            desired = np.asarray(desired)
            actual = np.asarray(actual)
            err = desired - actual
            body = "\n  ".join(
                f"{name:<26} des={np.degrees(d):8.2f}  act={np.degrees(a):8.2f}  "
                f"err={np.degrees(e):+7.2f} deg"
                for name, d, a, e in zip(names, desired, actual, err)
            )
            worst = names[int(np.argmax(np.abs(err)))]
            norm = np.degrees(np.linalg.norm(err))
            return body, worst, norm

        # Arm: desired = IK solution, actual = measured.
        arm_body, arm_worst, arm_norm = _rows(
            self._arm_names,
            np.asarray(right_arm_q)[self._arm_indices],
            full_q[self._arm_indices],
        )
        # Waist: desired = nominal hold (IK-excluded), actual = measured.
        waist_body, waist_worst, waist_norm = _rows(
            self._waist_names,
            self._waist_nominal,
            full_q[self._waist_indices],
        )

        self._logger.info(
            f"[goal-pose] right-arm joint tracking "
            f"(worst: {arm_worst}, ‖err‖={arm_norm:.2f} deg):\n  {arm_body}\n"
            f"  -- waist (hold, IK-excluded; "
            f"worst: {waist_worst}, ‖err‖={waist_norm:.2f} deg) --\n  {waist_body}",
            throttle_duration_sec=1.0,
        )
        self._log_ff_residual()

    def _log_ff_residual(self) -> None:
        """Compare gravity feedforward with the reported ``tau_est`` values.

        In Isaac Sim ``tau_est`` is the projected joint force; the meaning is
        backend-dependent.
        """
        g_PR = self._controller.last_gravity_ff
        tau_reported = self._ctx.robot_state.joint_torque
        if g_PR is None or tau_reported is None:
            return

        s, e = definitions.JOINT_PR_SLICES["right_arm"]
        g_arm = np.asarray(g_PR)[s:e]
        tau_arm = np.asarray(tau_reported)[s:e]
        rows = "\n  ".join(
            f"{name:<26} g_PR={g:+7.2f}  tau_est={t:+7.2f}  "
            f"ratio={(t / g) if abs(g) > 1e-3 else float('nan'):+6.2f}  "
            f"difference={t - g:+6.2f} Nm"
            for name, g, t in zip(self._arm_names, g_arm, tau_arm)
        )
        self._logger.info(
            f"[goal-pose] gravity FF vs reported tau_est:\n  {rows}",
            throttle_duration_sec=1.0,
        )

    def _publish_markers(self, goal_se3: pin.SE3) -> None:
        """Publish goal (blue) and achieved-tip (green) spheres for RViz."""
        markers = MarkerArray()
        markers.markers.append(
            sphere_marker(
                "goal_pose", 0, goal_se3.translation, (0.1, 0.4, 0.9, 0.9), 0.04
            )
        )
        current = self._ctx.current_right_eef_9d
        if current is not None:
            markers.markers.append(
                sphere_marker("goal_pose", 1, current[:3], (0.1, 0.9, 0.1, 0.9), 0.04)
            )
        self._marker_pub.publish(markers)
