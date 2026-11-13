# MIT License
"""VLA_PLANNER state: VLA approach, then scripted precision finish.

Phases:

* ``APPROACHING`` — VLA action chunks drive the arm until the handover gate
  fires (:mod:`g1_control.vla.handover`).
* ``ACCUMULATING`` — arm frozen; fine-source anchored predictions are
  batch-averaged until converged or timed out (optionally re-ranged by
  ``depth_source``).
* ``STANDOFF`` — tip trajectory to the standoff knot (``finalize_standoff_m``
  off the wall, ``finalize_standoff_rise_m`` up it), then re-accumulate.
* ``VISUAL_SERVOING`` — optional image-plane P controller aligning the target
  with the projected drill-tip pixel.
* ``FINALIZING`` — one-segment push onto the target; with
  ``finalize_correct_target`` the goal follows the live estimate's drift.
* ``RECOVERING`` — back to the default pose and restart (up to
  ``abort_max_restarts``), e.g. when the standoff batch is beyond
  ``approach_distance``.

Coarse/fine source roles are resolved in :mod:`g1_control.vla.sources`. The
phase is latched on ``TOPIC_VLA_PLANNER_PHASE_STATUS`` and logged in the
metrics CSV; ``TOPIC_VLA_PLANNER_COMPLETE`` signals arrival (not accuracy).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pinocchio as pin
import rclpy

from g1_definitions import definitions
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray

from g1_control import topics
from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils.ros_helpers import (
    delete_all_markers,
    LATCHED_QOS,
    lookup_position,
    sphere_marker,
)
from g1_control.utils.target_correction import TargetCorrection
from g1_control.utils.task_clock import TaskClock
from g1_control.utils.tip_motion import TipPoseCommander
from g1_control.utils.tip_trajectory import (
    aligned_drill_rotation,
    quintic,
    TipTrajectory,
)
from g1_control.vla.handover import (
    FineEstimateFresh,
    HandoverEvidence,
    HandoverGate,
    SeqFreshness,
)
from g1_control.vla.planner import (
    DEPTH_SOURCE_STEREO,
    DEPTH_SOURCES,
    DepthBiasedSource,
    DepthDonor,
    drill_entry_point,
    finalize_knots,
    FinalizeResidual,
    LiveTargetTracker,
    StereoDepthDonor,
    TargetPoseAccumulator,
    VlaPlannerConfig,
    wall_lateral_of,
    wall_normal_of,
    wall_up_of,
    WallPlaneSlot,
)
from g1_control.vla.visual_servo import ImagePlaneVisualServo

_Phase = definitions.VlaPlannerPhase


def freeze_arm(
    ctx: ControlContext,
    commander: TipPoseCommander,
    vla_actions,
    live_upper_body_command=None,
) -> None:
    """Stop the arm on the live controller reference.

    Falls back to the action waypoint, then the measured pose (which bakes in
    gravity droop). A commander already holding a pose keeps it; reset it first
    to stop *here*.
    """
    ctx.robot_state.upper_body_target_chunk = None
    ctx.targets.left_9d = None
    if commander.commanded_upper_body is None:
        frozen = (
            live_upper_body_command() if live_upper_body_command is not None else None
        )
        if frozen is None:
            frozen = (
                vla_actions.get("upper_body_joint_chunk")
                if vla_actions.has_first_chunk
                else ctx.full_q[ctx.upper_body_indices]
            )
        commander.command_upper_body(np.asarray(frozen, dtype=np.float64))


def commanded_tip_pose(
    ctx: ControlContext, commander: TipPoseCommander, tip_frame: str
) -> pin.SE3:
    """FK of the current command, falling back to measured FK."""
    q = np.asarray(ctx.full_q, dtype=float).copy()
    if commander.commanded_upper_body is not None:
        q[ctx.upper_body_indices] = commander.commanded_upper_body
    model = ctx.robot_model
    model.cache_forward_kinematics(q, auto_clip=False)
    pose = model.frame_placement(tip_frame)
    return pin.SE3(
        np.asarray(pose.rotation, dtype=float).copy(),
        np.asarray(pose.translation, dtype=float).copy(),
    )


class PlannerPhase(ABC):
    """One VLA_PLANNER substate: entered once, then ticked until it yields."""

    def on_enter(self) -> None:
        """Apply the phase's entry side effects."""

    @abstractmethod
    def update(self) -> _Phase | None:
        """Advance one control tick; return the next phase or None to stay."""


class ApproachingPhase(PlannerPhase):
    """VLA chunk playback until the handover gate fires.

    Whole chunks go to the controller's interpolator; the playback pointer
    advances at the record fps.
    """

    def __init__(
        self,
        ctx: ControlContext,
        logger,
        coarse_client,
        fine_client,
        vla_actions,
        gate: HandoverGate,
        advance_every: int,
        timeout_s: float,
        debug_publisher,
        clock,
    ) -> None:
        self._ctx = ctx
        self._logger = logger
        self._coarse_client = coarse_client
        self._fine_client = fine_client
        self._vla_actions = vla_actions
        self._gate = gate
        self._fine_freshness = SeqFreshness()
        self._advance_every = advance_every
        self._timeout_s = timeout_s
        self._debug = debug_publisher
        self._clock = clock
        self._step_counter = 0
        self._started: float = 0.0
        self._last_blocked_log: float = 0.0
        self._last_detection_count = 0

    def on_enter(self) -> None:
        self._gate.reset()
        self._fine_freshness.reset()
        self._step_counter = 0
        self._started = self._clock()
        self._last_blocked_log = self._started
        self._last_detection_count = self._fine_client.detection_count
        # Drop the previous session's chunk so its last waypoint isn't replayed.
        self._vla_actions.reset()

    def _evidence(self, now: float) -> HandoverEvidence:
        """This tick's detector reading (non-blocking slot read)."""
        fine = self._fine_client.get_tip_target_anchored()
        return HandoverEvidence(
            fine=fine,
            fine_age_s=self._fine_freshness.update(fine, now),
            now=now,
        )

    def update(self) -> _Phase | None:
        now = self._clock()
        evidence = self._evidence(now)
        if self._gate.update(evidence):
            self._logger.info(
                f"[vla-planner] approach gate fired ({self._fired_range(evidence)}) "
                "— freezing to accumulate."
            )
            return _Phase.ACCUMULATING
        self._log_blocked(now)

        if now - self._started > self._timeout_s:
            self._logger.warn(
                f"[vla-planner] approach never satisfied the handover gate in "
                f"{self._timeout_s:.0f} s — last blocked on: "
                f"{self._gate.reason or 'nothing (no predictions arrived)'}."
            )
            # Not terminal: restart from the default pose.
            return _Phase.RECOVERING

        is_new_chunk = self._vla_actions.new_action_chunk.is_set()
        self._vla_actions.step()
        if is_new_chunk:
            chunk = self._vla_actions.local_action_buffer.get("upper_body_joint_chunk")
            if chunk is not None:
                self._ctx.robot_state.upper_body_target_chunk = chunk.copy()
            # No fused estimate while approaching; 2D detection comes from the
            # fine client, tip-target frame from the coarse (steering) one.
            self._debug.publish(
                self._vla_actions,
                self._coarse_client,
                None,
                detection_client=self._fine_client,
            )

        right_9d = self._vla_actions.get_right_eef_9d()
        if right_9d is not None:
            self._ctx.targets.right_9d = right_9d

        self._step_counter += 1
        if self._step_counter >= self._advance_every:
            self._step_counter = 0
            self._vla_actions.advance_index()
        return None

    def _fired_range(self, evidence: HandoverEvidence) -> str:
        """The fine source's range at handover, for the log."""
        if evidence.fine is None:
            return "close-range source is producing estimates"
        fine_mm = float(np.linalg.norm(np.asarray(evidence.fine[0][:3], float))) * 1e3
        return f"close-range source estimates {fine_mm:.0f} mm to target"

    def _log_blocked(self, now: float) -> None:
        """Log what the gate is waiting for, with the detection count (rate-limited)."""
        if not self._gate.reason or now - self._last_blocked_log < 5.0:
            return
        elapsed = now - self._last_blocked_log
        count = self._fine_client.detection_count
        seen, self._last_detection_count = count - self._last_detection_count, count
        self._last_blocked_log = now
        self._logger.info(
            f"[vla-planner] approach gate waiting: {self._gate.reason} "
            f"[{seen} detections in {elapsed:.0f} s]"
        )


class AccumulatingPhase(PlannerPhase):
    """Hold the arm still and average detections until the estimate converges.

    ``fine_client`` supplies anchored estimates (``get_tip_target_anchored``).
    An optional ``donor`` re-ranges each detection, keeping its bearing.
    Sampling starts once the tip is within ``accumulate_settle_tol_mm`` of the
    held pose, so arm motion isn't mistaken for perception noise.
    """

    TIP_FRAME = definitions.DRILL_TIP_FRAME

    def __init__(
        self,
        ctx: ControlContext,
        logger,
        fine_client,
        vla_actions,
        accumulator: TargetPoseAccumulator,
        donor: DepthDonor | None,
        commander: TipPoseCommander,
        config: VlaPlannerConfig,
        clock,
        live_upper_body_command=None,
    ) -> None:
        self._ctx = ctx
        self._logger = logger
        self._clock = clock
        self._fine_client = fine_client
        self._vla_actions = vla_actions
        self._accumulator = accumulator
        self._donor = donor
        self._commander = commander
        self._live_upper_body_command = live_upper_body_command
        self._cfg = config
        self._started: float = 0.0
        self._entered: float = 0.0
        self._settled = False
        self._last_skip_log: float = 0.0
        self._last_transition_wait_log: float = 0.0
        self._transition_settled_since: float | None = None

    def on_enter(self) -> None:
        # At the standoff knot the commander already holds the transit end pose.
        freeze_arm(
            self._ctx,
            self._commander,
            self._vla_actions,
            self._live_upper_body_command,
        )

        # Fresh batch each round: perception error is viewpoint-dependent.
        self._accumulator.reset()
        if self._donor is not None:
            self._donor.reset()
        self._last_skip_log = 0.0
        self._last_transition_wait_log = 0.0
        self._transition_settled_since = None
        # Robot clock: detections are produced on it too.
        self._started = self._clock()
        # Reset when the tip settles, so the budget measures sampling only.
        self._entered = self._started
        self._settled = False
        self._logger.info(
            "[vla-planner] arm frozen — waiting for the tip to reach the "
            f"commanded pose (at least {self._cfg.accumulate_settle_min_s:.0f} s, "
            f"then within {self._cfg.accumulate_settle_tol_mm:.0f} mm, "
            f"or {self._cfg.accumulate_settle_timeout_s:.0f} s) before "
            f"accumulating (min {self._cfg.accumulate_min_samples}, spread ≤ "
            f"{self._cfg.accumulate_max_spread_mm:.0f} mm, timeout "
            f"{self._cfg.accumulate_timeout_s:.0f} s"
            + (
                ", range donated by the measured wall)."
                if self._donor is not None
                else ")."
            )
        )

    def _tip_error_m(self) -> float | None:
        """Distance from the drill tip to the held pose.

        ``None`` (treated as settled) without a commanded pose or FK.
        """
        commanded = self._commander.commanded_upper_body
        if commanded is None:
            return None
        model = self._ctx.robot_model
        if not hasattr(model, "cache_forward_kinematics"):
            return None
        q = np.asarray(self._ctx.full_q, dtype=float).copy()
        model.cache_forward_kinematics(q, auto_clip=False)
        measured = np.asarray(model.frame_placement(self.TIP_FRAME).translation)
        q[self._ctx.upper_body_indices] = commanded
        model.cache_forward_kinematics(q, auto_clip=False)
        target = np.asarray(model.frame_placement(self.TIP_FRAME).translation)
        return float(np.linalg.norm(measured - target))

    def _settle(self, now: float) -> bool:
        """Whether sampling may start this tick.

        Timing out still starts sampling: samples are anchored to their own tip
        pose, so a residual offset doesn't bias the batch.
        """
        if self._settled:
            return True
        waited = now - self._entered
        # Minimum wait: an unreacted controller trivially reads as settled.
        if waited < self._cfg.accumulate_settle_min_s:
            return False
        error_m = self._tip_error_m()
        timed_out = waited >= self._cfg.accumulate_settle_timeout_s
        if error_m is None:
            self._settled = True
        elif error_m * 1e3 <= self._cfg.accumulate_settle_tol_mm:
            self._settled = True
            self._logger.info(
                f"[vla-planner] tip settled {error_m * 1e3:.1f} mm from the "
                f"commanded pose after {waited:.1f} s — accumulating."
            )
        elif timed_out:
            self._settled = True
            self._logger.warn(
                f"[vla-planner] tip still {error_m * 1e3:.1f} mm from the "
                f"commanded pose after "
                f"{self._cfg.accumulate_settle_timeout_s:.0f} s — starting "
                "anchored accumulation, but keeping the next phase blocked "
                f"until tracking is within "
                f"{self._cfg.accumulate_settle_tol_mm:.1f} mm."
            )
        if self._settled:
            self._started = now
        return self._settled

    def _ready_to_change_phase(self, now: float) -> bool:
        """Whether the tip has held within the settle tolerance long enough to move
        on."""
        error_m = self._tip_error_m()
        if error_m is None:
            return True
        if error_m * 1e3 <= self._cfg.accumulate_settle_tol_mm:
            if self._transition_settled_since is None:
                self._transition_settled_since = now
            return (
                now - self._transition_settled_since
                >= self._cfg.accumulate_settle_hold_s
            )
        self._transition_settled_since = None
        if now - self._last_transition_wait_log >= 2.0:
            self._last_transition_wait_log = now
            self._logger.warn(
                f"[vla-planner] estimate ready, but tip tracking error is "
                f"{error_m * 1e3:.1f} mm — holding until it is within "
                f"{self._cfg.accumulate_settle_tol_mm:.1f} mm for "
                f"{self._cfg.accumulate_settle_hold_s:.1f} s before changing "
                "phase."
            )
        return False

    def _rejected_suffix(self) -> str:
        """Outlier-rejection note for log lines; empty when none."""
        rejected = self._accumulator.rejected
        return "" if rejected == 0 else f" ({rejected} rejected as outliers)"

    def _sample(self, now: float) -> tuple | None:
        """The detection to accumulate this tick, re-ranged by the donor if any.

        Returns ``None`` when the donor cannot build a sample (no fallback).
        """
        anchored = self._fine_client.get_tip_target_anchored()
        if self._donor is None:
            return anchored

        donation = self._donor.fuse(anchored, now)
        if donation.anchored is not None:
            self._logger.debug(
                f"[vla-planner] range donated: {donation.fine_depth_m * 1e3:.0f} "
                f"-> {donation.fused_depth_m * 1e3:.0f} mm "
                f"({donation.shift_mm:.1f} mm moved)"
            )
        elif donation.reason and now - self._last_skip_log >= 2.0:
            # Throttled; the batch summary carries the count.
            self._last_skip_log = now
            self._logger.warn(
                f"[vla-planner] detection not accumulated — {donation.reason} "
                f"({self._donor.skipped} skipped, {self._donor.fused} built)"
            )
        return donation.anchored

    def _donor_suffix(self) -> str:
        """The donor's tally for a batch log line, empty when it is off."""
        return "" if self._donor is None else f" — {self._donor.summary()}"

    def update(self) -> _Phase | None:
        now = self._clock()
        if not self._settle(now):
            return None

        if self._accumulator.add(self._sample(now)):
            self._logger.info(
                f"[vla-planner] detection {self._accumulator.count}: spread "
                f"{self._accumulator.position_spread() * 1000.0:.1f} mm / "
                f"{self._accumulator.rotation_spread_deg():.2f} deg"
                + self._rejected_suffix()
            )

        if self._accumulator.converged(
            self._cfg.accumulate_min_samples,
            self._cfg.accumulate_max_spread_mm / 1000.0,
        ):
            if not self._ready_to_change_phase(now):
                return None
            if self._donor is not None:
                self._logger.info(f"[vla-planner] {self._donor.summary()}.")
            return _Phase.FINALIZING

        if now - self._started > self._cfg.accumulate_timeout_s:
            if self._accumulator.count > 0:
                if not self._ready_to_change_phase(now):
                    return None
                self._logger.warn(
                    "[vla-planner] accumulation timed out at spread "
                    f"{self._accumulator.position_spread() * 1000.0:.1f} mm "
                    f"({self._accumulator.count} samples"
                    f"{self._rejected_suffix()}) — planning on the mean anyway"
                    f"{self._donor_suffix()}."
                )
                return _Phase.FINALIZING
            self._logger.error(
                f"[vla-planner] no detections arrived in "
                f"{self._cfg.accumulate_timeout_s:.0f} s of accumulating: "
                f"{self._no_detection_diagnosis()} Giving up."
            )
            return _Phase.HOLDING
        return None

    def _no_detection_diagnosis(self) -> str:
        """Why the estimate slot stayed empty, from what the fine client saw."""
        # Checked first: a donor can starve the batch while the detector works.
        if self._donor is not None and self._donor.fused == 0:
            return (
                f"nothing could be built at the measured wall's range "
                f"({self._donor.skipped} detections skipped) — the "
                "close-range source may be fine; look for the "
                "'[vla-planner] detection not accumulated' warning above "
                "for which half was missing, check that g1_sensors' "
                "wall_plane node is up and publishing "
                f"'{definitions.TOPIC_WALL_PLANE}', and set "
                "vla_planner_depth_source to 'fine' to accumulate the "
                "detector's own depth instead."
            )
        detection = self._fine_client.get_last_detection()
        if detection is None:
            return (
                "the fine source has not returned a parseable detection at all "
                "— either nothing is answering it (check fine_server_port), "
                "the model served has no perception head "
                "(look for '[perception] server reply carries no perception "
                "head'), or its inference loop is still waiting on camera "
                "frames."
            )
        return (
            f"the model answered, but nothing reached the accumulator — newest "
            f"detection is uv={np.round(detection.uv_px, 1)} px, depth "
            f"{detection.depth_m:.3f} m, visibility {detection.visibility:.2f} "
            f"(reported, not gated on). Either every sample was rejected as an "
            f"outlier (see vla_planner_accumulate_outlier_mm — the log above "
            f"counts them), or the lift itself failed on missing CameraInfo or "
            f"a missing camera->tip TF, which warns on its own account."
        )


class VisualServoingPhase(PlannerPhase):
    """Align target and drill-tip pixels without changing wall distance."""

    TIP_FRAME = definitions.DRILL_TIP_FRAME

    def __init__(
        self,
        ctx: ControlContext,
        logger,
        accumulator: TargetPoseAccumulator,
        projector,
        commander: TipPoseCommander,
        config: VlaPlannerConfig,
        clock,
        fine_client=None,
    ) -> None:
        self._ctx = ctx
        self._logger = logger
        self._accumulator = accumulator
        self._commander = commander
        self._cfg = config
        self._clock = clock
        self._fine_client = fine_client
        self._servo = ImagePlaneVisualServo(
            kp_s=config.visual_servo_kp_s,
            max_rate_m_s=config.visual_servo_max_rate_mm_s * 1e-3,
            max_offset_m=config.visual_servo_max_offset_mm * 1e-3,
            jacobian_step_m=config.visual_servo_jacobian_step_mm * 1e-3,
            min_singular_px_per_m=config.visual_servo_min_singular_px_per_m,
            ema_alpha=config.visual_servo_ema_alpha,
            max_error_px=config.visual_servo_max_error_px,
            projector=projector,
            # Drill-axis point level with the target, avoiding parallax.
            setpoint_of=drill_entry_point,
        )
        self._base_pose: pin.SE3 | None = None
        self._target_pelvis_m: np.ndarray | None = None
        self._wall_basis = np.zeros((3, 2))
        self._started = 0.0
        self._hold_s = 0.0
        self._last_reason = ""

    @property
    def setpoint_uv_px(self) -> np.ndarray | None:
        return self._servo.setpoint_uv_px

    @property
    def target_uv_px(self) -> np.ndarray | None:
        return self._servo.target_uv_px

    def on_enter(self) -> None:
        self._servo.reset()
        self._base_pose = commanded_tip_pose(self._ctx, self._commander, self.TIP_FRAME)
        self._target_pelvis_m = None
        self._wall_basis = np.zeros((3, 2))
        if not self._load_target():
            self._logger.warn(
                "[vla-planner] visual servo entered before a target estimate "
                "arrived; holding while waiting for one."
            )
        self._started = self._clock()
        self._hold_s = 0.0
        self._last_reason = ""
        self._logger.info(
            "[vla-planner] visual servo engaged: holding wall distance and "
            "aligning target/drill pixels."
        )

    def _load_target(self) -> bool:
        """Latch the newest pelvis-frame target, including a late first sample."""
        estimate = self._accumulator.pose_9d()
        if estimate is None and self._fine_client is not None:
            self._accumulator.add(self._fine_client.get_tip_target_anchored())
            estimate = self._accumulator.pose_9d()
        if estimate is None:
            return False
        estimate = np.asarray(estimate, dtype=float).copy()
        self._target_pelvis_m = estimate[:3].copy()
        self._wall_basis = np.column_stack(
            (wall_lateral_of(estimate), wall_up_of(estimate))
        )
        return True

    def _translation_at(self, offset_wall_m: np.ndarray) -> np.ndarray:
        return self._base_pose.translation + self._wall_basis @ offset_wall_m

    def update(self) -> _Phase | None:
        if self._target_pelvis_m is None:
            self._load_target()
        if (
            self._base_pose is None
            or self._target_pelvis_m is None
            or not np.any(self._wall_basis)
        ):
            return self._timeout_or_wait("no wall-plane basis")

        model = self._ctx.robot_model
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        current_tip = model.frame_placement(self.TIP_FRAME)
        current_rotation = np.asarray(current_tip.rotation, dtype=float)
        current_translation = np.asarray(current_tip.translation, dtype=float)
        target_tip_m = current_rotation.T @ (
            self._target_pelvis_m - current_translation
        )

        result = self._servo.update(
            target_tip_m=target_tip_m,
            tip_rotation_pelvis=current_rotation,
            wall_basis_pelvis=self._wall_basis,
            dt=self._ctx.control_dt,
        )
        if not result.valid or result.error_px is None:
            return self._timeout_or_wait(result.reason or "invalid image geometry")

        translation = self._translation_at(result.offset_wall_m)
        self._commander.command_tip_pose(pin.SE3(self._base_pose.rotation, translation))
        error_px = float(np.linalg.norm(result.error_px))
        self._hold_s = (
            self._hold_s + self._ctx.control_dt
            if error_px <= self._cfg.visual_servo_success_px
            else 0.0
        )
        self._last_reason = ""
        if self._hold_s >= self._cfg.visual_servo_success_hold_s:
            offset_mm = float(np.linalg.norm(result.offset_wall_m)) * 1e3
            self._logger.info(
                f"[vla-planner] visual servo aligned at {error_px:.1f} px "
                f"after {offset_mm:.1f} mm of wall-plane correction."
            )
            return _Phase.FINALIZING
        return self._timeout_or_wait("")

    def _timeout_or_wait(self, reason: str) -> _Phase | None:
        if reason:
            self._hold_s = 0.0
            if reason != self._last_reason:
                self._logger.warn(f"[vla-planner] visual servo paused: {reason}.")
                self._last_reason = reason
        if self._clock() - self._started < self._cfg.visual_servo_timeout_s:
            return None
        self._logger.error(
            f"[vla-planner] visual servo did not converge within "
            f"{self._cfg.visual_servo_timeout_s:.1f} s; restarting the attempt."
        )
        return _Phase.RECOVERING


class FinalizingPhase(PlannerPhase):
    """One leg of the finalizing maneuver.

    ``stop_at_standoff`` flies to the standoff knot (then back to
    ``ACCUMULATING``); otherwise it flies the straight push onto the aim.
    With ``finalize_correct_target`` the command is displaced by the live
    estimate's drift since the plan (:class:`TargetCorrection`). Live
    predictions use the same ``donor`` as the batch so the difference is
    range-consistent.
    """

    TIP_FRAME = definitions.DRILL_TIP_FRAME
    VIZ_FRAME = "pelvis"

    def __init__(
        self,
        ctx: ControlContext,
        logger,
        fine_client,
        accumulator: TargetPoseAccumulator,
        donor: DepthDonor | None,
        commander: TipPoseCommander,
        config: VlaPlannerConfig,
        path_pub,
        marker_pub,
        clock,
        *,
        stop_at_standoff: bool = False,
    ) -> None:
        self._ctx = ctx
        self._logger = logger
        self._fine_client = fine_client
        self._accumulator = accumulator
        self._donor = donor
        self._commander = commander
        self._cfg = config
        self._path_pub = path_pub
        self._marker_pub = marker_pub
        self._clock = clock
        self._stop_at_standoff = bool(stop_at_standoff)
        self._trajectory: TipTrajectory | None = None
        self._motion_start: float = 0.0
        self._completed = False
        # Estimate the current plan was built from (for the arrival log).
        self._plan_estimate_9d: np.ndarray | None = None
        # Live target tracker driving the correction while the tip moves.
        self._tracker = self._build_tracker(donor, logger=logger)
        # Shadow tracker on the detector's own depth: logged, never applied.
        self._shadow = (
            self._build_tracker(None, logger=None, label="fine-only")
            if (
                config.finalize_correct_target
                and donor is not None
                and config.finalize_shadow_fine_only
            )
            else None
        )
        self.set_geometry(
            config.finalize_standoff_m,
            config.finalize_standoff_rise_m,
            config.finalize_transit_duration_s,
            config.finalize_insert_duration_s,
        )

    def _build_tracker(self, donor, *, logger, label: str = "") -> LiveTargetTracker:
        """Build a live tracker from this phase's config; only the donor varies."""
        config = self._cfg
        return LiveTargetTracker(
            correction=TargetCorrection(
                max_rate_m_s=config.finalize_correction_rate_mm_s * 1e-3,
                max_pos_m=config.finalize_correction_max_mm * 1e-3,
                max_rate_rad_s=np.deg2rad(config.finalize_correction_rate_deg_s),
                max_ang_rad=np.deg2rad(config.finalize_correction_max_deg),
            ),
            donor=donor,
            ema_alpha=config.finalize_correction_ema_alpha,
            gate_m=config.finalize_correction_gate_mm * 1e-3,
            max_age_s=config.finalize_correction_max_age_s,
            logger=logger,
            label=label,
        )

    def set_geometry(
        self, standoff_m: float, rise_m: float, transit_s: float, insert_s: float
    ) -> None:
        """Plan geometry for the next entry (standoff uses ``transit_s``, push
        ``insert_s``)."""
        self._standoff_m = float(standoff_m)
        self._rise_m = float(rise_m)
        self._transit_s = float(transit_s)
        self._insert_s = float(insert_s)

    @property
    def completed(self) -> bool:
        """Whether the last entry ran its trajectory to the end."""
        return self._completed

    @property
    def _leg(self) -> str:
        """Which half of the maneuver this instance flies, for the log lines."""
        return "standoff transit" if self._stop_at_standoff else "straight push"

    def on_enter(self) -> None:
        self._trajectory = None
        self._completed = False
        self._plan_estimate_9d = None
        estimate_9d = self._accumulator.pose_9d()
        if estimate_9d is None:
            self._logger.error(f"[vla-planner] {self._leg} with no estimate.")
            return
        self._plan_estimate_9d = np.asarray(estimate_9d, dtype=float).copy()

        tip_se3 = self._trajectory_start_pose()

        knots = finalize_knots(
            tip_se3.translation,
            estimate_9d,
            self._standoff_m,
            self._cfg.finalize_wall_offset_m,
            self._rise_m,
            self._cfg.finalize_wall_lateral_offset_m,
            self._cfg.finalize_wall_up_offset_m,
        )
        # finalize_align_drill fades the drill axis onto the estimated wall
        # normal (not the oblique standoff->aim chord).
        self._rotation0 = tip_se3.rotation.copy()
        self._wall_normal = wall_normal_of(estimate_9d)
        if self._stop_at_standoff:
            motion_knots = knots[:2]
        elif self._cfg.visual_servo_enabled:
            # Keep the servoed wall-plane position; close only the normal distance.
            start = knots[0]
            normal_distance = float(self._wall_normal @ (knots[2] - start))
            motion_knots = [start, start + normal_distance * self._wall_normal]
        else:
            motion_knots = [knots[0], knots[2]]
        duration = self._align_duration_s
        self._trajectory = TipTrajectory(
            knots=motion_knots,
            rotation=self._rotation0,
            segment_durations=[duration],
        )
        # Shared anchor, so tracker divergence isolates the depth source.
        for tracker in self._trackers:
            tracker.reset(estimate_9d, self._accumulator.last_seq)
        self._motion_start = self._clock()
        self._logger.info(
            f"[vla-planner] planned {self._trajectory.length:.3f} m {self._leg} "
            f"path onto the estimate at {estimate_9d[:3].round(4)} "
            f"({self._accumulator.count} detections, spread "
            f"{self._accumulator.position_spread() * 1000.0:.1f} mm): "
            f"{self._trajectory.duration:.1f} s total."
        )
        if self._stop_at_standoff:
            self._log_standoff_plan(knots[1], estimate_9d)
        else:
            self._log_push_plan(motion_knots[-1], estimate_9d)
        self._publish_viz(knots)

    def _trajectory_start_pose(self) -> pin.SE3:
        """FK of the commanded pose (bumpless vs. measured FK), else measured FK."""
        return commanded_tip_pose(self._ctx, self._commander, self.TIP_FRAME)

    def update(self) -> _Phase | None:
        if self._trajectory is None:
            return _Phase.HOLDING

        now = self._clock()
        elapsed = now - self._motion_start
        u = elapsed / self._trajectory.duration
        planned = self._trajectory.pose_at(u)
        if self._cfg.finalize_align_drill:
            planned = pin.SE3(self._aligned_rotation(elapsed), planned.translation)
        if self._cfg.finalize_correct_target:
            self._update_correction(now)
        # Applied last, once; only the steering tracker (not the shadow).
        if self._cfg.finalize_correct_target and not (
            self._cfg.visual_servo_enabled and not self._stop_at_standoff
        ):
            planned = self._tracker.apply(planned)

        self._commander.command_tip_pose(planned)

        if u >= 1.0:
            if self._stop_at_standoff:
                self._log_standoff_arrival()
            else:
                self._log_final_errors()
            if self._cfg.finalize_correct_target:
                self._log_correction()
            self._completed = True
            return _Phase.ACCUMULATING if self._stop_at_standoff else _Phase.HOLDING
        return None

    @property
    def _trackers(self) -> tuple[LiveTargetTracker, ...]:
        """Every live tracker, steering one first. Reset and ticked together."""
        return (
            (self._tracker,) if self._shadow is None else (self._tracker, self._shadow)
        )

    def _update_correction(self, now: float) -> None:
        """Tick every live tracker on one shared reading of the fine client.

        Runs even when the correction is not applied (measurement only).
        """
        fine = self._fine_client.get_tip_target_anchored()
        for tracker in self._trackers:
            tracker.update(now, fine)

    @property
    def live_target_9d(self):
        """Planned estimate plus active correction (published as TF); None until
        planned."""
        if not self._cfg.finalize_correct_target:
            return None
        return self._tracker.corrected_target_9d

    @property
    def shadow_target_9d(self):
        """Shadow (fine-only depth) corrected target; None without a shadow."""
        if not self._cfg.finalize_correct_target or self._shadow is None:
            return None
        return self._shadow.corrected_target_9d

    def _log_correction(self) -> None:
        """Report how far the estimate moved under the plan during the finalize."""
        applied_mm, applied_deg = self._tracker.applied
        raw_mm, raw_deg = self._tracker.raw
        correction_applied = self._cfg.finalize_correct_target and not (
            self._cfg.visual_servo_enabled and not self._stop_at_standoff
        )
        suffix = "" if correction_applied else " (disabled — measuring only)"
        gated = (
            ""
            if self._tracker.gated_out == 0
            else (
                f" {self._tracker.gated_out} prediction(s) ignored as further "
                f"than {self._cfg.finalize_correction_gate_mm:.0f} mm from the plan."
            )
        )
        donated = "" if self._donor is None else f" {self._donor.summary()}."
        self._logger.info(
            f"[vla-planner] target correction: {applied_mm:.1f} mm / "
            f"{applied_deg:.2f} deg applied, {raw_mm:.1f} mm / {raw_deg:.2f} deg "
            f"measured{suffix}.{gated}{donated}"
        )
        self._log_shadow()

    def _log_shadow(self) -> None:
        """Report the fine-only shadow correction and its separation from the flown
        target."""
        if self._shadow is None:
            return
        shadow_mm, shadow_deg = self._shadow.applied
        raw_mm, raw_deg = self._shadow.raw
        steered = self._tracker.corrected_target
        shadowed = self._shadow.corrected_target
        separation = (
            ""
            if steered is None or shadowed is None
            else (
                " Its target sits "
                f"{1000.0 * float(np.linalg.norm(steered.translation - shadowed.translation)):.1f}"
                " mm from the one flown."
            )
        )
        gated = (
            ""
            if self._shadow.gated_out == 0
            else f" {self._shadow.gated_out} prediction(s) gated out."
        )
        self._logger.info(
            f"[vla-planner] fine-only shadow (not applied): {shadow_mm:.1f} mm / "
            f"{shadow_deg:.2f} deg, {raw_mm:.1f} mm / {raw_deg:.2f} deg "
            f"measured.{gated}{separation}"
        )

    @property
    def _align_duration_s(self) -> float:
        """Duration of this instance's leg (also the rotation blend duration)."""
        return self._transit_s if self._stop_at_standoff else self._insert_s

    def _aligned_rotation(self, elapsed: float) -> np.ndarray:
        """Rotation aligning the drill axis to the estimated wall normal.

        Blended 0 -> 1 over the leg with the path's quintic, so it never steps.
        """
        blend = quintic(
            float(np.clip(elapsed / max(self._align_duration_s, 1e-6), 0.0, 1.0))
        )
        return aligned_drill_rotation(self._wall_normal, self._rotation0, blend)

    def _log_final_errors(self) -> None:
        """Log final miss vs. the planned aim (control) and vs. true target TF (sim
        only)."""
        model = self._ctx.robot_model
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        tip = model.frame_placement(self.TIP_FRAME).translation
        # Include the correction so it doesn't read as control error.
        aim = self._trajectory.pose_at(1.0)
        if self._cfg.finalize_correct_target:
            aim = self._tracker.apply(aim)
        aim = aim.translation
        line = f"[vla-planner] finalize done: tip {1000.0 * float(np.linalg.norm(tip - aim)):.1f} mm from the planned aim"
        true_target = lookup_position(
            self._ctx.tf_buffer, self._ctx.drill_target_frame, self.VIZ_FRAME
        )
        if true_target is not None:
            line += (
                f", {1000.0 * float(np.linalg.norm(tip - true_target)):.1f} mm "
                f"from the true {self._ctx.drill_target_frame}"
            )
        self._logger.info(line + ".")

    def _ground_truth_knot(
        self, estimate_9d: np.ndarray, index: int = 1
    ) -> np.ndarray | None:
        """The knot this plan would use with the true target position (sim only).

        ``index`` follows :func:`finalize_knots`: 1 is the standoff, 2 the aim.
        Only position is substituted (the sim target frame has identity
        rotation). Returns None when the true target frame is unavailable.
        """
        true_target = lookup_position(
            self._ctx.tf_buffer, self._ctx.drill_target_frame, self.VIZ_FRAME
        )
        if true_target is None:
            return None
        truth_9d = np.asarray(estimate_9d, dtype=float).copy()
        truth_9d[:3] = true_target
        return finalize_knots(
            np.zeros(3),  # start knot, unused
            truth_9d,
            self._standoff_m,
            self._cfg.finalize_wall_offset_m,
            self._rise_m,
            self._cfg.finalize_wall_lateral_offset_m,
            self._cfg.finalize_wall_up_offset_m,
        )[index]

    def _wall_local_mm(self, delta: np.ndarray, estimate_9d: np.ndarray) -> str:
        """A pelvis-frame offset in wall-local mm: normal (depth), lateral, up."""
        return (
            f"normal {1000.0 * float(delta @ wall_normal_of(estimate_9d)):+.1f}, "
            f"lateral {1000.0 * float(delta @ wall_lateral_of(estimate_9d)):+.1f}, "
            f"up {1000.0 * float(delta @ wall_up_of(estimate_9d)):+.1f} mm"
        )

    def _log_standoff_plan(self, standoff: np.ndarray, estimate_9d) -> None:
        """Log the planned standoff knot vs. the true one (pure estimate error)."""
        truth = self._ground_truth_knot(estimate_9d)
        if truth is None:
            return
        delta = standoff - truth
        self._logger.info(
            f"[vla-planner] standoff planned at {standoff.round(4)}, true "
            f"standoff {truth.round(4)} off {self._ctx.drill_target_frame}: "
            f"estimate is {1000.0 * float(np.linalg.norm(delta)):.1f} mm out "
            f"({self._wall_local_mm(delta, estimate_9d)})."
        )

    def _log_push_plan(self, aim: np.ndarray, estimate_9d) -> None:
        """Log the planned push endpoint and the estimate error vs. truth (sim only).

        Both are logged since, under visual servoing, only the normal coordinate
        of the aim comes from the estimate.
        """
        truth_aim = self._ground_truth_knot(estimate_9d, 2)
        true_target = lookup_position(
            self._ctx.tf_buffer, self._ctx.drill_target_frame, self.VIZ_FRAME
        )
        if truth_aim is None or true_target is None:
            return
        delta = aim - truth_aim
        estimate_error = np.asarray(estimate_9d, dtype=float)[:3] - true_target
        self._logger.info(
            f"[vla-planner] push aims at {aim.round(4)}, true aim "
            f"{truth_aim.round(4)} off {self._ctx.drill_target_frame} at "
            f"{true_target.round(4)}: lands "
            f"{1000.0 * float(np.linalg.norm(delta)):.1f} mm out "
            f"({self._wall_local_mm(delta, estimate_9d)}); the estimate itself "
            f"is {1000.0 * float(np.linalg.norm(estimate_error)):.1f} mm out "
            f"({self._wall_local_mm(estimate_error, estimate_9d)})."
        )

    def _log_standoff_arrival(self) -> None:
        """Log the reached standoff vs. commanded and true standoff."""
        if self._trajectory is None:
            return
        model = self._ctx.robot_model
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        tip = model.frame_placement(self.TIP_FRAME).translation
        # Include the correction so it doesn't read as tracking error.
        aim = self._trajectory.pose_at(1.0)
        if self._cfg.finalize_correct_target:
            aim = self._tracker.apply(aim)
        aim = aim.translation
        line = (
            f"[vla-planner] standoff reached at {tip.round(4)}: "
            f"{1000.0 * float(np.linalg.norm(tip - aim)):.1f} mm from the "
            "commanded standoff"
        )
        truth = (
            None
            if self._plan_estimate_9d is None
            else self._ground_truth_knot(self._plan_estimate_9d)
        )
        if truth is not None:
            delta = tip - truth
            line += (
                f", {1000.0 * float(np.linalg.norm(delta)):.1f} mm from the true "
                f"standoff {truth.round(4)} "
                f"({self._wall_local_mm(delta, self._plan_estimate_9d)})"
            )
        self._logger.info(line + ".")

    def _publish_viz(self, knots: list[np.ndarray]) -> None:
        """Planned path plus estimate (red) / standoff (blue) spheres for RViz."""
        stamp = rclpy.time.Time().to_msg()
        path = Path()
        path.header.frame_id = self.VIZ_FRAME
        path.header.stamp = stamp
        for u in np.linspace(0.0, 1.0, 100):
            pose = PoseStamped()
            pose.header.frame_id = self.VIZ_FRAME
            pose.header.stamp = stamp
            position = self._trajectory.pose_at(float(u)).translation
            pose.pose.position.x = float(position[0])
            pose.pose.position.y = float(position[1])
            pose.pose.position.z = float(position[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

        markers = MarkerArray()
        markers.markers.append(
            sphere_marker("vla_planner", 0, knots[-1], (0.9, 0.1, 0.1, 0.9), 0.01)
        )
        markers.markers.append(
            sphere_marker("vla_planner", 1, knots[-2], (0.1, 0.4, 0.9, 0.6), 0.02)
        )
        self._marker_pub.publish(markers)


class RecoveringPhase(PlannerPhase):
    """Drive the arm back to its default pose in joint space, then retry."""

    #: Per-joint error (rad) below which the arm counts as home.
    SETTLE_TOL_RAD = 0.05

    def __init__(
        self,
        ctx: ControlContext,
        logger,
        vla_actions,
        commander: TipPoseCommander,
        config: VlaPlannerConfig,
        clock,
    ) -> None:
        self._ctx = ctx
        self._logger = logger
        self._vla_actions = vla_actions
        self._commander = commander
        self._cfg = config
        self._clock = clock
        self._started: float = 0.0

    def on_enter(self) -> None:
        # Reset first so the previous hold doesn't override the default pose.
        self._commander.reset()
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.targets.left_9d = None
        self._vla_actions.reset()
        self._commander.command_upper_body(
            np.asarray(self._ctx.default_upper_body_pose, dtype=np.float64)
        )
        self._started = self._clock()
        self._logger.warn(
            "[vla-planner] returning to the default pose before retrying."
        )

    def update(self) -> _Phase | None:
        commanded = self._commander.commanded_upper_body
        if commanded is None:
            return _Phase.APPROACHING
        measured = np.asarray(self._ctx.full_q, float)[self._ctx.upper_body_indices]
        error = float(np.max(np.abs(measured - commanded)))
        if error <= self.SETTLE_TOL_RAD:
            self._logger.info(
                "[vla-planner] back at the default pose — restarting the " "approach."
            )
            return _Phase.APPROACHING
        if self._clock() - self._started >= self._cfg.recover_timeout_s:
            self._logger.warn(
                f"[vla-planner] still {error:.2f} rad from the default pose "
                f"after {self._cfg.recover_timeout_s:.0f} s — restarting the "
                "approach from here anyway."
            )
            return _Phase.APPROACHING
        return None


class HoldingPhase(PlannerPhase):
    """Terminal: the handler keeps re-applying the commander's last target."""

    def update(self) -> _Phase | None:
        return None


class VlaPlannerHandler(StateHandler):
    """Routes ticks through the :class:`PlannerPhase` objects.

    Owns phase routing, the command tail, FSM handoff, metrics and the
    phase-status topic.
    """

    def __init__(
        self,
        node,
        ctx: ControlContext,
        coarse_client,
        fine_client,
        vla_actions,
        metrics,
        debug_publisher,
        config: VlaPlannerConfig,
        tip_projector,
        controller=None,
    ) -> None:
        self._ctx = ctx
        self._logger = node.get_logger()
        self._clock = TaskClock(node)
        # Coarse drives the approach; fine supplies the close-range estimate.
        self._coarse = coarse_client
        self._clients = [coarse_client, fine_client]
        # Depth bias applied once at the source, shared by all phases.
        fine_client = self._fine = self._depth_biased(
            fine_client, config, tip_projector
        )
        self._vla_actions = vla_actions
        self._metrics = metrics
        self._debug = debug_publisher
        self._cfg = config
        self._last_debug_pub: float = 0.0

        self._accumulator = TargetPoseAccumulator(
            max_outlier_m=config.accumulate_outlier_mm / 1000.0
        )
        # Measured wall plane, shared by all donors.
        self._wall = WallPlaneSlot()
        self._warned_wall_frame = False
        self._wall_sub = self._subscribe_wall_plane(node, config)
        # One donor per phase: each has its own tally and freshness clock.
        self._donor = self._depth_donor(config, tip_projector, self._wall.latest)
        standoff_donor = self._depth_donor(config, tip_projector, self._wall.latest)
        finalize_donor = self._depth_donor(config, tip_projector, self._wall.latest)
        self._commander = TipPoseCommander(
            ctx.robot_model, ctx, definitions.DRILL_TIP_FRAME
        )

        self._phase_pub = node.create_publisher(
            String, definitions.TOPIC_VLA_PLANNER_PHASE_STATUS, LATCHED_QOS
        )
        self._complete_pub = node.create_publisher(
            Bool, definitions.TOPIC_VLA_PLANNER_COMPLETE, LATCHED_QOS
        )
        self._path_pub = node.create_publisher(
            Path, topics.TOPIC_VLA_PLANNER_TRAJECTORY, LATCHED_QOS
        )
        self._marker_pub = node.create_publisher(
            MarkerArray, topics.TOPIC_VLA_PLANNER_MARKERS, LATCHED_QOS
        )

        # Control ticks per recorded chunk step.
        advance_every = max(1, round(1.0 / (ctx.control_dt * definitions.RECORD_FPS)))
        self._phases: dict[_Phase, PlannerPhase] = {
            _Phase.APPROACHING: ApproachingPhase(
                ctx,
                self._logger,
                coarse_client,
                fine_client,
                vla_actions,
                self._approach_gate(config),
                advance_every,
                config.approach_timeout_s,
                debug_publisher,
                self._clock,
            ),
            _Phase.ACCUMULATING: AccumulatingPhase(
                ctx,
                self._logger,
                fine_client,
                vla_actions,
                self._accumulator,
                self._donor,
                self._commander,
                config,
                self._clock,
                live_upper_body_command=(
                    (lambda: controller.last_upper_body_action)
                    if controller is not None
                    else None
                ),
            ),
            _Phase.VISUAL_SERVOING: VisualServoingPhase(
                ctx,
                self._logger,
                self._accumulator,
                tip_projector,
                self._commander,
                config,
                self._clock,
                fine_client=fine_client,
            ),
            _Phase.STANDOFF: FinalizingPhase(
                ctx,
                self._logger,
                fine_client,
                self._accumulator,
                standoff_donor,
                self._commander,
                config,
                self._path_pub,
                self._marker_pub,
                self._clock,
                stop_at_standoff=True,
            ),
            _Phase.FINALIZING: FinalizingPhase(
                ctx,
                self._logger,
                fine_client,
                self._accumulator,
                finalize_donor,
                self._commander,
                config,
                self._path_pub,
                self._marker_pub,
                self._clock,
            ),
            _Phase.RECOVERING: RecoveringPhase(
                ctx,
                self._logger,
                vla_actions,
                self._commander,
                config,
                self._clock,
            ),
            _Phase.HOLDING: HoldingPhase(),
        }
        self._phase = _Phase.HOLDING
        # Failed attempts abandoned back to the default pose this session.
        self._restarts = 0
        # True between a completed transit and its push (ACCUMULATING at the knot).
        self._at_standoff = False
        self._success_time: float | None = None
        self._settled_since: float | None = None
        self._completion_logged = False

        self._logger.info(
            "VLA-planner state ready: approach gate fresh estimate x"
            f"{config.approach_consecutive}, accumulate "
            f"{config.accumulate_min_samples} samples / "
            f"{config.accumulate_max_spread_mm:.0f} mm spread at the "
            f"{config.depth_source} range, finalize "
            f"standoff {config.finalize_standoff_m * 1000.0:.0f} mm"
            + (
                f" raised {config.finalize_standoff_rise_m * 1000.0:.0f} mm "
                f"({self._oblique_deg(config):.0f} deg oblique)"
                if config.finalize_standoff_rise_m
                else " (straight in)"
            )
            + (
                f", close-range depth biased {config.fine_depth_bias_mm:+.1f} mm."
                if config.fine_depth_bias_mm
                else "."
            )
        )

    @staticmethod
    def _oblique_deg(config: VlaPlannerConfig) -> float:
        """How far off the wall normal a raised standoff tilts the push."""
        return float(
            np.rad2deg(
                np.arctan2(config.finalize_standoff_rise_m, config.finalize_standoff_m)
            )
        )

    @staticmethod
    def _depth_biased(fine_client, config: VlaPlannerConfig, tip_projector):
        """``fine_client`` reading ``fine_depth_bias_mm`` further out; unwrapped if
        zero."""
        bias_m = float(config.fine_depth_bias_mm) / 1000.0
        if not bias_m:
            return fine_client
        return DepthBiasedSource(fine_client, bias_m, tip_projector.extrinsic_tip_cam)

    #: Required wall-plane frame: StereoDepthDonor uses pelvis-frame tip anchors.
    _WALL_PLANE_FRAME = "pelvis"

    def _subscribe_wall_plane(self, node, config: VlaPlannerConfig):
        """Subscribe (latest-only QoS) to the wall plane when ``depth_source`` is
        stereo."""
        if str(config.depth_source) != DEPTH_SOURCE_STEREO:
            return None
        from g1_interfaces.msg import WallPlane
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._logger.info(
            f"[vla-planner] depth_source '{DEPTH_SOURCE_STEREO}': taking the "
            f"range from '{definitions.TOPIC_WALL_PLANE}'. Nothing accumulates "
            "until that topic publishes — check g1_sensors' wall_plane node is "
            "up and its head depth stream is enabled."
        )
        return node.create_subscription(
            WallPlane,
            definitions.TOPIC_WALL_PLANE,
            self._wall_plane_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

    def _wall_plane_callback(self, msg) -> None:
        """Store the newest wall plane; fit quality is gated by the publisher."""
        from g1_common.plane import Plane

        if msg.header.frame_id != self._WALL_PLANE_FRAME:
            if not self._warned_wall_frame:
                self._warned_wall_frame = True
                self._logger.error(
                    f"[vla-planner] wall plane arrived in "
                    f"'{msg.header.frame_id}' but the range is computed as "
                    f"though it were in '{self._WALL_PLANE_FRAME}' — the "
                    "donor carries it into a detection's camera frame through "
                    "that detection's own tip anchor, which is a pelvis-frame "
                    "pose. Ignoring these planes; re-launch the wall_plane "
                    f"node with reference_frame:={self._WALL_PLANE_FRAME}."
                )
            return

        self._wall.publish(
            Plane(
                point=np.array([msg.point.x, msg.point.y, msg.point.z]),
                normal=np.array([msg.normal.x, msg.normal.y, msg.normal.z]),
                inliers=int(msg.inliers),
                sampled=int(msg.sampled),
                rms_m=float(msg.rms_m),
            ),
            frame_id=msg.header.frame_id,
        )

    @staticmethod
    def _depth_donor(
        config: VlaPlannerConfig, tip_projector, plane_provider=None
    ) -> DepthDonor | None:
        """The range donor for one phase, or None to accumulate as published.

        Uses the projector's extrinsic so donor and lift agree. Raises on an
        unknown source or ``"stereo"`` without a plane provider.
        """
        source = str(config.depth_source)
        if source not in DEPTH_SOURCES:
            raise ValueError(
                f"vla_planner_depth_source must be one of "
                f"{', '.join(DEPTH_SOURCES)}; got {source!r}"
            )
        if source == DEPTH_SOURCE_STEREO:
            if plane_provider is None:
                raise ValueError(
                    f"vla_planner_depth_source is {source!r} but no wall-plane "
                    "source was wired in; the handler must pass one"
                )
            return StereoDepthDonor(
                tip_projector.extrinsic_tip_cam,
                plane_provider,
                max_age_s=config.depth_max_age_s,
                max_shift_m=config.depth_max_shift_mm / 1000.0,
            )
        return None

    @staticmethod
    def _approach_gate(config: VlaPlannerConfig) -> HandoverGate:
        """Handover gate: a fresh fine estimate for N consecutive ticks."""
        return HandoverGate(
            FineEstimateFresh(config.approach_detection_max_age_s),
            config.approach_consecutive,
            rearm_s=config.approach_rearm_s,
        )

    # ── State lifecycle ───────────────────────────────────────────────────────

    def on_enter(self, previous_state: int) -> None:
        self._accumulator.reset()
        self._commander.reset()
        self._restarts = 0
        self._at_standoff = False
        self._success_time = None
        self._settled_since = None
        self._completion_logged = False
        # Clear the latched completion flag for this rollout.
        self._complete_pub.publish(Bool(data=False))
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None

        # Score the calibrated estimate this state plans on.
        self._metrics.set_estimate_source(self._fine)
        self._metrics.start(
            prefix="vla_planner",
            tip_estimator=self._accumulator,
        )
        # Both sources infer for the whole session.
        for client in self._clients:
            client.start_inference_loop()

        if not self._ctx.has_drill_tip:
            self._logger.error(
                "[vla-planner] no drill-tip frame in the URDF — cannot plan; "
                "holding."
            )
            self._enter_phase(_Phase.HOLDING)
        else:
            self._enter_phase(_Phase.APPROACHING)

    def on_exit(self) -> None:
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.targets.clear()
        for client in self._clients:
            client.pause_inference_loop()
        self._metrics.finalize()

        empty = Path()
        empty.header.frame_id = FinalizingPhase.VIZ_FRAME
        self._path_pub.publish(empty)
        self._marker_pub.publish(delete_all_markers("vla_planner"))

    # ── Per-tick handler ──────────────────────────────────────────────────────

    def update(self) -> None:
        self._ctx.robot_state.twist_command = np.zeros(3)

        next_phase = self._phases[self._phase].update()
        if next_phase is not None:
            self._enter_phase(self._route(next_phase))

        # Outside the approach, re-apply the commander's cached target each tick.
        if self._phase is not _Phase.APPROACHING:
            self._ctx.robot_state.upper_body_target_chunk = None
            if self._commander.commanded_upper_body is not None:
                self._ctx.robot_state.upper_body_target = (
                    self._commander.commanded_upper_body
                )
            self._ctx.targets.right_9d = self._commander.commanded_eef_9d
            # RViz overlays from the active motion leg's trackers; 10 Hz while
            # visual servoing, else 1 Hz.
            motion = self._phases[
                (
                    _Phase.STANDOFF
                    if self._phase is _Phase.STANDOFF
                    or (self._phase is _Phase.ACCUMULATING and self._at_standoff)
                    else _Phase.FINALIZING
                )
            ]
            now = time.monotonic()
            debug_period_s = 0.1 if self._phase is _Phase.VISUAL_SERVOING else 1.0
            if now - self._last_debug_pub >= debug_period_s:
                self._last_debug_pub = now
                visual = self._phases[_Phase.VISUAL_SERVOING]
                self._debug.publish_perception(
                    self._fine.get_tip_target(),
                    self._accumulator.pose_9d(),
                    self._fine.get_last_detection(),
                    driver_tip_target_9d=self._coarse.get_tip_target(),
                    live_9d=motion.live_target_9d,
                    live_fine_only_9d=motion.shadow_target_9d,
                    setpoint_uv_px=visual.setpoint_uv_px,
                    servo_target_uv_px=(
                        visual.target_uv_px
                        if self._phase is _Phase.VISUAL_SERVOING
                        else None
                    ),
                )

        self._maybe_log_completion()
        # Sessions are segmented by ``phase``, not waypoints.
        self._metrics.record(
            self._clock(), waypoint_idx=-1, phase=float(self._phase.value)
        )

    # ── Phase plumbing ────────────────────────────────────────────────────────

    def _route(self, requested: _Phase) -> _Phase:
        """Route a phase's requested successor.

        Applies cross-phase rules: restart budget, which ACCUMULATING viewpoint
        asked, and the far-mode check at the standoff.
        """
        standoff = self._phases[_Phase.STANDOFF]
        finalizing = self._phases[_Phase.FINALIZING]

        # Restart budget is owned here.
        if self._phase is _Phase.APPROACHING and requested is _Phase.RECOVERING:
            return self._restart()
        if self._phase is _Phase.VISUAL_SERVOING and requested is _Phase.RECOVERING:
            return self._restart()

        # Standoff reached: re-accumulate before the push.
        if (
            self._phase is _Phase.STANDOFF
            and requested is _Phase.ACCUMULATING
            and standoff.completed
        ):
            self._at_standoff = True
            self._logger.info(
                "[vla-planner] standoff knot reached — stopping to "
                "re-accumulate before the straight push."
            )
            return _Phase.ACCUMULATING

        # Push landed: done.
        if (
            self._phase is _Phase.FINALIZING
            and requested is _Phase.HOLDING
            and finalizing.completed
        ):
            self._at_standoff = False
            self._mark_success()
            return _Phase.HOLDING

        # ACCUMULATING leads to STANDOFF before the transit, to the push at the knot.
        if self._phase is _Phase.ACCUMULATING and requested is _Phase.FINALIZING:
            if not self._at_standoff:
                return _Phase.STANDOFF
            # At the knot, a batch beyond approach_distance is far-mode, not a miss.
            residual = self._residual()
            if (
                residual is not None
                and residual.distance_m > self._cfg.approach_distance
            ):
                return self._abort(residual)
            self._logger.info(
                "[vla-planner] standoff estimate converged — starting "
                + (
                    "visual alignment."
                    if self._cfg.visual_servo_enabled
                    else "the straight push."
                )
            )
            return (
                _Phase.VISUAL_SERVOING
                if self._cfg.visual_servo_enabled
                else _Phase.FINALIZING
            )

        return requested

    def _abort(self, residual: FinalizeResidual) -> _Phase:
        """The measurement is not of the target; abandon the attempt."""
        self._logger.error(
            f"[vla-planner] measured {residual.distance_m * 1e3:.0f} mm from "
            f"the target with the tip one standoff away, past the "
            f"{self._cfg.approach_distance * 1e3:.0f} mm far-mode bound — the "
            "close-range source is not seeing the target, so this estimate "
            "cannot be corrected towards."
        )
        return self._restart()

    def _restart(self) -> _Phase:
        """Abandon this attempt and recover to the default pose, up to
        ``abort_max_restarts``."""
        self._restarts += 1
        limit = self._cfg.abort_max_restarts
        if self._restarts > limit:
            self._logger.error(
                f"[vla-planner] {limit} restart(s) already spent; giving up "
                "rather than cycling. Holding."
            )
            return _Phase.HOLDING
        # Reset per-attempt state.
        self._accumulator.reset()
        self._at_standoff = False
        self._success_time = None
        self._logger.warn(
            f"[vla-planner] restart {self._restarts}/{limit}: abandoning this "
            "attempt and approaching again from the default pose."
        )
        return _Phase.RECOVERING

    def _residual(self) -> FinalizeResidual | None:
        """Batch target in the tip frame (+x = drill axis), split about the axis.

        None when the batch is empty or FK is unavailable.
        """
        position = self._accumulator.position()
        model = self._ctx.robot_model
        if position is None or not hasattr(model, "cache_forward_kinematics"):
            return None
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        placement = model.frame_placement(definitions.DRILL_TIP_FRAME)
        target_tip = np.asarray(placement.rotation, dtype=float).T @ (
            np.asarray(position, dtype=float)
            - np.asarray(placement.translation, dtype=float)
        )
        return FinalizeResidual.of(target_tip, self._cfg.finalize_wall_offset_m)

    def _enter_phase(self, phase: _Phase) -> None:
        self._phase = phase
        self._logger.info(f"[vla-planner] phase -> {phase.name}")
        self._phase_pub.publish(String(data=phase.name.lower()))
        self._phases[phase].on_enter()

    def _mark_success(self) -> None:
        """Finalize completed; the tracking-qualified completion clock may start."""
        self._success_time = self._clock()
        self._settled_since = None

    def _maybe_log_completion(self) -> None:
        """Latch completion once the final pose is held within tolerance for
        ``complete_hold_s``."""
        if self._completion_logged or self._success_time is None:
            return
        now = self._clock()
        error_m = self._phases[_Phase.ACCUMULATING]._tip_error_m()
        if error_m is not None and error_m * 1e3 > self._cfg.accumulate_settle_tol_mm:
            self._settled_since = None
            return
        if self._settled_since is None:
            self._settled_since = now
            return
        if now - self._settled_since < self._cfg.complete_hold_s:
            return
        self._completion_logged = True
        self._complete_pub.publish(Bool(data=True))
        self._logger.info(
            f"[vla-planner] COMPLETE: tracking held within "
            f"{self._cfg.accumulate_settle_tol_mm:.1f} mm for "
            f"{self._cfg.complete_hold_s:.1f} s — holding the final pose."
        )
