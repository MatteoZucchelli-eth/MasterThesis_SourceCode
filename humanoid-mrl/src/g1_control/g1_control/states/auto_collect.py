# MIT License
"""Scripted, recorded drill-tip approaches onto a target square.

Phases: go to a randomized start pose, plan through a standoff, record the
approach, save only after a measured settled hold, then reset the scene.
"""
from __future__ import annotations

import time

import numpy as np
import pinocchio as pin

from g1_definitions import definitions
from std_msgs.msg import Empty, String, UInt8

from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils.drill_task_sampler import DrillTask, DrillTaskSampler
from g1_control.utils.ros_helpers import LATCHED_QOS, lookup_se3, pose_error
from g1_control.utils.target_correction import TargetCorrection
from g1_control.utils.target_visibility import TargetVisibility, VisibilityReport
from g1_control.utils.task_clock import TaskClock
from g1_control.utils.tip_trajectory import (
    aligned_drill_rotation,
    levelled_drill_rotation,
    quintic,
    TipTrajectory,
    unit,
)

_Phase = definitions.AutoCollectPhase

_VISIBILITY_ALARM_SKIPS = 10
"""Consecutive off-camera configurations before the skip log escalates to an error."""


class AutoCollectHandler(StateHandler):
    """Phase-sequenced scripted approach of the drill tip onto the target square."""

    TIP_FRAME = definitions.DRILL_TIP_FRAME
    BASE_FRAME = "pelvis"

    def __init__(self, node, ctx: ControlContext) -> None:
        self._ctx = ctx
        self._node = node
        self._logger = node.get_logger()
        self._robot_model = ctx.robot_model

        # Task durations use ROS time; only liveness checks use wall time.
        self._clock = TaskClock(node)
        self._clock_ok = False
        self._start_wall = time.monotonic()

        self._standoff_m = self._param(node, "auto_collect_standoff_m", 0.12)
        # Transit (start → standoff) and insertion (standoff → target) are timed
        # separately; the slow insertion lets the arm integral term null droop.
        self._transit_duration_s = self._param(
            node, "auto_collect_transit_duration_s", 8.0
        )
        self._insert_duration_s = self._param(
            node, "auto_collect_insert_duration_s", 12.0
        )
        self._return_duration_s = self._param(
            node, "auto_collect_return_duration_s", 4.0
        )
        # Standoff settle gate: freeze the command at the standoff until the
        # lateral miss is nulled, since contact friction pins it once loaded.
        # Zero disables the hold.
        self._standoff_settle_mm = float(
            self._param(node, "auto_collect_standoff_settle_mm", 1.0)
        )
        self._standoff_settle_hold_s = float(
            self._param(node, "auto_collect_standoff_settle_hold_s", 0.5)
        )
        self._standoff_settle_timeout_s = float(
            self._param(node, "auto_collect_standoff_settle_timeout_s", 6.0)
        )
        # Return arrival gate: the measured arm lags the ramp, so the next
        # episode starts only once every upper-body joint has held within
        # `_return_settle_deg` of the start pose for `_return_settle_hold_s`.
        # Otherwise the first recorded action jumps. 1 deg ≈ 12 mm at the tip.
        self._return_settle_deg = float(
            self._param(node, "auto_collect_return_settle_deg", 1.0)
        )
        self._return_settle_hold_s = float(
            self._param(node, "auto_collect_return_settle_hold_s", 0.3)
        )
        # Give up this long after the ramp ends and proceed with an error log.
        self._return_settle_timeout_s = float(
            self._param(node, "auto_collect_return_settle_timeout_s", 10.0)
        )
        # Turn the drill axis onto the wall normal (from the target TF) so the
        # insertion is perpendicular; otherwise hold the start FK orientation.
        self._align_drill = bool(self._param(node, "auto_collect_align_drill", True))
        # Level the roll about the wall normal: tip y-axis perpendicular to the
        # wall up-direction (target TF z-axis), so the handle hangs straight down.
        self._level_drill = bool(self._param(node, "auto_collect_level_drill", True))
        # Roll offset from level about the drill axis (positive: handle towards
        # robot's right). Only used with auto_collect_level_drill.
        self._drill_roll_rad = np.radians(
            float(self._param(node, "auto_collect_drill_roll_deg", 0.0))
        )
        # Target correction: re-read the target TF every tick and shift the
        # commanded pose by its drift since planning, so FK model error at the
        # plan and execution configurations cancels. Not a servo; the limits
        # below are glitch protection only.
        self._correct_target = bool(
            self._param(node, "auto_collect_correct_target", True)
        )
        # When disabled, the correction is still computed and logged, not applied.
        self._correction = TargetCorrection(
            # Slew-rate limit, not a gain.
            max_rate_m_s=float(
                self._param(node, "auto_collect_correction_rate_mm_s", 5.0)
            )
            * 1e-3,
            # Beyond this is a tracking fault, not FK error; saturation is logged.
            max_pos_m=float(self._param(node, "auto_collect_correction_max_mm", 30.0))
            * 1e-3,
            max_rate_rad_s=np.deg2rad(
                float(self._param(node, "auto_collect_correction_rate_deg_s", 3.0))
            ),
            max_ang_rad=np.deg2rad(
                float(self._param(node, "auto_collect_correction_max_deg", 3.0))
            ),
        )
        phase_topic = (
            self._param(
                node, "auto_collect_phase_topic", definitions.TOPIC_AUTO_COLLECT_PHASE
            )
            or definitions.TOPIC_AUTO_COLLECT_PHASE
        )

        # Recording starts when APPROACHING begins and is saved during the hold.
        self._auto_record = bool(self._param(node, "auto_collect_auto_record", True))
        self._hold_record_s = float(
            self._param(node, "auto_collect_hold_record_s", 3.0)
        )

        # Settle gate: save only after the measured tip holds within these
        # thresholds for `_hold_record_s`; discard after `_settle_timeout_s`.
        self._settle_pos_mm = float(
            self._param(node, "auto_collect_settle_pos_mm", 5.0)
        )
        self._settle_ang_deg = float(
            self._param(node, "auto_collect_settle_ang_deg", 3.0)
        )
        self._settle_timeout_s = float(
            self._param(node, "auto_collect_settle_timeout_s", 15.0)
        )

        # Right-arm start-pose jitter. Disable for eval.
        self._randomize_start = bool(
            self._param(node, "auto_collect_randomize_start", True)
        )
        self._start_noise_rad = np.deg2rad(
            float(self._param(node, "auto_collect_start_pose_noise_deg", 3.0))
        )

        # Per-episode target cell and instruction. Disable for eval (fixed red task).
        self._randomize_task = bool(
            self._param(node, "auto_collect_randomize_task", True)
        )
        self._task_sampler = DrillTaskSampler()
        self._episode_task: DrillTask | None = None

        # Camera gate: plan only if the target projects inside at least one
        # prepared (squared, rectified) camera frame at the start pose;
        # otherwise skip the configuration.
        self._require_target_visible = bool(
            self._param(node, "auto_collect_require_target_visible", True)
        )
        # Border clearance of the projected centre in metres (square side = 1 cm).
        self._visibility_margin_m = float(
            self._param(node, "auto_collect_visibility_margin_m", 0.01)
        )
        # Cameras placed by URDF FK: headless workers have no robot_state_publisher.
        self._visibility = (
            TargetVisibility(node, self._robot_model)
            if self._require_target_visible
            else None
        )
        if self._visibility is not None and not self._visibility.cameras:
            self._logger.error(
                "Auto-collect: auto_collect_require_target_visible is set but the "
                "loaded URDF carries no camera mount frame, so nothing can be "
                "checked — collecting without the gate."
            )
            self._visibility = None
        # Consecutive skips; reset by the first visible configuration.
        self._visibility_skips = 0

        # Loop mode: chain episodes unattended. Finishing an episode requests a
        # sim reset, whose announcement drives the retract. Manual phase
        # publishes still pre-empt the loop.
        self._loop = bool(self._param(node, "auto_collect_loop", False))
        # Idle time in HOLDING (no episode open) before a sim reset is re-requested.
        self._loop_stall_timeout_s = float(
            self._param(node, "auto_collect_loop_stall_timeout_s", 30.0)
        )
        self._last_progress: float = time.monotonic()

        # Only the right arm is jittered; APPROACHING resets the left to default.
        self._rel_right_arm = self._robot_model.get_relative_group_indices(
            "right_arm", "upper_body"
        )
        abs_right_arm = self._robot_model.get_joint_group_indices("right_arm")
        self._right_arm_lower = self._robot_model.lower_joint_limits[abs_right_arm]
        self._right_arm_upper = self._robot_model.upper_joint_limits[abs_right_arm]

        # Upper-body slot labels for arrival-gate logs.
        n_upper = len(self._robot_model.get_joint_group_indices("upper_body"))
        self._upper_body_labels = [f"upper_body[{i}]" for i in range(n_upper)]
        for group in ("left_arm", "right_arm", "left_hand"):
            for slot, rel in enumerate(
                self._robot_model.get_relative_group_indices(group, "upper_body")
            ):
                self._upper_body_labels[rel] = f"{group}[{slot}]"

        # ── Phase / motion state (all reset in on_enter) ──────────────────────
        self._phase = _Phase.TRAJECTORY_GENERATION
        self._trajectory: TipTrajectory | None = None
        self._motion_start: float | None = None
        # Per-plan alignment inputs: start FK rotation, wall normal, wall up.
        self._rotation0: np.ndarray | None = None
        self._wall_normal: np.ndarray | None = None
        self._wall_up: np.ndarray | None = None
        # Clock reading of the last target-correction update, for its slew limit.
        self._last_correction_t: float | None = None
        # Currently commanded upper-body joints (also held in HOLDING).
        self._commanded_upper_body: np.ndarray | None = None
        self._commanded_eef_9d: np.ndarray | None = None
        self._return_start: np.ndarray | None = None
        # GOING_TO_DEFAULT target: default pose plus optional right-arm jitter.
        self._return_target: np.ndarray | None = None
        # Start of the current continuous stretch inside the start-pose band.
        self._return_settled_since: float | None = None
        # An episode spans approach + hold; _capture_active tracks it even when
        # auto_record is off. _hold_started marks the hold start.
        self._capture_active: bool = False
        self._recording: bool = False
        self._hold_started: float | None = None
        # Start of the current continuous stretch inside the tip settle band.
        self._settled_since: float | None = None
        # Standoff hold: _standoff_pause is trajectory time swallowed by the hold.
        self._standoff_pause: float = 0.0
        self._standoff_released: bool = False
        self._standoff_wait_start: float | None = None
        self._standoff_settled_since: float | None = None

        self._phase_sub = node.create_subscription(
            String, phase_topic, self._phase_callback, 1
        )
        # Recorder contract: 1 start, 0 save, 3 discard.
        self._record_pub = node.create_publisher(
            UInt8, definitions.TOPIC_RECORD_EPISODE, 1
        )
        # Sim reset announcement marks an episode boundary.
        self._sim_reset_sub = node.create_subscription(
            Empty, definitions.TOPIC_SIM_RESET, self._sim_reset_callback, 1
        )
        # Loop mode: request a scene randomization; answered on TOPIC_SIM_RESET.
        self._reset_request_pub = node.create_publisher(
            Empty, definitions.TOPIC_SIM_RESET_REQUEST, 1
        )
        self._phase_pub = node.create_publisher(
            String, definitions.TOPIC_AUTO_COLLECT_PHASE_STATUS, LATCHED_QOS
        )
        # Sampled instruction; latched so a late recorder still sees it.
        self._task_pub = node.create_publisher(
            String, definitions.TOPIC_TASKINSTRUCTION, LATCHED_QOS
        )

        self._logger.info(
            f"Auto-collect state ready: phases on '{phase_topic}' "
            f"({', '.join(p.name.lower() for p in _Phase)}); drives {self.TIP_FRAME}"
            f"{'; LOOP mode: episodes chain automatically' if self._loop else ''}."
        )

    @staticmethod
    def _param(node, name: str, default):
        if not node.has_parameter(name):
            return default
        value = node.get_parameter(name).value
        return default if value is None else value

    # ── State lifecycle ───────────────────────────────────────────────────────

    def on_enter(self, previous_state: int) -> None:
        """Start in TRAJECTORY_GENERATION, or in loop mode HOLDING plus a sim reset."""
        self._ctx.robot_state.upper_body_target = None
        self._ctx.robot_state.upper_body_target_chunk = None

        self._trajectory = None
        self._rotation0 = None
        self._wall_normal = None
        self._wall_up = None
        self._correction.reset(None)
        self._last_correction_t = None
        self._motion_start = None
        self._commanded_upper_body = None
        self._commanded_eef_9d = None
        self._return_start = None
        self._return_target = None
        self._return_settled_since = None
        self._capture_active = False
        self._recording = False
        self._hold_started = None
        self._settled_since = None
        self._episode_task = None
        self._visibility_skips = 0
        if self._loop:
            self._set_phase(_Phase.HOLDING)
            self._request_sim_reset("entering AUTO_COLLECT in loop mode")
        else:
            self._set_phase(_Phase.TRAJECTORY_GENERATION)

    def on_exit(self) -> None:
        """Clear targets and the RViz path/trace/markers so nothing stale is left."""
        self._ctx.targets.right_9d = None
        self._ctx.robot_state.upper_body_target = None

        # Discard any partial episode.
        self._finish_episode(save=False, reason="AUTO_COLLECT exited")

    # ── Phase plumbing ────────────────────────────────────────────────────────

    def _sim_reset_callback(self, _msg: Empty) -> None:
        """On sim reset (AUTO_COLLECT only): drop the stale plan and retract."""
        if self._ctx.fsm_state != definitions.FSMState.AUTO_COLLECT.value:
            return
        self._logger.info("Auto-collect: sim reset — retracting to a new start pose.")
        self._trajectory = None
        self._enter_phase(_Phase.GOING_TO_DEFAULT)

    def _phase_callback(self, msg: String) -> None:
        try:
            phase = _Phase.parse(msg.data)
        except (KeyError, ValueError):
            valid = ", ".join(p.name.lower() for p in _Phase)
            self._logger.warn(f"Auto-collect: unknown phase '{msg.data}' ({valid})")
            return
        self._logger.info(f"Auto-collect: phase requested -> {phase.name}")
        self._enter_phase(phase)

    def _enter_phase(self, phase: _Phase) -> None:
        """Apply the per-phase entry side effects, then latch the new phase."""
        previous = self._phase

        # Only APPROACHING -> HOLDING keeps an episode alive; anything else aborts it.
        if self._capture_active and not (
            previous is _Phase.APPROACHING and phase is _Phase.HOLDING
        ):
            self._finish_episode(
                save=False, reason=f"phase changed to {phase.name} mid-approach"
            )

        if phase is _Phase.TRAJECTORY_GENERATION:
            self._trajectory = None
        elif phase is _Phase.APPROACHING:
            # A re-issued `approaching` replays the plan from its start.
            self._motion_start = self._clock()
            self._standoff_pause = 0.0
            self._standoff_released = False
            self._standoff_wait_start = None
            self._standoff_settled_since = None
            # Only record if there is a plan to execute.
            if self._trajectory is not None:
                self._begin_episode()
        elif phase is _Phase.GOING_TO_DEFAULT:
            self._motion_start = self._clock()
            self._return_start = self._current_upper_body()
            self._return_target = self._sample_start_pose()
            self._return_settled_since = None
            # Per-episode reset: draw the next task before the first frame.
            self._sample_episode_task()
        elif phase is _Phase.HOLDING:
            self._hold_started = self._clock()
            self._settled_since = None

        self._set_phase(phase)

    def _set_phase(self, phase: _Phase) -> None:
        self._phase = phase
        self._last_progress = time.monotonic()
        self._phase_pub.publish(String(data=phase.name.lower()))

    def _request_sim_reset(self, reason: str) -> None:
        """Ask the sim to randomize the scene (loop mode's reset-key press)."""
        self._last_progress = time.monotonic()
        self._reset_request_pub.publish(Empty())
        self._logger.info(f"Auto-collect: sim reset requested ({reason}).")

    # ── Episode capture (dataset recording) ───────────────────────────────────

    def _begin_episode(self) -> None:
        """Open an episode when the approach begins."""
        if self._capture_active:
            return
        self._capture_active = True
        self._hold_started = None
        if self._auto_record:
            self._recording = True
            self._record_pub.publish(UInt8(data=1))
            self._logger.info("Auto-collect: recording started (approach begins).")

    def _finish_episode(
        self, *, save: bool, reason: str = "", chain_next: bool = False
    ) -> None:
        """Close the episode, saving or discarding the recorder's buffer.

        ``chain_next`` (settle gate only) requests the next sim reset in loop mode;
        mid-approach aborts must not set it or the scene double-randomizes.
        """
        if not self._capture_active:
            return
        self._capture_active = False
        self._hold_started = None
        self._settled_since = None

        if self._recording:
            self._recording = False
            self._record_pub.publish(UInt8(data=0 if save else 3))
            if save:
                self._logger.info(
                    "Auto-collect: recording saved after "
                    f"{self._hold_record_s:.1f} s settled on target."
                )
            else:
                self._logger.warn(f"Auto-collect: recording discarded ({reason}).")

        if self._loop and chain_next:
            self._request_sim_reset("episode finished, starting the next one")

    def _tip_error_to_target(self) -> tuple[float, float]:
        """Measured-FK drill-tip error (mm, deg) against the final goal pose."""
        model = self._robot_model
        goal = self._goal_pose()
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        tip = model.frame_placement(self.TIP_FRAME)
        pos, ang = pose_error(goal, tip)
        return pos * 1e3, float(np.degrees(ang))

    def _maybe_finish_episode(self) -> None:
        """Save after ``auto_collect_hold_record_s`` settled; discard on timeout."""
        if not self._capture_active or self._hold_started is None:
            return
        if self._trajectory is None:
            return

        now = self._clock()
        pos_mm, ang_deg = self._tip_error_to_target()
        settled = pos_mm <= self._settle_pos_mm and ang_deg <= self._settle_ang_deg

        if settled:
            if self._settled_since is None:
                self._settled_since = now
                self._logger.info(
                    f"Auto-collect: tip settled ({pos_mm:.1f} mm / {ang_deg:.2f} deg); "
                    f"holding {self._hold_record_s:.1f} s before saving."
                )
            if now - self._settled_since >= self._hold_record_s:
                self._finish_episode(save=True, chain_next=True)
            return

        self._settled_since = None
        if now - self._hold_started >= self._settle_timeout_s:
            self._finish_episode(
                save=False,
                reason=(
                    f"tip never settled within {self._settle_timeout_s:.0f} s "
                    f"(last {pos_mm:.1f} mm / {ang_deg:.2f} deg)"
                ),
                chain_next=True,
            )

    def _maybe_kick_stalled_loop(self) -> None:
        """Re-request a sim reset if the loop idles in HOLDING with no episode open."""
        if not self._loop or self._capture_active:
            return
        # Wall clock: a stalled sim also stalls ROS time.
        idle_s = time.monotonic() - self._last_progress
        if idle_s < self._loop_stall_timeout_s:
            return
        self._logger.warn(
            f"Auto-collect: loop idle for {idle_s:.0f} s with no episode open — "
            "requesting a sim reset to restart the cycle."
        )
        self._request_sim_reset("loop stalled")

    # ── Task randomisation ────────────────────────────────────────────────────

    def _sample_episode_task(self) -> None:
        """Sample the next target square and instruction and publish it.

        Writes ``ctx.drill_target_frame``, which planner, labels and metrics read.
        No-op when task randomisation is off.
        """
        if not self._randomize_task:
            return
        task = self._task_sampler.sample()
        self._episode_task = task
        self._ctx.drill_target_frame = task.target_frame
        self._task_pub.publish(String(data=task.description))
        self._logger.info(
            f"Auto-collect: task sampled -> '{task.description}' "
            f"({task.target_frame})."
        )

    # ── Camera gate ───────────────────────────────────────────────────────────

    def _target_in_shot(self, target_se3: pin.SE3) -> bool:
        """True if the pelvis-frame target is visible from the measured configuration.

        False means "not yet" (retry next tick) or "no" (configuration skipped).
        """
        if self._visibility is None:
            return True

        report = self._visibility.check(
            self._ctx.full_q, target_se3, self._visibility_margin_m
        )
        if report is None:
            self._logger.warn(
                "Auto-collect: cannot check target visibility yet — "
                f"{self._visibility.waiting_on()}.",
                throttle_duration_sec=2.0,
            )
            return False

        if report.visible:
            self._visibility_skips = 0
            self._logger.debug(f"Auto-collect: target in shot — {report.summary}.")
            return True

        self._skip_configuration(report)
        return False

    def _skip_configuration(self, report: VisibilityReport) -> None:
        """Abandon a configuration whose target lands in neither camera frame."""
        self._visibility_skips += 1
        message = (
            f"Auto-collect: skipping this configuration — "
            f"{self._ctx.drill_target_frame} falls outside every recorded camera "
            f"frame ({report.summary}); {self._visibility_skips} in a row."
        )
        # Separate call sites per severity: rclpy rejects severity changes per line.
        if self._visibility_skips >= _VISIBILITY_ALARM_SKIPS:
            self._logger.error(
                message + " Something is pointing the wrong way — check the "
                "camera frames and the target TFs; no episode will be recorded "
                "until one of them has the square in view."
            )
        else:
            self._logger.warn(message)

        # No plan, so hold; loop mode re-randomizes the scene.
        self._enter_phase(_Phase.HOLDING)
        if self._loop:
            self._request_sim_reset("target not visible in any recorded camera")

    # ── Start-pose randomisation ──────────────────────────────────────────────

    def _sample_start_pose(self) -> np.ndarray:
        """The pose GOING_TO_DEFAULT drives to: default, plus optional arm jitter."""
        target = np.asarray(self._ctx.default_upper_body_pose, dtype=np.float64).copy()
        if not self._randomize_start or self._start_noise_rad <= 0.0:
            return target

        jitter = np.random.uniform(
            -self._start_noise_rad, self._start_noise_rad, size=len(self._rel_right_arm)
        )
        target[self._rel_right_arm] = np.clip(
            target[self._rel_right_arm] + jitter,
            self._right_arm_lower,
            self._right_arm_upper,
        )
        self._logger.info(
            "Auto-collect: start pose jittered by "
            f"{np.degrees(jitter).round(2)} deg (right arm)."
        )
        return target

    # ── Clocks ────────────────────────────────────────────────────────────────

    def _check_clock_is_running(self) -> None:
        """Log an error if ``use_sim_time`` is on but ``/clock`` never publishes."""
        if self._clock_ok or not self._clock.stalled():
            self._clock_ok = self._clock() > 0.0
            return
        if time.monotonic() - self._start_wall < 5.0:
            return  # still plausibly waiting for the sim's first message
        self._logger.error(
            "use_sim_time is set but /clock has published nothing — ROS time is "
            "stuck at 0, so no approach, hold or timeout can ever elapse. Start "
            "the simulation, or launch with use_sim_time:=false (the correct "
            "setting on the real robot, where there is no /clock).",
            throttle_duration_sec=10.0,
        )

    # ── Per-tick handler ──────────────────────────────────────────────────────

    def update(self) -> None:
        """Advance the active phase by one tick; base held, ``upper_body_target``
        set."""
        self._check_clock_is_running()

        rs = self._ctx.robot_state
        rs.twist_command = np.zeros(3)
        rs.upper_body_target_chunk = None

        if self._phase is _Phase.TRAJECTORY_GENERATION:
            self._generate_trajectory()
        elif self._phase is _Phase.APPROACHING:
            self._approach()
        elif self._phase is _Phase.GOING_TO_DEFAULT:
            self._go_to_default()
        elif self._phase is _Phase.HOLDING:
            self._maybe_finish_episode()
            self._maybe_kick_stalled_loop()

        # Keep commanding the last solution.
        if self._commanded_upper_body is not None:
            rs.upper_body_target = self._commanded_upper_body
        self._ctx.targets.right_9d = self._commanded_eef_9d

    # ── Phases ────────────────────────────────────────────────────────────────

    def _generate_trajectory(self) -> None:
        """Plan ``tip -> standoff -> target`` (straight-in along the wall normal).

        Skips the configuration if the target is off camera. Then HOLDING, or
        APPROACHING in loop mode.
        """
        # Sample a task if none was drawn at GOING_TO_DEFAULT.
        if self._episode_task is None:
            self._sample_episode_task()

        target_se3 = self._target_pose()
        if target_se3 is None:
            return
        if not self._target_in_shot(target_se3):
            return
        target_pos = target_se3.translation.copy()

        self._robot_model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        tip_se3 = self._robot_model.frame_placement(self.TIP_FRAME)

        # Into-the-wall normal (pelvis frame) = target frame x-axis.
        self._wall_normal = unit(target_se3.rotation[:, 0])
        # Wall up = target frame z-axis.
        self._wall_up = unit(target_se3.rotation[:, 2])
        self._rotation0 = tip_se3.rotation.copy()
        standoff = target_pos - self._standoff_m * self._wall_normal
        # Anchor the correction to the planning-time target.
        self._correction.reset(target_se3)
        self._last_correction_t = None

        self._trajectory = TipTrajectory(
            knots=[tip_se3.translation.copy(), standoff, target_pos],
            rotation=self._rotation0,
            segment_durations=[self._transit_duration_s, self._insert_duration_s],
        )
        self._logger.info(
            f"Auto-collect: planned {self._trajectory.length:.3f} m tip path to "
            f"{self._ctx.drill_target_frame} at {target_pos.round(4)} (pelvis frame): "
            f"{self._transit_duration_s:.1f} s transit + "
            f"{self._insert_duration_s:.1f} s insertion "
            f"({self._trajectory.duration:.1f} s total); "
            f"drill axis {self._start_misalignment_deg():.1f} deg off the wall "
            f"normal ({'aligning' if self._align_drill else 'holding it'}), "
            f"roll {self._start_roll_deg():.1f} deg off level "
            f"({'levelling' if self._level_drill else 'keeping it'})."
        )
        self._enter_phase(_Phase.APPROACHING if self._loop else _Phase.HOLDING)

    def _approach(self) -> None:
        """Track the planned tip pose for this instant; hold on target at the end."""
        if self._trajectory is None:
            self._logger.warn(
                "Auto-collect: APPROACHING with no trajectory — send "
                "'trajectory_generation' first.",
                throttle_duration_sec=2.0,
            )
            return

        now = self._clock()
        self._update_correction(now)
        elapsed = self._standoff_gate(now, now - self._motion_start)
        u = elapsed / self._trajectory.duration
        self._command_tip_pose(self._planned_pose(elapsed))

        if u >= 1.0:
            self._log_residual(self._goal_pose())
            self._log_correction()
            self._logger.info("Auto-collect: approach complete — holding on target.")
            self._enter_phase(_Phase.HOLDING)

    def _standoff_gate(self, now: float, raw_elapsed: float) -> float:
        """Trajectory time for this tick, pinned at the standoff while the hold is
        active.

        Returns ``raw_elapsed`` minus the time swallowed by the hold. Releases after
        ``auto_collect_standoff_settle_timeout_s`` regardless.
        """
        elapsed = raw_elapsed - self._standoff_pause
        if (
            self._standoff_settle_mm <= 0.0
            or self._standoff_released
            or elapsed < self._transit_duration_s
        ):
            return elapsed

        # Swallow this tick: the command stays at the standoff.
        self._standoff_pause = raw_elapsed - self._transit_duration_s
        if self._standoff_wait_start is None:
            self._standoff_wait_start = now

        lateral_mm, normal_mm = self._standoff_miss_mm()
        if lateral_mm <= self._standoff_settle_mm:
            if self._standoff_settled_since is None:
                self._standoff_settled_since = now
            if now - self._standoff_settled_since >= self._standoff_settle_hold_s:
                self._release_standoff(now, lateral_mm, normal_mm, "nulled")
        else:
            self._standoff_settled_since = None

        if (
            not self._standoff_released
            and now - self._standoff_wait_start >= self._standoff_settle_timeout_s
        ):
            self._release_standoff(now, lateral_mm, normal_mm, "timed out")
        return self._transit_duration_s

    def _standoff_miss_mm(self) -> tuple[float, float]:
        """Measured tip miss at the standoff as (lateral, along-normal) in mm."""
        model = self._robot_model
        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        tip = model.frame_placement(self.TIP_FRAME)
        delta = (
            tip.translation - self._planned_pose(self._transit_duration_s).translation
        )
        along = float(delta @ self._wall_normal)
        lateral = delta - along * self._wall_normal
        return float(np.linalg.norm(lateral)) * 1e3, along * 1e3

    def _release_standoff(
        self, now: float, lateral_mm: float, normal_mm: float, why: str
    ) -> None:
        """Open the gate and record what the tip looked like when the push began."""
        self._standoff_released = True
        waited = now - (self._standoff_wait_start or now)
        message = (
            f"Auto-collect: standoff {why} after {waited:.1f} s — lateral miss "
            f"{lateral_mm:.2f} mm (tolerance {self._standoff_settle_mm:.2f} mm), "
            f"{normal_mm:+.2f} mm along the normal; starting the insertion."
        )
        # Separate call sites per severity: rclpy rejects severity changes per line.
        if why == "nulled":
            self._logger.info(message)
        else:
            self._logger.warning(message)

    def _update_correction(self, now: float) -> None:
        """Fold the live target estimate into the correction; missing TF holds it."""
        target_se3 = self._target_pose()
        dt = 0.0 if self._last_correction_t is None else now - self._last_correction_t
        self._last_correction_t = now
        if target_se3 is not None:
            self._correction.update(target_se3, dt)

    def _planned_pose(self, elapsed: float) -> pin.SE3:
        """Commanded tip pose ``elapsed`` seconds into the approach.

        With alignment on, rotation blends (quintic) onto the wall normal over the
        transit leg. Target correction is applied last so all readers share it.
        """
        planned = self._trajectory.pose_at(elapsed / self._trajectory.duration)
        if self._align_drill and self._wall_normal is not None:
            blend = quintic(
                float(np.clip(elapsed / max(self._transit_duration_s, 1e-6), 0.0, 1.0))
            )
            planned = pin.SE3(
                aligned_drill_rotation(
                    self._wall_normal,
                    self._rotation0,
                    blend,
                    up=self._wall_up if self._level_drill else None,
                    roll_rad=self._drill_roll_rad,
                ),
                planned.translation,
            )
        return self._correction.apply(planned) if self._correct_target else planned

    def _log_correction(self) -> None:
        """Report how far the plan had gone stale by the end of the approach."""
        applied_mm, applied_deg = self._correction.applied
        raw_mm, raw_deg = self._correction.raw
        suffix = (
            "" if self._correct_target else " (correction disabled — measuring only)"
        )
        self._logger.info(
            f"[auto-collect] target correction: {applied_mm:.1f} mm / "
            f"{applied_deg:.2f} deg applied, {raw_mm:.1f} mm / {raw_deg:.2f} deg "
            f"measured{suffix}"
        )

    def _start_misalignment_deg(self) -> float:
        """Angle between the start pose's drill axis and the wall normal [deg]."""
        if self._rotation0 is None or self._wall_normal is None:
            return float("nan")
        cos = float(np.clip(self._rotation0[:, 0] @ self._wall_normal, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _start_roll_deg(self) -> float:
        """Roll of the start pose about the wall normal, measured off level [deg]."""
        if (
            self._rotation0 is None
            or self._wall_normal is None
            or self._wall_up is None
        ):
            return float("nan")
        levelled = levelled_drill_rotation(
            self._wall_normal, self._wall_up, self._drill_roll_rad
        )
        if levelled is None:
            return float("nan")
        anchored = aligned_drill_rotation(self._wall_normal, self._rotation0, 1.0)
        cos = float(np.clip(anchored[:, 1] @ levelled[:, 1], -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _goal_pose(self) -> pin.SE3:
        """Final commanded pose: the target, fully aligned and corrected."""
        return self._planned_pose(self._trajectory.duration)

    def _start_pose_error(self) -> tuple[float, str]:
        """Worst measured-vs-commanded upper-body joint error (deg) and its label."""
        measured = self._ctx.full_q[self._ctx.upper_body_indices]
        err = np.abs(np.asarray(measured, dtype=np.float64) - self._return_target)
        worst = int(np.argmax(err))
        return float(np.degrees(err[worst])), self._upper_body_labels[worst]

    def _start_pose_reached(self, elapsed: float) -> bool:
        """True once all upper-body joints held within tolerance, or on timeout."""
        now = self._clock()
        worst_deg, worst_label = self._start_pose_error()

        if worst_deg <= self._return_settle_deg:
            if self._return_settled_since is None:
                self._return_settled_since = now
            return now - self._return_settled_since >= self._return_settle_hold_s

        self._return_settled_since = None
        if elapsed - self._return_duration_s >= self._return_settle_timeout_s:
            self._logger.error(
                "Auto-collect: arm never reached the start pose — worst joint "
                f"{worst_label} off by {worst_deg:.2f} deg after "
                f"{self._return_settle_timeout_s:.0f} s (tolerance "
                f"{self._return_settle_deg:.2f} deg); starting the episode anyway. "
                "Raise auto_collect_return_duration_s or the tolerance."
            )
            return True

        self._logger.info(
            f"Auto-collect: waiting for the start pose — {worst_label} off by "
            f"{worst_deg:.2f} deg (tolerance {self._return_settle_deg:.2f} deg).",
            throttle_duration_sec=1.0,
        )
        return False

    def _go_to_default(self) -> None:
        """Ramp the upper body to the (optionally jittered) start pose.

        Advances only once the measured joints have arrived.
        """
        if self._return_start is None:
            self._return_start = self._current_upper_body()
        if self._return_target is None:
            self._return_target = self._sample_start_pose()

        elapsed = self._clock() - self._motion_start
        u = np.clip(elapsed / self._return_duration_s, 0.0, 1.0)
        s = quintic(float(u))
        self._command_upper_body(
            self._return_start + s * (self._return_target - self._return_start)
        )

        if u >= 1.0 and self._start_pose_reached(elapsed):
            worst_deg, worst_label = self._start_pose_error()
            residual = (
                f"arrived in {elapsed:.1f} s, worst joint {worst_label} "
                f"{worst_deg:.2f} deg"
            )
            if self._loop:
                self._logger.info(
                    f"Auto-collect: back at the start pose ({residual}) — "
                    "planning next episode."
                )
                self._enter_phase(_Phase.TRAJECTORY_GENERATION)
            else:
                self._logger.info(
                    f"Auto-collect: back at the start pose ({residual}) — holding."
                )
                self._enter_phase(_Phase.HOLDING)

    # ── Commanding ────────────────────────────────────────────────────────────

    def _command_tip_pose(self, tip_se3: pin.SE3) -> None:
        """Solve right-arm IK for a tip pose and cache the joint target.

        Warm-started from the last command, not the measured (drooping) arm.
        """
        arm_q = self._robot_model.solve_ik(
            self.TIP_FRAME,
            tip_se3,
            self._seed_q(),
            # No neutral retry: it can jump IK basins between consecutive ticks.
            max_iters=20,
            retry_from_neutral=False,
        )
        self._command_upper_body(
            self._robot_model.assemble_upper_body_target(
                self._ctx.default_upper_body_pose,
                # Hold the left arm at its default pose.
                self._robot_model.default_body_pose,
                arm_q,
            )
        )

    def _seed_q(self) -> np.ndarray:
        """Full q to warm-start IK from: the last command, else the measured pose."""
        q = self._ctx.full_q.copy()
        if self._commanded_upper_body is not None:
            q[self._ctx.upper_body_indices] = self._commanded_upper_body
        return q

    def _command_upper_body(self, upper_body_target: np.ndarray) -> None:
        """Cache an upper-body joint target and its commanded tip pose (recorded
        action)."""
        self._commanded_upper_body = np.asarray(upper_body_target, dtype=np.float64)

        q = self._ctx.full_q.copy()
        q[self._ctx.upper_body_indices] = self._commanded_upper_body
        self._commanded_eef_9d = self._robot_model.frame_pose_9d(
            q, self._ctx.right_eef_frame
        )

    def _current_upper_body(self) -> np.ndarray:
        """Seed for a joint-space move: the last command, else the measured pose."""
        if self._commanded_upper_body is not None:
            return self._commanded_upper_body.copy()
        return self._ctx.full_q[self._ctx.upper_body_indices].copy()

    def _target_pose(self) -> pin.SE3 | None:
        """Target-square pose in the pelvis frame (x-axis = wall normal), or None."""
        se3 = lookup_se3(
            self._ctx.tf_buffer, self._ctx.drill_target_frame, self.BASE_FRAME
        )
        if se3 is None:
            self._logger.warn(
                f"Auto-collect: waiting for {self._ctx.drill_target_frame} TF...",
                throttle_duration_sec=2.0,
            )
        return se3

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _log_residual(self, goal_se3: pin.SE3) -> None:
        """Log final tip error split into IK residual, tracking error and total."""
        model = self._robot_model

        q = self._ctx.full_q.copy()
        q[self._ctx.upper_body_indices] = self._commanded_upper_body
        model.cache_forward_kinematics(q, auto_clip=False)
        ik_se3 = model.frame_placement(self.TIP_FRAME)

        model.cache_forward_kinematics(self._ctx.full_q, auto_clip=False)
        achieved_se3 = model.frame_placement(self.TIP_FRAME)

        def _err(a: pin.SE3, b: pin.SE3) -> tuple[float, float]:
            pos, ang = pose_error(a, b)
            return pos * 1e3, np.degrees(ang)

        ik = _err(goal_se3, ik_se3)
        track = _err(ik_se3, achieved_se3)
        total = _err(goal_se3, achieved_se3)
        self._logger.info(
            f"[auto-collect] final tip error: ik={ik[0]:.1f}mm/{ik[1]:.2f}deg | "
            f"track={track[0]:.1f}mm/{track[1]:.2f}deg | "
            f"total={total[0]:.1f}mm/{total[1]:.2f}deg"
        )
