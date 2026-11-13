# MIT License
"""VLA state: play back GR00T action chunks from the coarse role, with metrics.

``vla_perception_only`` runs inference, estimates, overlays and metrics as usual
but commands nothing and holds the entry pose (for detector-only checkpoints or
auditing a perception head).
"""
from __future__ import annotations

import numpy as np
from g1_definitions import definitions

from g1_control.states.base import ControlContext, StateHandler
from g1_control.utils.task_clock import TaskClock


class VlaHandler(StateHandler):
    def __init__(
        self,
        node,
        ctx: ControlContext,
        sources,
        metrics,
        progress_test,
        debug_publisher,
    ) -> None:
        self._ctx = ctx
        self._clock = TaskClock(node)
        self._logger = node.get_logger()
        self._sources = sources
        self._metrics = metrics
        self._progress_test = progress_test
        self._debug = debug_publisher

        # The coarse role drives; the fine role is only queried.
        self._role = sources.coarse

        self._perception_only = bool(
            node.get_parameter("vla_perception_only").value
            if node.has_parameter("vla_perception_only")
            else False
        )
        # Perception-only hold pose, latched on the first tick.
        self._hold_pose: np.ndarray | None = None

        # Advance the chunk pointer at RECORD_FPS.
        self._advance_every: int = max(
            1, round(1.0 / (ctx.control_dt * definitions.RECORD_FPS))
        )
        self._step_counter: int = 0

        # Perception overlays at ~10 Hz when no chunk is played back.
        self._overlay_every: int = max(1, round(0.1 / ctx.control_dt))
        self._overlay_counter: int = 0


    def on_enter(self, previous_state: int) -> None:
        self._step_counter = 0
        # Drop any chunk carried over from the last session.
        self._role.actions.reset()
        self._metrics.set_estimate_source(self._role.client)
        # One CSV per entry.
        self._metrics.start(
            prefix="progress_test" if self._progress_test.enabled else "vla_metrics"
        )
        self._progress_test.reset()
        self._role.client.start_inference_loop()
        if self._perception_only:
            self._logger.warning(
                "[vla] PERCEPTION-ONLY session: the model is being queried and "
                "scored, but no action will be commanded and the arm holds its "
                "entry pose. Unset vla_perception_only to let it drive."
            )

    def on_exit(self) -> None:
        self._ctx.robot_state.upper_body_target_chunk = None
        self._ctx.robot_state.upper_body_target = None
        # Recorder falls back to measured FK while targets are None.
        self._ctx.targets.clear()
        for client in self._sources.unique:
            client.pause_inference_loop()
        self._metrics.finalize()

    # ── Per-tick handler ──────────────────────────────────────────────────────

    def update(self) -> None:
        # Perception-only takes precedence over the sweep.
        if self._perception_only:
            self._update_perception_only()
            return

        if self._progress_test.enabled:
            self._progress_test.update()
            self._publish_perception_debug()
            return

        ctx = self._ctx
        now = self._clock()

        actions = self._role.actions

        is_new_chunk = actions.new_action_chunk.is_set()

        # Double-buffers the chunk and clears the flag.
        actions.step()

        if is_new_chunk:
            upper_body_chunk = actions.local_action_buffer.get("upper_body_joint_chunk")

            if upper_body_chunk is not None:
                ctx.robot_state.upper_body_target_chunk = upper_body_chunk.copy()

            self._publish_chunk_debug()

        right_9d = actions.get_right_eef_9d()

        if right_9d is not None:
            ctx.targets.right_9d = right_9d

        self._step_counter += 1
        if self._step_counter >= self._advance_every:
            self._step_counter = 0
            actions.advance_index()

        # waypoint_idx = -1 marks a VLA session (not a sweep waypoint).
        self._metrics.record(now, waypoint_idx=-1)

    # ── Perception-only session ───────────────────────────────────────────────

    def _update_perception_only(self) -> None:
        """Observation-only tick: hold the arm, zero the twist, record and draw.

        ``ctx.targets`` stays cleared so the recorder logs the measured pose.
        """
        ctx = self._ctx

        if self._hold_pose is None:
            self._hold_pose = np.asarray(
                ctx.full_q[ctx.upper_body_indices], dtype=np.float64
            ).copy()

        ctx.robot_state.twist_command = np.zeros(3)
        ctx.robot_state.upper_body_target_chunk = None
        ctx.robot_state.upper_body_target = self._hold_pose
        ctx.targets.clear()

        self._metrics.record(self._clock(), waypoint_idx=-1)
        self._publish_perception_debug()

    # ── Debug chunk publishing ────────────────────────────────────────────────

    def _publish_chunk_debug(self) -> None:
        """Publish the predicted chunks + target estimates for the debug node."""
        self._debug.publish(
            self._role.actions,
            self._role.client,
            None,
            detection_client=self._sources.fine.client,
        )

    def _publish_perception_debug(self) -> None:
        """Publish detection + target overlays (no chunks), throttled to ~10 Hz."""
        self._overlay_counter += 1
        if self._overlay_counter % self._overlay_every != 0:
            return
        client = self._role.client
        self._debug.publish_perception(
            client.get_tip_target(), None, client.get_last_detection()
        )
