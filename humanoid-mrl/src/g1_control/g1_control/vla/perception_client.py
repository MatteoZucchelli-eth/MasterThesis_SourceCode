# MIT License
"""Perception client: queries the wrist-camera target detector and lifts it to 3D.

Builds the GR00T observation with
:class:`~g1_control.vla.observation.ObservationBuilder`
and passes each detection to a :class:`~g1_control.vla.tip_target.TipTargetLifter`
(gating + back-projection into the drill-tip frame). Exposes the same tip-target
accessors as :class:`~g1_control.vla.client.VLAClient`.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from g1_common.data import RobotState
from gr00t.policy.server_client import PolicyClient

from g1_control.utils.task_clock import TaskClock

from .observation import ObservationBuilder
from .perception import PerceptionDetection

from .tip_target import DetectionSlot, TipTargetLifter, TipTargetState

logger = logging.getLogger(__name__)


class PerceptionClient:
    """Queries the perception server and publishes tip-frame target estimates."""

    def __init__(
        self,
        server_host: str,
        server_port: int,
        inference_dt: float,
        tip_target_lifter: TipTargetLifter,
        clock: TaskClock | None = None,
    ) -> None:
        self._inference_dt = inference_dt
        self._clock = clock
        self._lifter = tip_target_lifter

        self._obs_builder = ObservationBuilder()

        # Tip-frame estimate (same holder as the action client).
        self._tip_target = TipTargetState()

        # Every raw detection, gated or not, plus a received count.
        self._detections = DetectionSlot()

        # Mutable sensor state fed by the control loop.
        self._state_lock = threading.Lock()
        self._joint_pos: np.ndarray | None = None
        self._image_histories: dict[str, np.ndarray] = {}
        self._image_indices: dict[str, int] = {}
        self._video_horizon = 1
        self._latest_right_eef_pose: np.ndarray | None = None
        self._task = "drill in the red square"
        self._modality_config = None

        self._policy = PolicyClient(
            host=server_host, port=server_port, timeout_ms=5000, strict=False
        )
        logger.info("PerceptionClient connecting to %s:%s", server_host, server_port)

        self._running = False
        self._thread: threading.Thread | None = None
        logger.info("Perception Client initialized")

    # ── Sensor state setters (called from the control loop) ───────────────────

    def set_robot_state(self, robot_state: RobotState) -> None:
        with self._state_lock:
            if robot_state.joint_pos is not None:
                self._joint_pos = np.copy(robot_state.joint_pos)

    def set_image(self, image: np.ndarray, view: str) -> None:
        with self._state_lock:
            history = self._image_histories.get(view)
            if history is None:
                h, w, c = image.shape
                history = np.zeros((self._video_horizon, h, w, c), dtype=np.uint8)
                self._image_histories[view] = history
                self._image_indices[view] = 0
            index = self._image_indices[view]
            history[index] = image.copy()
            self._image_indices[view] = (index + 1) % history.shape[0]

    def set_right_eef(self, right_eef_pose: np.ndarray) -> None:
        with self._state_lock:
            self._latest_right_eef_pose = right_eef_pose

    def set_task(self, task: str) -> None:
        self._task = task

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def start_inference_loop(self) -> None:
        """Start or resume the background network loop, dropping stale state."""
        self._tip_target.reset()
        self._detections.reset()

        if self._thread is not None and self._thread.is_alive():
            self._running = True
            return
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info("Perception Client background thread started.")

    def pause_inference_loop(self) -> None:
        self._running = False
        logger.info("Perception Client background thread paused.")

    def stop(self) -> None:
        self._running = False
        logger.info("Perception Client thread stopped completely.")

    # ── Inference loop ────────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        while True:
            if not self._running:
                time.sleep(0.1)
                continue
            try:
                self._run_inference_once()
            except Exception:
                logger.exception("Perception inference iteration failed; continuing")
                time.sleep(self._inference_dt)

    def _run_inference_once(self) -> None:
        start_time = time.time()
        cycle_start = None if self._clock is None else self._clock()

        if not self._ensure_modality_config():
            time.sleep(0.5)
            return

        snapshot = self._snapshot_sensor_state()
        if snapshot is None:
            logger.info("Waiting for state/image data from ROS node...")
            time.sleep(0.5)
            return

        joint_pos, image_seqs, right_eef = snapshot
        obs = self._obs_builder.build(joint_pos, image_seqs, right_eef, self._task)

        try:
            action_dict, info = self._policy.get_action(observation=obs)
        except Exception as e:
            logger.error("Perception inference error: %s", e)
            time.sleep(self._inference_dt)
            return

        self._update_target(action_dict, info, right_eef)

        # Pace on the robot clock, independent of machine load.
        if self._clock is None or cycle_start is None:
            time.sleep(max(0.0, self._inference_dt - (time.time() - start_time)))
        elif not self._clock.sleep_until(cycle_start + self._inference_dt):
            logger.warning(
                "Perception pacing: the simulation clock stopped advancing; "
                "resuming on the wall clock until it comes back."
            )

    def _update_target(
        self, action_dict: dict, info: dict | None, anchor_tip_9d: np.ndarray
    ) -> None:
        """Parse the detection (from the action dict or info) and lift it to the tip
        frame."""
        detection = PerceptionDetection.from_reply(action_dict, info)
        if detection is None:
            logger.warning(
                "[perception] server reply carries no perception head "
                "(action keys: %s); is this the right model?",
                sorted(action_dict) if isinstance(action_dict, dict) else action_dict,
            )
            return
        self._detections.publish(detection)

        pred_tip_9d = self._lifter.lift(detection)
        if pred_tip_9d is not None:
            self._tip_target.publish(pred_tip_9d, anchor_tip_9d)

    def _snapshot_sensor_state(self):
        required_views = self._obs_builder.video_keys or []
        with self._state_lock:
            if (
                self._joint_pos is None
                or any(v not in self._image_histories for v in required_views)
                or self._latest_right_eef_pose is None
            ):
                return None

            image_seqs = {}
            for view in required_views:
                history = self._image_histories[view]
                horizon = history.shape[0]
                start = self._image_indices[view]
                order = [(start + i) % horizon for i in range(horizon)]
                image_seqs[view] = history[order].copy()

            return (
                self._joint_pos.copy(),
                image_seqs,
                self._latest_right_eef_pose.copy(),
            )

    def _ensure_modality_config(self) -> bool:
        if self._modality_config is not None:
            return True
        try:
            mc = self._policy.get_modality_config()
            new_horizon = self._obs_builder.update_modality(mc)
            with self._state_lock:
                if new_horizon != self._video_horizon:
                    self._image_histories.clear()
                    self._image_indices.clear()
                self._video_horizon = new_horizon
            self._modality_config = mc
            return True
        except Exception as e:
            logger.warning("Waiting for perception server modality config: %s", e)
            return False

    # ── Accessors (same interface as VLAClient's tip-target head) ─────────────

    def get_tip_target(self) -> np.ndarray | None:
        """Latest tip-frame target pose (9D), or None until the first gated-in
        detection of the current session."""
        return self._tip_target.latest()

    def get_tip_target_anchored(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """``(pred_tip_9d, anchor_tip_9d, stamp, seq)`` for the latest gated-in
        detection, or None. ``seq`` increments per detection so consumers can
        ignore repeats."""
        return self._tip_target.anchored()

    def get_last_detection(self) -> PerceptionDetection | None:
        """Newest raw detection (before gating), for the 2D overlay; None until
        the first inference of the current session."""
        return self._detections.latest()

    @property
    def detection_count(self) -> int:
        """Detections received this session, before any gating."""
        return self._detections.count

    def get_visibility(self) -> float:
        """Newest detection confidence (0..1), regardless of the gate."""
        detection = self.get_last_detection()
        return 0.0 if detection is None else detection.visibility

    def is_visible(self) -> bool:
        """Whether the newest detection cleared the model's ``visible`` flag."""
        detection = self.get_last_detection()
        return False if detection is None else detection.visible
