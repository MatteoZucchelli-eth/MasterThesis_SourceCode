# MIT License
"""VLA client: queries a GR00T policy server and post-processes its actions."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from g1_common.data import RobotState
from g1_definitions import definitions
from gr00t.policy.server_client import PolicyClient

from g1_control.utils.obs_ood_scorer import ObsOODScorer
from g1_control.utils.task_clock import TaskClock
from g1_control.utils.wbc_robot_model import RobotModel
from .deadband import chunk_within_deadband

from .observation import ObservationBuilder
from .perception import PerceptionDetection
from .postprocess import ChunkProcessor

from .tip_target import DetectionSlot, TipTargetLifter, TipTargetState

logger = logging.getLogger(__name__)


class VLAClient:
    """Queries a GR00T VLA server at a configured frequency on a background thread.

    The control loop feeds sensor state via the ``set_*`` methods; the background
    loop builds an observation, queries the server, post-processes the chunk
    (IK, optional OOD correction, smoothing) and copies it into ``action_buffer``.

    A target estimate arrives either as a ``tip_target_pred`` pose or as a 2D
    detection back-projected by ``tip_target_lifter``; both surface through the
    same accessors.
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        inference_dt: float,
        action_buffer: dict,
        action_lock: threading.Lock,
        new_action_chunk: threading.Event,
        robot_model: RobotModel,
        default_upper_body_pose: np.ndarray,
        ood_scorer: ObsOODScorer | None = None,
        ood_correction_enabled: bool = False,
        tip_target_lifter: TipTargetLifter | None = None,
        clock: TaskClock | None = None,
        smooth_radius: int = 2,
        action_deadband: float = 0.0,
        action_deadband_rot_deg: float = 1.0,
    ):
        # Replanning period [s].
        self._inference_dt = inference_dt
        # Pacing clock (robot time); None paces on the wall clock.
        self._clock = clock
        self._lifter = tip_target_lifter

        self._obs_builder = ObservationBuilder()
        self._processor = ChunkProcessor(
            robot_model,
            default_upper_body_pose,
            ood_scorer=ood_scorer,
            smooth_radius=smooth_radius,
        )
        self._ood_scorer = ood_scorer
        self._ood_correction_enabled = bool(ood_correction_enabled)

        # Minimum new motion vs. the last published chunk before switching;
        # 0 disables. See `chunk_within_deadband`.
        self._action_deadband = float(action_deadband)
        self._action_deadband_rot_deg = float(action_deadband_rot_deg)
        # The tip chunk currently in the shared buffer — the deadband reference.
        self._published_tip_chunk: np.ndarray | None = None
        self._deadband_held = 0

        # Target pose in the drill-tip frame plus its pelvis-frame anchor.
        self._tip_target = TipTargetState()

        # Every raw detection (before gating) and their count.
        self._detections = DetectionSlot()

        self._state_lock = threading.Lock()
        self._joint_pos: np.ndarray | None = None
        # One rolling frame buffer (and write index) per camera view, keyed by
        # the GR00T video key ("ego_view", "wrist_view").
        self._image_histories: dict[str, np.ndarray] = {}
        self._image_indices: dict[str, int] = {}
        self._video_horizon = 2
        self._latest_right_eef_pose: np.ndarray | None = None
        self._task = "drill in the red square"
        self._modality_config = None

        self._policy = PolicyClient(
            host=server_host,
            port=server_port,
            timeout_ms=5000,
            strict=False,
        )
        logger.info("PolicyClient connecting to %s:%s", server_host, server_port)

        self._running = False
        self._thread: threading.Thread | None = None

        # Shared action chunk written for the control loop.
        self._action_chunk_lock = action_lock
        self._latest_action_chunk = action_buffer
        self._new_action_chunk = new_action_chunk
        # Latches so a missing action head is reported once.
        self._warned_no_action_head = False

        # Pre-correction drill-tip chunk, exposed to the node for RViz
        # visualization. None means the last chunk was not corrected.
        self._vis_lock = threading.Lock()
        self._prev_tip_chunk: np.ndarray | None = None

        logger.info("VLA Client initialized")

    # ── Sensor state setters (called from the control loop) ───────────────────

    def set_robot_state(self, robot_state: RobotState):
        """Update joint positions from the high-frequency control loop."""
        with self._state_lock:
            if robot_state.joint_pos is not None:
                self._joint_pos = np.copy(robot_state.joint_pos)

    def set_image(self, image: np.ndarray, view: str):
        """Update the latest image in the rolling history window of ``view``."""
        with self._state_lock:
            history = self._image_histories.get(view)
            # Lazy init: size the buffer from the first incoming image.
            if history is None:
                h, w, c = image.shape
                history = np.zeros((self._video_horizon, h, w, c), dtype=np.uint8)
                self._image_histories[view] = history
                self._image_indices[view] = 0

            index = self._image_indices[view]
            history[index] = image.copy()
            self._image_indices[view] = (index + 1) % history.shape[0]

    def set_right_eef(self, right_eef_pose: np.ndarray):
        with self._state_lock:
            self._latest_right_eef_pose = right_eef_pose

    def set_task(self, task: str):
        self._task = task

    def get_prev_tip_chunk(self) -> np.ndarray | None:
        """Return the uncorrected drill-tip 9D chunk of the last correction, or
        None if the last chunk was not corrected."""
        with self._vis_lock:
            return self._prev_tip_chunk

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def start_inference_loop(self):
        """Start or resume the background network loop."""
        # Drop estimates from a previous session so stale readings cannot fire a gate.
        self._tip_target.reset()
        self._detections.reset()

        if self._thread is not None and self._thread.is_alive():
            self._running = True  # resume a persistent thread
            return

        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info("VLA Client background thread started.")

    def pause_inference_loop(self):
        """Pause background processing without tearing down the thread."""
        self._running = False
        logger.info("VLA Client background thread paused.")

    def stop(self):
        """Stop the background loop completely."""
        self._running = False
        logger.info("VLA Client thread stopped completely.")

    # ── Inference loop ────────────────────────────────────────────────────────

    def _inference_loop(self):
        """Query the server at ``self._inference_dt`` intervals; errors are logged, not
        fatal."""
        while True:
            if not self._running:
                time.sleep(0.1)
                continue

            try:
                self._run_inference_once()
            except Exception:
                logger.exception("VLA inference iteration failed; continuing")
                time.sleep(self._inference_dt)

    def _pace_next_cycle(self, wall_start: float, cycle_start: float | None) -> None:
        """Wait out the rest of this inference period on the pacing clock."""
        if self._clock is None or cycle_start is None:
            time.sleep(max(0.0, self._inference_dt - (time.time() - wall_start)))
            return
        if not self._clock.sleep_until(cycle_start + self._inference_dt):
            logger.warning(
                "Inference pacing: the simulation clock stopped advancing; "
                "resuming on the wall clock until it comes back."
            )

    def _run_inference_once(self) -> None:
        """Run a single inference + post-processing pass."""
        start_time = time.time()
        cycle_start = None if self._clock is None else self._clock()

        # Modality config first: the snapshot gates on every camera view the
        # server requests, so the required view keys must be known.
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

        if self._ood_scorer is not None:
            flat_state = ObservationBuilder.flat_state(joint_pos, right_eef)
            self._log_ood_scores(self._ood_scorer.score(flat_state))

        try:
            action_dict, info = self._policy.get_action(observation=obs)
            self._update_tip_target(action_dict, info, right_eef)
            post_joint_pos = self._snapshot_proprioception()
            action_dict = self._processor.process_ik(action_dict, post_joint_pos)
        except Exception as e:
            logger.error("VLA inference error: %s", e)
            # Back off so an unresponsive server is not flooded.
            time.sleep(self._inference_dt)
            return

        if self._ood_scorer is not None and self._ood_correction_enabled:
            action_dict, prev_tip_chunk = self._processor.correct_ood(
                action_dict, post_joint_pos
            )
            with self._vis_lock:
                self._prev_tip_chunk = prev_tip_chunk

        action_dict = self._processor.smooth(action_dict, post_joint_pos)
        self.copy_action_chunk(action_dict)

        self._pace_next_cycle(start_time, cycle_start)

    def _snapshot_sensor_state(self):
        """Return a consistent (joint_pos, image_seqs, right) snapshot.

        Returns ``None`` until joints, every camera view requested by the server
        and the drill-tip pose are available. Each view's frames are reordered
        chronologically (oldest first).
        """
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
                chronological = [(start + i) % horizon for i in range(horizon)]
                image_seqs[view] = history[chronological].copy()

            return (
                self._joint_pos.copy(),
                image_seqs,
                self._latest_right_eef_pose.copy(),
            )

    def _snapshot_proprioception(self) -> np.ndarray:
        with self._state_lock:
            return self._joint_pos.copy()

    def _ensure_modality_config(self) -> bool:
        """Fetch the modality config from the server once. True once ready."""
        if self._modality_config is not None:
            return True
        try:
            mc = self._policy.get_modality_config()
            new_horizon = self._obs_builder.update_modality(mc)

            with self._state_lock:
                if new_horizon != self._video_horizon:
                    # set_image re-allocates at the new horizon.
                    self._image_histories.clear()
                    self._image_indices.clear()
                self._video_horizon = new_horizon

            self._modality_config = mc
            return True
        except Exception as e:
            logger.warning("Waiting for VLA server modality config: %s", e)
            return False

    def copy_action_chunk(self, action_dict):
        """Move output commands from the ZMQ thread to the control buffer.

        Signals a fresh chunk only if the reply carried an action head; a
        perception-only reply would otherwise publish a zero (invalid) 9D pose.
        """
        with self._action_chunk_lock:
            if self._latest_action_chunk is None:
                return

            tip_chunk = action_dict.get(definitions.RIGHT_EEF_KEY)
            if tip_chunk is not None:
                tip_chunk = np.asarray(tip_chunk)
                if tip_chunk.ndim > 2:
                    tip_chunk = tip_chunk.squeeze(0)
            if tip_chunk is not None and chunk_within_deadband(
                tip_chunk,
                self._published_tip_chunk,
                self._action_deadband,
                self._action_deadband_rot_deg,
            ):
                # Keep the current chunk; a sustained hold parks the arm at its
                # last waypoint.
                self._deadband_held += 1
                if self._deadband_held % 10 == 1:
                    logger.debug(
                        "Action deadband held the chunk (%d in a row, <%.1f mm / "
                        "%.2f deg of new motion)",
                        self._deadband_held,
                        self._action_deadband * 1000.0,
                        self._action_deadband_rot_deg,
                    )
                return

            copied = 0
            for key, array_value in action_dict.items():
                if key in self._latest_action_chunk:
                    # Strip batch dimensions if the server prepends them.
                    if array_value.ndim > self._latest_action_chunk[key].ndim:
                        array_value = array_value.squeeze(0)

                    np.copyto(self._latest_action_chunk[key], array_value)
                    copied += 1

            if copied == 0:
                self._warn_no_action_head(action_dict)
                return

            if tip_chunk is not None:
                self._published_tip_chunk = tip_chunk.copy()
            self._deadband_held = 0
            self._new_action_chunk.set()

    def _warn_no_action_head(self, action_dict) -> None:
        """Warn once that the server replies without an action chunk."""
        if self._warned_no_action_head:
            return
        self._warned_no_action_head = True
        logger.error(
            "VLA server reply carries no action chunk — got keys %s, expected one "
            "of %s. A perception-only checkpoint is being served where an action "
            "model is required; the arm will not be driven.",
            sorted(action_dict),
            sorted(self._latest_action_chunk),
        )

    def _update_tip_target(
        self, action_dict: dict | None, info: dict | None, anchor_tip_9d: np.ndarray
    ) -> None:
        """Record the tip->target estimate the model reported with its actions.

        Accepts either:

        * ``tip_target_pred`` — a 9D (xyz + rot6d) pose in the drill-tip frame.
        * a 2D perception head — back-projected by the lifter; weak detections
          are dropped.

        ``anchor_tip_9d`` is the pelvis-frame tip pose from the snapshot this
        inference consumed.
        """
        if isinstance(info, dict) and "tip_target_pred" in info:
            pose = np.asarray(info["tip_target_pred"], dtype=float).reshape(-1)[:9]
            self._tip_target.publish(pose, anchor_tip_9d)
            logger.debug(
                "[VLA] predicted tip->target error: x=%.1f mm yz=%.1f mm",
                abs(pose[0]) * 1000.0,
                float(np.linalg.norm(pose[1:3])) * 1000.0,
            )
            return

        if self._lifter is None:
            return
        detection = PerceptionDetection.from_reply(action_dict, info)
        if detection is None:
            return
        self._detections.publish(detection)
        pred_tip_9d = self._lifter.lift(detection)
        if pred_tip_9d is not None:
            self._tip_target.publish(pred_tip_9d, anchor_tip_9d)

    def get_tip_target(self) -> np.ndarray | None:
        """Return the latest tip->target pose prediction (9D xyz+rot6d, in the
        drill-tip frame), or None if the current VLA session has not produced an
        inference yet."""
        return self._tip_target.latest()

    def get_tip_target_anchored(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """Return ``(pred_9d, anchor_tip_9d, stamp, seq)`` for the latest
        prediction — the tip-frame prediction together with the pelvis-frame
        tip pose fed to that same inference — or None if the current session
        has not produced one. ``seq`` increments per prediction so consumers
        can ignore repeats."""
        return self._tip_target.anchored()

    def get_last_detection(self) -> PerceptionDetection | None:
        """Newest raw detection (before gating), for the 2D overlay.

        None for a ``tip_target_pred`` model or before the first inference.
        """
        return self._detections.latest()

    @property
    def detection_count(self) -> int:
        """Detections received this session, before any gating."""
        return self._detections.count

    def get_visibility(self) -> float:
        """Newest detection confidence (0..1), regardless of the gate; 0 if the
        model has no perception head."""
        detection = self.get_last_detection()
        return 0.0 if detection is None else detection.visibility

    def is_visible(self) -> bool:
        """Whether the newest detection cleared the model's ``visible`` flag."""
        detection = self.get_last_detection()
        return False if detection is None else detection.visible

    def _log_ood_scores(self, scores: dict) -> None:
        logger.info(
            "[OOD] energy=%.4f mean_knn_dist=%.4f",
            scores["energy"],
            scores["mean_knn_dist"],
        )
