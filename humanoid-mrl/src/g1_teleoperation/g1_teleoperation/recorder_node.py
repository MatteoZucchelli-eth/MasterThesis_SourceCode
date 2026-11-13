# MIT License
"""Synchronize ROS camera frames with robot state and persist valid episodes.

The node is a transport adapter: :class:`EpisodeCapture` owns the transaction,
and :class:`LeRobotDatasetWriter` owns disk I/O. State is interpolated at the
wrist image stamp in simulation; unrelated hardware clocks use arrival time.
"""
import bisect
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from g1_common.camera import preparer_from_camera_info
from g1_definitions import definitions
from g1_interfaces.msg import ControlStep
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String, UInt8

from g1_teleoperation.episode_capture import EpisodeCapture
from g1_teleoperation.lerobot_writer import LeRobotDatasetWriter

# GR00T video modality key → ROS image topic. Each entry is recorded into its
# own ``observation.images.<key>`` video stream and declared in modality.json.
CAMERAS: dict[str, str] = {
    "ego_view": definitions.TOPIC_CAMERA,  # head (torso d435)
    "wrist_view": definitions.TOPIC_CAMERA_WRIST,  # wrist (drill hand_cam)
}

# CameraInfo per camera; frames are squared and rectified against it, as at deploy.
CAMERA_INFOS: dict[str, str] = {
    "ego_view": definitions.TOPIC_CAMERA_INFO,
    "wrist_view": definitions.TOPIC_CAMERA_WRIST_INFO,
}

SYNC_CAMERA: str = "wrist_view"
"""Camera whose header stamp the state is synchronized to (2D labels live in it)."""

_CLOCK_MISMATCH_S: float = 3600.0
"""Gap that distinguishes firmware-vs-epoch clocks from camera backlog."""

_IMAGE_STALE_S: float = 0.5
"""Maximum healthy image lag; normal measured render lag is about 0.054 s."""

_CAMERA_SKEW_PERIODS: float = 1.5
"""Cross-camera capture-stamp separation allowed in one row, in camera periods.

Rows pick each camera's frame nearest the row instant, so this bounds capture
time, not delivery lag. The half period keeps it off the whole-period grid
that stamp offsets land on."""

_CAMERA_SKEW_S: float = _CAMERA_SKEW_PERIODS / definitions.RECORD_FPS
"""Maximum timestamp separation between camera frames in one dataset row."""

_CAMERA_BUFFER_FRAMES: int = 16
"""Recent frames kept per camera for a row to select from.

Must span the worst delivery offset between streams (~4 periods, grows with load)."""

_STALL_REPORT_TICKS: int = 2 * definitions.RECORD_FPS
"""Blocked sampling ticks (~1 s at 2x FPS polling) before a stall is reported."""


def clocks_agree(ref_stamp: float, newest_tick: float) -> bool:
    """Whether an image stamp and the newest ControlStep tick share a timebase."""
    return abs(ref_stamp - newest_tick) < _CLOCK_MISMATCH_S


def image_is_stale(ref_stamp: float, newest_tick: float) -> bool:
    """Whether an image trails the newest state by more than one render lag.

    Only meaningful once :func:`clocks_agree`; a foreign clock makes the
    difference meaningless rather than large.
    """
    return newest_tick - ref_stamp > _IMAGE_STALE_S


def bracketing_weights(times: list[float], ref: float) -> tuple[int, int, float]:
    """Indices (i0, i1) and weight w so times[i0]..times[i1] straddle ref.

    ``ref`` outside the buffered range clamps to the nearest end (w 0 or 1);
    the caller decides whether the clamped gap is worth a warning.
    """
    i = bisect.bisect_left(times, ref)
    if i <= 0:
        return 0, 0, 0.0
    if i >= len(times):
        return len(times) - 1, len(times) - 1, 0.0
    dt = times[i] - times[i - 1]
    w = (ref - times[i - 1]) / dt if dt > 0 else 0.0
    return i - 1, i, w


def interpolate_steps(msg0: ControlStep, msg1: ControlStep, w: float) -> dict:
    """State/action/label fields linearly interpolated between two steps.

    Lerp is adequate for rot6d at adjacent ticks (consumers re-orthonormalize).
    The target label is NaN unless both endpoints carry a valid one.
    """
    w = min(max(w, 0.0), 1.0)

    def lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((1.0 - w) * a + w * b).astype(np.float32)

    state = lerp(_extract_state_vector(msg0), _extract_state_vector(msg1))
    action = lerp(_extract_action_vector(msg0), _extract_action_vector(msg1))
    if msg0.tip_target_valid and msg1.tip_target_valid:
        tip_target_9d = lerp(
            np.array(msg0.tip_target_9d, dtype=np.float32),
            np.array(msg1.tip_target_9d, dtype=np.float32),
        )
        tip_target_dist = float(
            (1.0 - w) * msg0.tip_target_dist + w * msg1.tip_target_dist
        )
    else:
        tip_target_9d = np.full(9, np.nan, dtype=np.float32)
        tip_target_dist = float("nan")
    return {
        "observation.state": state,
        "action": action,
        "tip_target_9d": tip_target_9d,
        "tip_target_dist": tip_target_dist,
    }


def _extract_state_vector(msg: ControlStep) -> np.ndarray:
    """Pack one LowState into the state layout of ``definitions.STATE_SLICES``.

    Right arm only: drill tip pose and joint positions (PR order).
    """
    motors = msg.low_state.motor_state[: definitions.NUM_MOTOR_G1_BASE]
    start, end = definitions.JOINT_PR_SLICES["right_arm"]
    right_arm_motors = motors[start:end]

    # Right EEF = drill tip, same frame as the action.
    return np.concatenate(
        [
            np.array(msg.right_eef_pose, dtype=np.float32),
            np.array([m.q for m in right_arm_motors], dtype=np.float32),
        ]
    )


def _extract_action_vector(msg: ControlStep) -> np.ndarray:
    return np.array(msg.right_eef_target, dtype=np.float32)


class RecorderNode(Node):
    _FPS: int = definitions.RECORD_FPS

    def __init__(self):
        super().__init__("recorder_node")

        self.declare_parameter("dataset_path", "/workspace/g1_dataset")
        self.declare_parameter("task_description", "drill in the red square")

        self._writer = LeRobotDatasetWriter(
            Path(self.get_parameter("dataset_path").value),
            fps=self._FPS,
            camera_keys=list(CAMERAS),
        )
        self._default_task: str = self.get_parameter("task_description").value

        # ── Buffers ────────────────────────────────────────────────────────
        # Per camera: (frame, header_stamp, arrival). States: (tick_s, arrival_s, msg),
        # 1024 entries (>2 s at 500 Hz), so each image pairs with the state at its
        # stamp.
        self._image_buffers: dict[str, deque[tuple[np.ndarray, float, float]]] = {
            cam_key: deque(maxlen=_CAMERA_BUFFER_FRAMES) for cam_key in CAMERAS
        }
        # Per-camera frame preparers, built from the first CameraInfo.
        self._preparers: dict[str, object | None] = {
            cam_key: None for cam_key in CAMERAS
        }
        self._step_buffer: deque[tuple[float, float, ControlStep]] = deque(maxlen=1024)
        self._sync_timebase: str | None = None  # logged when chosen / changed
        # Latched once tick and image stamps disagree (switch to arrival pairing).
        self._timebase_diverged: bool = False
        self._latest_task: str = self._default_task

        self._episode = EpisodeCapture()
        # Consecutive sampling ticks without a row (see _report_stall).
        self._blocked_ticks: int = 0

        self._bridge = CvBridge()
        self._setup_ros_interfaces()

        self.get_logger().info(
            f"RecorderNode ready. Dataset: {self.get_parameter('dataset_path').value}"
        )
        self.get_logger().info(
            f"Resuming from episode {self._writer.episode_index}, "
            f"global frame {self._writer.global_frame_index}"
        )
        self.get_logger().info(
            "Publish UInt8 to /record_episode: 1 start, 2 freeze, 0 save, 3 discard."
        )

    # ── ROS interfaces ─────────────────────────────────────────────────────

    def _setup_ros_interfaces(self) -> None:
        """Create subscribers and the write timer."""
        self._sync_sub = self.create_subscription(
            ControlStep,
            definitions.TOPIC_RECORDED_STEP,
            self._control_step_callback,
            10,
        )

        # Latest frame only, matching the publisher; the node buffers history itself.
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._camera_subs = {
            cam_key: self.create_subscription(
                Image, topic, self._make_image_callback(cam_key), cam_qos
            )
            for cam_key, topic in CAMERAS.items()
        }
        self._camera_info_subs = {
            cam_key: self.create_subscription(
                CameraInfo, topic, self._make_camera_info_callback(cam_key), cam_qos
            )
            for cam_key, topic in CAMERA_INFOS.items()
        }
        self._task_sub = self.create_subscription(
            String,
            definitions.TOPIC_TASKINSTRUCTION,
            self._task_instruction_callback,
            1,
        )
        self._record_sub = self.create_subscription(
            UInt8, "record_episode", self._record_control_callback, 1
        )

        # Poll at 2x FPS; row cadence is set on the image clock in _recording_loop.
        self._record_timer = self.create_timer(
            1.0 / (2 * self._FPS), self._recording_loop
        )

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _make_camera_info_callback(self, cam_key: str):
        """Return a CameraInfo callback that resolves that camera's preparer.

        Same construction as controller_manager_node, so recorded frames match
        deploy-time frames.
        """

        def _cb(msg: CameraInfo) -> None:
            if self._preparers.get(cam_key) is not None:
                return
            side = min(int(msg.width), int(msg.height))
            try:
                preparer = preparer_from_camera_info(msg, side)
            except ValueError as exc:
                # Unsupported lens: stay without a preparer (frames dropped).
                self.get_logger().error(
                    f"{cam_key}: cannot prepare frames from "
                    f"{CAMERA_INFOS[cam_key]}: {exc}",
                    throttle_duration_sec=10.0,
                )
                return
            self._preparers[cam_key] = preparer
            self.get_logger().info(
                f"{cam_key}: {msg.width}x{msg.height} -> {side}x{side} via "
                f"{type(preparer).__name__}"
            )

        return _cb

    def _make_image_callback(self, cam_key: str):
        """Return an Image callback that stores the latest frame + its stamp."""

        def _cb(msg: Image) -> None:
            try:
                preparer = self._preparers.get(cam_key)
                if preparer is None:
                    self.get_logger().warning(
                        f"{cam_key} frame dropped: no CameraInfo on "
                        f"{CAMERA_INFOS[cam_key]} yet",
                        throttle_duration_sec=5.0,
                    )
                    return
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                arrival = self.get_clock().now().nanoseconds * 1e-9
                self._image_buffers[cam_key].append(
                    (
                        preparer(self._bridge.imgmsg_to_cv2(msg, "rgb8")),
                        stamp,
                        arrival,
                    )
                )
            except Exception as exc:
                self.get_logger().error(f"Image conversion error ({cam_key}): {exc}")

        return _cb

    def _control_step_callback(self, msg: ControlStep) -> None:
        arrival = self.get_clock().now().nanoseconds * 1e-9
        self._step_buffer.append((msg.low_state.tick * 1e-3, arrival, msg))

    def _task_instruction_callback(self, msg: String) -> None:
        self._latest_task = msg.data

    def _record_control_callback(self, msg: UInt8) -> None:
        if msg.data == 1 and not self._episode.recording:
            self._start_episode()
        elif msg.data == 2 and self._episode.recording:
            self._episode.freeze()
            self.get_logger().info(
                f"Episode {self._writer.episode_index} frozen. "
                "Awaiting save/discard..."
            )
        elif msg.data == 0:
            if self._stop_episode():
                self.get_logger().info("Saved episode successfully.")
        elif msg.data == 3:
            self._discard_episode()
            self.get_logger().warn(
                f"Discarded episode {self._writer.episode_index} data."
            )

    def _recording_loop(self) -> None:
        """Emit one row per 1/FPS of image-clock time, with its interpolated state.

        Cadence follows camera stamps, not wall time, so row timestamps stay
        valid at any real-time factor. Each row takes every camera's frame
        nearest the row instant from the per-camera history.
        """
        if not self._episode.recording:
            return

        empty = [CAMERAS[key] for key, buf in self._image_buffers.items() if not buf]
        if not self._step_buffer or empty:
            # Wait until state and every camera stream are cached.
            self._report_stall(
                f"no state on {definitions.TOPIC_RECORDED_STEP} yet"
                if not self._step_buffer
                else f"no frame yet on {', '.join(empty)}"
            )
            return

        period = 1.0 / self._FPS
        # Instant every camera has rendered past; rows wait for the slowest stream.
        covered = min(buf[-1][1] for buf in self._image_buffers.values())
        if not self._episode.frame_due(covered, period):
            behind = min(
                self._image_buffers, key=lambda key: self._image_buffers[key][-1][1]
            )
            self._report_stall(f"waiting for {behind} to reach the next row's instant")
            return

        # Each camera's frame nearest the row's instant.
        due = self._episode.next_frame_stamp
        chosen = {
            key: min(buf, key=lambda entry: abs(entry[1] - due))
            for key, buf in self._image_buffers.items()
        }
        camera_stamps = {key: entry[1] for key, entry in chosen.items()}
        ref_stamp = camera_stamps[SYNC_CAMERA]
        max_camera_skew = max(camera_stamps.values()) - min(camera_stamps.values())
        # Stale frames (pre-start or a missed render) are not a sync fault: skip the
        # row.
        if not self._episode.cameras_fresh(camera_stamps):
            self._episode.skip_row(period)
            self._report_stall(
                "a camera has not produced a fresh frame for this row; the row was skipped"
            )
            return
        if not self._episode.cameras_ready(camera_stamps, _CAMERA_SKEW_S):
            if max_camera_skew > _CAMERA_SKEW_S:
                behind = min(camera_stamps, key=camera_stamps.get)
                self.get_logger().warning(
                    f"The frames nearest this row's instant are {max_camera_skew:.3f}s "
                    f"apart (limit {_CAMERA_SKEW_S:.3f}s), {behind} the older — this "
                    "row will be skipped. Selection already removes a delivery offset, "
                    "so these frames were captured at different instants.",
                    throttle_duration_sec=5.0,
                )
                # Advance past the rejected instant so skew does not accumulate.
                self._episode.skip_row(period)
                self._report_stall("camera frames for the row exceeded the skew limit")
                return
            # Missed render: frames already used by the previous row; skip ahead.
            self._episode.skip_row(period)
            self._report_stall("a camera missed a render; the row was skipped")
            return

        steps = list(self._step_buffer)
        # In sim, tick and image stamps share sim time; on the robot tick is a
        # firmware counter, so fall back (latched) to arrival-time pairing.
        gap = abs(ref_stamp - steps[-1][0])
        if not self._timebase_diverged and not clocks_agree(ref_stamp, steps[-1][0]):
            self._timebase_diverged = True
            self.get_logger().error(
                f"low_state.tick and the {SYNC_CAMERA} stamp differ by {gap:.0f}s "
                f"(limit {_CLOCK_MISMATCH_S}s) — they are not the same clock. "
                "Falling back to arrival-time pairing for the rest of the session."
                + (
                    f" Episode {self._writer.episode_index} is mid-recording and "
                    "straddles both pairings — discard it."
                    if self._episode.recording
                    else ""
                )
            )
        use_tick = not self._timebase_diverged
        if use_tick and image_is_stale(ref_stamp, steps[-1][0]):
            # Same clock, lagging stream: pairing stays correct; health report only.
            self.get_logger().error(
                f"The {SYNC_CAMERA} stream is {gap:.2f}s behind the state stream "
                "(healthy is ~0.05s). Pairing still uses the image stamp, so "
                "states match their frames, but the episode's video trails what "
                "its timestamps claim. Check camera bandwidth / machine load.",
                throttle_duration_sec=5.0,
            )
        timebase = "low_state.tick (sim time)" if use_tick else "arrival time"
        if timebase != self._sync_timebase:
            self._sync_timebase = timebase
            self.get_logger().info(f"Image<->state pairing timebase: {timebase}")

        times = [s[0 if use_tick else 1] for s in steps]
        # Arrival fallback: image arrival time proxies capture time.
        ref = ref_stamp if use_tick else chosen[SYNC_CAMERA][2]
        i0, i1, w = bracketing_weights(times, ref)
        if i0 == i1 and abs(times[i0] - ref) > 0.1:
            # Unbracketed image: counted and reported at save.
            self._episode.clamped_frames += 1
            self.get_logger().warning(
                f"Image stamp {ref:.3f} outside the state buffer "
                f"[{times[0]:.3f}, {times[-1]:.3f}] — clamping to the nearest "
                "state; check that the controller is publishing.",
                throttle_duration_sec=5.0,
            )

        # Target label is NaN whenever the controller could not resolve the target.
        fields = interpolate_steps(steps[i0][2], steps[i1][2], w)
        fields["frames"] = {
            cam_key: entry[0].copy() for cam_key, entry in chosen.items()
        }
        fields["task"] = self._latest_task
        self._episode.append(
            fields,
            lag_s=times[-1] - ref,
            camera_stamps=camera_stamps,
            reference_stamp=covered,
            period_s=period,
        )
        self._blocked_ticks = 0

    # ── Episode management ─────────────────────────────────────────────────

    def _report_stall(self, reason: str) -> None:
        """Warn, with ``reason``, when a recording episode goes ~1 s without a row."""
        self._episode.blocked_reason = reason
        self._blocked_ticks += 1
        if self._blocked_ticks < _STALL_REPORT_TICKS:
            return
        self.get_logger().warning(
            f"Episode {self._writer.episode_index} is recording but capturing "
            f"nothing: {reason}.",
            throttle_duration_sec=5.0,
        )

    def _start_episode(self) -> None:
        self._blocked_ticks = 0
        # Drop pre-start frames; all cameras must cross a common boundary stamp
        # (the newest cached one) before rows are emitted.
        cached_stamps = [buf[-1][1] for buf in self._image_buffers.values() if buf]
        boundary = max(cached_stamps) if cached_stamps else None
        for buf in self._image_buffers.values():
            buf.clear()
        self._episode.start({cam_key: boundary for cam_key in CAMERAS})
        self.get_logger().info(
            f"Started recording episode {self._writer.episode_index}"
        )

    def _discard_episode(self) -> None:
        """Drop all per-episode state without writing anything."""
        self._episode.discard()

    def _stop_episode(self) -> bool:
        """Save a healthy episode, otherwise discard it. Return whether saved."""
        self._episode.freeze()
        failures = self._episode.failures(stale_s=_IMAGE_STALE_S, skew_s=_CAMERA_SKEW_S)
        if failures:
            self.get_logger().error(
                f"Episode {self._writer.episode_index} failed automatic quality "
                f"checks and will be discarded: {'; '.join(failures)}."
            )
            self._discard_episode()
            return False

        frames = self._episode.frames
        n = len(frames)
        repeats = sum(
            1
            for a, b in zip(frames, frames[1:])
            if np.array_equal(a["observation.state"], b["observation.state"])
        )
        if repeats > 0.02 * n:
            self.get_logger().warning(
                f"Episode {self._writer.episode_index}: {repeats}/{n} frames "
                f"({repeats / n:.0%}) repeat the previous state."
            )
        # Per-episode worst skew and lag, to spot drift toward the limits.
        self.get_logger().info(
            f"Episode {self._writer.episode_index}: {n} frames, worst "
            f"cross-camera skew {self._episode.worst_camera_skew:.3f}s, worst "
            f"image lag {max(self._episode.image_lags):.3f}s."
        )
        self._writer.save_episode(frames)
        self._episode.discard()
        return True

    # ── Shutdown ───────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self._episode.recording or self._episode.frames:
            self.get_logger().info("Shutting down – discarding in-progress episode.")
            self._discard_episode()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
