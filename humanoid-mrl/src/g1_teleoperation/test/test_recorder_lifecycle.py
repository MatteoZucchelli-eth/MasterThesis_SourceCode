# MIT License
"""Regression tests for recorder episode acceptance and shutdown behavior."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest

from g1_definitions import definitions
from g1_interfaces.msg import ControlStep
from g1_teleoperation.episode_capture import EpisodeCapture

from g1_teleoperation.recorder_node import (
    _CAMERA_SKEW_S,
    _IMAGE_STALE_S,
    _STALL_REPORT_TICKS,
    CAMERAS,
    RecorderNode,
)


def _frame(*, tip_dist: float = 0.01) -> dict:
    return {
        "observation.state": np.zeros(2, dtype=np.float32),
        "tip_target_dist": tip_dist,
    }


def _recorder(frames: list[dict] | None = None) -> RecorderNode:
    recorder = object.__new__(RecorderNode)
    recorder._writer = Mock(episode_index=7)
    recorder.get_logger = Mock(return_value=Mock())
    recorder._episode = EpisodeCapture(
        frames=list(frames or []),
        image_lags=[0.05] if frames else [],
        recording=True,
        next_frame_stamp=1.0,
    )
    recorder._blocked_ticks = 0
    return recorder


def test_healthy_episode_is_saved():
    recorder = _recorder([_frame(), _frame()])

    assert recorder._stop_episode()
    recorder._writer.save_episode.assert_called_once()
    assert recorder._episode.frames == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recorder: setattr(recorder._episode, "clamped_frames", 1),
        lambda recorder: setattr(
            recorder._episode, "image_lags", [_IMAGE_STALE_S + 0.01]
        ),
        lambda recorder: setattr(recorder._episode, "camera_sync_fault", True),
        lambda recorder: recorder._episode.frames.append(_frame(tip_dist=float("nan"))),
    ],
)
def test_quality_faults_discard_instead_of_save(mutate):
    recorder = _recorder([_frame()])
    mutate(recorder)

    assert not recorder._stop_episode()
    recorder._writer.save_episode.assert_not_called()
    assert recorder._episode.frames == []


def test_empty_episode_is_not_reported_as_saved():
    recorder = _recorder()

    assert not recorder._stop_episode()
    recorder._writer.save_episode.assert_not_called()


def test_shutdown_discards_an_in_progress_episode(monkeypatch):
    recorder = _recorder([_frame()])
    base_destroy = Mock(return_value=True)
    monkeypatch.setattr("rclpy.node.Node.destroy_node", base_destroy)

    recorder.destroy_node()

    recorder._writer.save_episode.assert_not_called()
    assert recorder._episode.frames == []
    base_destroy.assert_called_once_with()


def test_shutdown_discards_a_frozen_unsaved_episode(monkeypatch):
    recorder = _recorder([_frame()])
    recorder._episode.recording = False
    monkeypatch.setattr("rclpy.node.Node.destroy_node", Mock(return_value=True))

    recorder.destroy_node()

    recorder._writer.save_episode.assert_not_called()
    assert recorder._episode.frames == []


PERIOD = 1.0 / definitions.RECORD_FPS


def _loop_recorder(streams: dict[str, list[float]]) -> RecorderNode:
    """A recorder whose cameras have delivered frames at the given stamps."""
    recorder = _recorder()
    recorder._episode.next_frame_stamp = None
    recorder._episode.last_camera_stamps = {cam_key: None for cam_key in CAMERAS}
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder._image_buffers = {
        cam_key: deque(
            ((image, stamp, stamp) for stamp in stamps), maxlen=len(stamps) + 8
        )
        for cam_key, stamps in streams.items()
    }
    # States bracketing every stamp used here; pairing is tested in test_recorder_sync.
    recorder._step_buffer = [(-1.0, -1.0, ControlStep()), (99.0, 99.0, ControlStep())]
    recorder._timebase_diverged = False
    recorder._sync_timebase = None
    recorder._latest_task = "drill in the red square"
    return recorder


def test_a_row_takes_the_frames_nearest_its_instant():
    """A delivery offset between the streams costs nothing.

    Render products are not delivered on the same tick (the wrist can lag the
    head by several periods); each row pairs frames at one shared instant.
    """
    recorder = _loop_recorder(
        {
            "ego_view": [i * PERIOD for i in range(9)],  # delivered to 8 periods
            "wrist_view": [i * PERIOD for i in range(5)],  # four periods behind
        }
    )

    recorder._recording_loop()

    assert len(recorder._episode.frames) == 1
    assert not recorder._episode.camera_sync_fault
    # Both frames come from the newest instant both streams have covered.
    assert recorder._episode.last_camera_stamps == pytest.approx(
        {"ego_view": 4 * PERIOD, "wrist_view": 4 * PERIOD}
    )


def test_a_row_waits_for_the_stream_that_has_not_caught_up():
    recorder = _loop_recorder(
        {
            "ego_view": [i * PERIOD for i in range(9)],
            "wrist_view": [i * PERIOD for i in range(5)],
        }
    )
    recorder._recording_loop()

    # No new wrist frame, so the next instant is covered by one stream only.
    recorder._image_buffers["ego_view"].append((np.zeros((2, 2, 3)), 9 * PERIOD, 0.0))
    recorder._recording_loop()

    assert len(recorder._episode.frames) == 1
    assert not recorder._episode.camera_sync_fault
    assert "wrist_view" in recorder._episode.blocked_reason


def test_a_missed_render_skips_a_row_instead_of_deadlocking():
    """A gap in rendered instants is stepped over rather than stalling the
    episode on a due time nothing advances."""
    recorder = _loop_recorder(
        {
            "ego_view": [i * PERIOD for i in range(5)],
            "wrist_view": [i * PERIOD for i in range(5)],
        }
    )
    recorder._recording_loop()  # row at four periods; the next is due at five

    for cam_key in CAMERAS:  # five periods never rendered
        recorder._image_buffers[cam_key].append((np.zeros((2, 2, 3)), 6 * PERIOD, 0.0))
    recorder._recording_loop()  # skips the row it cannot fill
    recorder._recording_loop()  # and takes the next instant

    assert len(recorder._episode.frames) == 2
    assert recorder._episode.last_camera_stamps == pytest.approx(
        {"ego_view": 6 * PERIOD, "wrist_view": 6 * PERIOD}
    )


def test_frames_from_different_instants_are_skipped_without_poisoning_episode():
    recorder = _loop_recorder({"ego_view": [0.0], "wrist_view": [2 * _CAMERA_SKEW_S]})

    recorder._recording_loop()

    assert recorder._episode.frames == []
    assert not recorder._episode.camera_sync_fault
    assert recorder._episode.worst_camera_skew == 0.0
    assert recorder._episode.next_frame_stamp == pytest.approx(PERIOD)


def test_preexisting_frames_are_excluded_from_a_new_episode():
    recorder = _loop_recorder({"ego_view": [1.0, 1.1], "wrist_view": [0.9, 1.0]})

    recorder._start_episode()

    assert all(not buf for buf in recorder._image_buffers.values())
    assert recorder._episode.last_camera_stamps == {
        "ego_view": 1.1,
        "wrist_view": 1.1,
    }


def test_preexisting_mixed_pair_does_not_latch_camera_skew():
    capture = EpisodeCapture()
    capture.start({"ego_view": 1.0, "wrist_view": 1.0})

    assert not capture.cameras_ready(
        {"ego_view": 1.1, "wrist_view": 0.9}, _CAMERA_SKEW_S
    )
    assert not capture.camera_sync_fault
    assert capture.worst_camera_skew == 0.0


def test_a_silent_camera_topic_is_named():
    recorder = _loop_recorder({"ego_view": [], "wrist_view": [0.0]})

    for _ in range(_STALL_REPORT_TICKS):
        recorder._recording_loop()

    (message,) = recorder.get_logger().warning.call_args.args
    assert CAMERAS["ego_view"] in message


def test_the_discard_message_reports_the_skew_it_measured():
    recorder = _recorder([_frame()])
    recorder._episode.camera_sync_fault = True
    recorder._episode.worst_camera_skew = 10 * _CAMERA_SKEW_S

    assert not recorder._stop_episode()
    (message,) = recorder.get_logger().error.call_args.args
    assert f"{10 * _CAMERA_SKEW_S:.3f}s" in message
