# MIT License
"""Unit tests for the VLA action-chunk manager session lifecycle."""

import numpy as np

from g1_control.vla_action_manager import VLAActionManager
from g1_definitions import definitions


def _publish_chunk(manager: VLAActionManager, value: float) -> None:
    """Simulate the VLA client delivering a chunk filled with ``value``."""
    with manager.action_lock:
        for arr in manager.action_buffer.values():
            arr.fill(value)
    manager.new_action_chunk.set()


def test_reset_drops_previous_session_chunk():
    """Re-entering VLA must not replay the previous session's waypoints."""
    m = VLAActionManager(chunk_size=4, num_upper_body_joints=3)
    _publish_chunk(m, 7.0)
    m.step()
    for _ in range(10):
        m.advance_index()  # session ends parked on the last waypoint
    assert m.has_first_chunk
    assert m.get_right_eef_9d() is not None

    # New session: without a fresh chunk there must be nothing to play back.
    m.reset()
    assert not m.has_first_chunk
    assert m.current_index == 0
    assert m.get_right_eef_9d() is None


def test_reset_clears_pending_chunk_flag():
    """A chunk delivered just before the pause must not survive the reset."""
    m = VLAActionManager(chunk_size=4, num_upper_body_joints=3)
    _publish_chunk(m, 7.0)  # arrives, but the session exits before step()
    m.reset()
    m.step()  # next session's first tick
    assert not m.has_first_chunk  # the stale delivery was not consumed


def test_fresh_chunk_after_reset_plays_back():
    m = VLAActionManager(chunk_size=4, num_upper_body_joints=3)
    _publish_chunk(m, 1.0)
    m.step()
    m.reset()

    _publish_chunk(m, 2.0)
    m.step()
    assert m.has_first_chunk
    assert np.allclose(m.get(definitions.RIGHT_EEF_KEY), 2.0)
