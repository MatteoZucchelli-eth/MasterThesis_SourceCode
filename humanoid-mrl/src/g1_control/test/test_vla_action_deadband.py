# MIT License
"""The between-chunk action deadband (no server, no ROS, no robot model).

The deadband is off unless configured, and its reference is the last
**published** chunk, so suppressed noise cannot random-walk the tip while a
genuine creep still gets through.
"""

import threading

import numpy as np
import pytest

from g1_control.vla.client import VLAClient
from g1_control.vla.deadband import chunk_within_deadband
from g1_definitions import definitions

IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def chunk(xyz=(0.0, 0.0, 0.0), steps=30, rot6d=IDENTITY_ROT6D):
    """A constant ``(steps, 9)`` tip chunk parked at ``xyz``."""
    pose = np.concatenate([np.asarray(xyz, dtype=float), np.asarray(rot6d, dtype=float)])
    return np.tile(pose, (steps, 1))


# ── the predicate ────────────────────────────────────────────────────────────


def test_identical_chunks_are_inside_the_deadband():
    assert chunk_within_deadband(chunk(), chunk(), pos_tol=0.002, rot_tol_deg=1.0)


def test_motion_under_the_tolerance_is_held():
    assert chunk_within_deadband(
        chunk((0.0019, 0, 0)), chunk(), pos_tol=0.002, rot_tol_deg=1.0
    )


def test_motion_over_the_tolerance_publishes():
    assert not chunk_within_deadband(
        chunk((0.0021, 0, 0)), chunk(), pos_tol=0.002, rot_tol_deg=1.0
    )


def test_a_single_step_over_tolerance_publishes_the_whole_chunk():
    """The bound is over every step, not the mean: one real waypoint is enough."""
    moving = chunk()
    moving[17, 0] = 0.05
    assert not chunk_within_deadband(moving, chunk(), pos_tol=0.002, rot_tol_deg=1.0)


def test_zero_tolerance_disables_the_deadband():
    """Zero tolerance never holds a chunk."""
    assert not chunk_within_deadband(chunk(), chunk(), pos_tol=0.0, rot_tol_deg=1.0)


def test_no_published_chunk_yet_publishes():
    assert not chunk_within_deadband(chunk(), None, pos_tol=0.002, rot_tol_deg=1.0)


def test_a_shape_change_publishes():
    assert not chunk_within_deadband(
        chunk(steps=30), chunk(steps=16), pos_tol=0.002, rot_tol_deg=1.0
    )


def test_rotation_alone_publishes_even_when_the_tip_does_not_move():
    """Position and rotation are ANDed, so a turn in place is not swallowed."""
    turned = chunk(rot6d=np.array([np.cos(0.2), np.sin(0.2), 0.0, -np.sin(0.2), np.cos(0.2), 0.0]))
    assert not chunk_within_deadband(turned, chunk(), pos_tol=0.002, rot_tol_deg=1.0)


def test_rotation_under_its_own_tolerance_is_held():
    tiny = np.radians(0.5)
    turned = chunk(rot6d=np.array([np.cos(tiny), np.sin(tiny), 0.0, -np.sin(tiny), np.cos(tiny), 0.0]))
    assert chunk_within_deadband(turned, chunk(), pos_tol=0.002, rot_tol_deg=1.0)


# ── the publish gate ─────────────────────────────────────────────────────────


def _client(deadband=0.002, rot_deg=1.0, steps=30):
    """A bare client carrying only what ``copy_action_chunk`` reads."""
    client = object.__new__(VLAClient)
    client._action_chunk_lock = threading.Lock()
    client._new_action_chunk = threading.Event()
    client._latest_action_chunk = {
        definitions.RIGHT_EEF_KEY: np.zeros((steps, 9), dtype=np.float32),
        "upper_body_joint_chunk": np.zeros((steps, 28), dtype=np.float32),
    }
    client._action_deadband = deadband
    client._action_deadband_rot_deg = rot_deg
    client._published_tip_chunk = None
    client._deadband_held = 0
    client._warned_no_action_head = False
    return client


def _publish(client, xyz):
    client._new_action_chunk.clear()
    client.copy_action_chunk({definitions.RIGHT_EEF_KEY: chunk(xyz)})
    return client._new_action_chunk.is_set()


def test_the_first_chunk_always_publishes():
    client = _client()
    assert _publish(client, (0.0, 0, 0))


def test_a_jittering_chunk_leaves_the_buffer_untouched():
    client = _client()
    _publish(client, (0.10, 0, 0))
    before = client._latest_action_chunk[definitions.RIGHT_EEF_KEY].copy()

    assert not _publish(client, (0.1005, 0, 0))
    assert np.array_equal(client._latest_action_chunk[definitions.RIGHT_EEF_KEY], before)


def test_sub_threshold_noise_does_not_random_walk_the_tip():
    """Every suppressed chunk is measured from the same published pose.

    Applies to noise strictly inside the tolerance.
    """
    rng = np.random.default_rng(0)
    client = _client(deadband=0.002)
    _publish(client, (0.10, 0, 0))
    origin = client._latest_action_chunk[definitions.RIGHT_EEF_KEY].copy()

    for _ in range(200):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        step = direction * rng.uniform(0.0, 0.0019)  # strictly inside the deadband
        _publish(client, np.array([0.10, 0.0, 0.0]) + step)

    assert np.array_equal(client._latest_action_chunk[definitions.RIGHT_EEF_KEY], origin)


def test_a_slow_creep_still_gets_through():
    """No permanent lockout: drift accumulates against the fixed reference."""
    client = _client(deadband=0.002)
    _publish(client, (0.0, 0, 0))

    published = [_publish(client, (0.0008 * i, 0, 0)) for i in range(1, 6)]
    assert published == [False, False, True, False, False]
    # ...and once it publishes, the reference moves with it.
    assert client._latest_action_chunk[definitions.RIGHT_EEF_KEY][0, 0] == pytest.approx(
        0.0024, abs=1e-6
    )


def test_a_disabled_deadband_publishes_every_chunk():
    client = _client(deadband=0.0)
    _publish(client, (0.10, 0, 0))
    assert _publish(client, (0.1000001, 0, 0))
