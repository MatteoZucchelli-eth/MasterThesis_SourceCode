# MIT License
"""Tests for the recorder's image<->state pairing (pure logic, no ROS init)."""

import numpy as np
import pytest

from g1_definitions import definitions
from g1_interfaces.msg import ControlStep

from g1_teleoperation.recorder_node import (
    _extract_action_vector,
    _extract_state_vector,
    bracketing_weights,
    clocks_agree,
    image_is_stale,
    interpolate_steps,
)


def _step(value: float, tip_valid: bool = True, tip_dist: float = 0.1) -> ControlStep:
    """ControlStep whose every scalar field is ``value`` (joints, eefs, label)."""
    msg = ControlStep()
    for m in msg.low_state.motor_state[: definitions.NUM_MOTOR_G1_BASE]:
        m.q = value
    msg.left_eef_pose = [value] * 9
    msg.right_eef_pose = [value] * 9
    msg.left_eef_target = [value] * 9
    msg.right_eef_target = [value] * 9
    msg.tip_target_valid = tip_valid
    msg.tip_target_9d = [value] * 9
    msg.tip_target_dist = tip_dist
    return msg


def test_bracketing_interior():
    times = [0.0, 0.01, 0.02, 0.03]
    i0, i1, w = bracketing_weights(times, 0.015)
    assert (i0, i1) == (1, 2)
    assert w == pytest.approx(0.5)


def test_bracketing_exact_hit():
    times = [0.0, 0.01, 0.02]
    i0, i1, w = bracketing_weights(times, 0.01)
    assert times[i0] * (1 - w) + times[i1] * w == pytest.approx(0.01)


def test_bracketing_clamps_ends():
    times = [1.0, 2.0]
    assert bracketing_weights(times, 0.5) == (0, 0, 0.0)
    assert bracketing_weights(times, 3.0) == (1, 1, 0.0)


def test_interpolate_midpoint():
    fields = interpolate_steps(_step(0.0, tip_dist=0.1), _step(1.0, tip_dist=0.3), 0.5)
    assert np.allclose(fields["observation.state"], 0.5)
    assert np.allclose(fields["action"], 0.5)
    assert np.allclose(fields["tip_target_9d"], 0.5)
    assert fields["tip_target_dist"] == pytest.approx(0.2)


def test_interpolate_endpoint_weights():
    fields = interpolate_steps(_step(0.0), _step(1.0), 0.0)
    assert np.allclose(fields["observation.state"], 0.0)
    fields = interpolate_steps(_step(0.0), _step(1.0), 1.0)
    assert np.allclose(fields["observation.state"], 1.0)


def test_interpolate_clamps_weight():
    fields = interpolate_steps(_step(0.0), _step(1.0), 1.7)
    assert np.allclose(fields["observation.state"], 1.0)


def test_invalid_target_stays_nan():
    fields = interpolate_steps(_step(0.5), _step(0.5, tip_valid=False), 0.5)
    assert np.isnan(fields["tip_target_9d"]).all()
    assert np.isnan(fields["tip_target_dist"])
    # State/action interpolation is unaffected by label validity.
    assert np.allclose(fields["observation.state"], 0.5)


def test_pairing_picks_the_state_at_the_image_stamp():
    """An image stamped between two states resolves to the state at that instant."""
    stamps = [10.00, 10.02, 10.04, 10.06]
    i0, i1, w = bracketing_weights(stamps, 10.03)
    assert (i0, i1) == (1, 2)
    fields = interpolate_steps(_step(float(i0)), _step(float(i1)), w)
    assert np.allclose(fields["observation.state"], 1.5)


def test_clocks_agree_in_sim():
    """Both sides on sim time: measured live at a steady 0.054 s render lag."""
    assert clocks_agree(6952.710, 6952.764)


def test_clocks_disagree_for_boot_counter():
    """Real robot: a firmware boot counter against a system-time stamp."""
    assert not clocks_agree(1.7e9, 812.4)


def test_a_lagging_camera_is_still_the_same_clock():
    """Seconds of gap is a backlog, not a second clock — and pairs on tick.

    A cold render pipeline can put a fresh worker's first frames 1-2 s behind.
    """
    assert clocks_agree(100.0, 101.4)
    assert clocks_agree(100.0, 109.0)
    assert image_is_stale(100.0, 101.4)
    assert image_is_stale(100.0, 109.0)


def test_one_render_lag_is_not_stale():
    """The measured 0.054 s render lag must never raise the staleness alarm."""
    assert not image_is_stale(6952.710, 6952.764)


def test_an_image_ahead_of_the_newest_state_is_not_stale():
    """Staleness is one-directional: a *late* image, not merely a distant one."""
    assert not image_is_stale(100.2, 100.0)


def test_state_vector_layout():
    state = _extract_state_vector(_step(0.25))
    assert state.shape == (definitions.STATE_DIM,)
    assert np.allclose(state, 0.25)


def test_state_vector_slices_the_right_arm():
    """Each recorded block must be the right arm's own motors, in PR order."""
    msg = _step(0.0)
    for i, m in enumerate(msg.low_state.motor_state[: definitions.NUM_MOTOR_G1_BASE]):
        m.q = float(i)
    msg.right_eef_pose = list(np.arange(9, dtype=np.float32))

    state = _extract_state_vector(msg)
    pr_lo, pr_hi = definitions.JOINT_PR_SLICES["right_arm"]

    lo, hi = definitions.STATE_SLICES[definitions.RIGHT_EEF_KEY]
    assert np.allclose(state[lo:hi], np.arange(9))
    lo, hi = definitions.STATE_SLICES["right_arm"]
    assert np.allclose(state[lo:hi], np.arange(pr_lo, pr_hi))


def test_action_vector_is_the_tip_target():
    msg = _step(0.0)
    msg.right_eef_target = list(np.arange(9, dtype=np.float32))
    action = _extract_action_vector(msg)
    assert action.shape == (definitions.ACTION_DIM,)
    assert np.allclose(action, np.arange(9))


# ── Camera-backlog pairing ────────────────────────────────────────────────────
#
# Both clocks are sim time; a camera backlog of B wall seconds makes the newest
# state lead the image by r*B sim seconds (r = real-time factor). Pairing on the
# image stamp stays correct; pairing by arrival time is off by r*B. A gap
# threshold (e.g. 10 s -> 300 frames at 30 fps) that switches to arrival pairing
# therefore injects exactly that offset. Only a foreign clock (boot counter,
# ~1e9 s gap) should switch the basis.

_OLD_TIMEBASE_COMPARABLE_S = 10.0
_FPS = 30


def _backlogged_pairing(rtf: float, backlog_wall_s: float, use_tick: bool) -> float:
    """Sim time of the state a pairing picks for an image rendered at sim t=0.

    The image arrives ``backlog_wall_s`` late. 0.0 is a correct pairing.
    """
    lag_sim = rtf * backlog_wall_s
    ticks = np.arange(-20.0, lag_sim + 1e-9, 0.01)  # 100 Hz sim ControlSteps
    arrivals = ticks / rtf  # wall time each one was published at
    if use_tick:
        i0, i1, w = bracketing_weights(list(ticks), 0.0)  # ref = the image stamp
    else:
        # Arrival pairing: render wall time (0) plus the backlog.
        i0, i1, w = bracketing_weights(list(arrivals), backlog_wall_s)
    return float(ticks[i0] + w * (ticks[i1] - ticks[i0]))


@pytest.mark.parametrize("rtf, backlog_wall_s", [(0.33, 33.0), (0.5, 24.0)])
def test_tick_pairing_is_correct_however_far_the_camera_lags(rtf, backlog_wall_s):
    """Pairing at the image *stamp* is right even with a huge backlog."""
    picked = _backlogged_pairing(rtf, backlog_wall_s, use_tick=True)
    assert picked == pytest.approx(0.0, abs=0.01), "the image's own instant"


@pytest.mark.parametrize("rtf, backlog_wall_s", [(0.33, 33.0), (0.5, 24.0)])
def test_arrival_pairing_under_a_backlog_is_off_by_the_backlog(rtf, backlog_wall_s):
    """Falling back to arrival time is what injects the offset."""
    picked = _backlogged_pairing(rtf, backlog_wall_s, use_tick=False)
    assert picked == pytest.approx(rtf * backlog_wall_s, abs=0.01)
    offset_frames = picked * _FPS
    assert offset_frames > 300, "the fallback only triggers past ~300 frames"


def test_the_old_tolerance_switched_bases_at_the_observed_offset():
    """A 10 s gap tolerance switches bases at a 300-frame offset."""
    flip_frames = _OLD_TIMEBASE_COMPARABLE_S * _FPS
    assert flip_frames == 300
    assert 105 < flip_frames < 463


@pytest.mark.parametrize("rtf, backlog_wall_s", [(0.33, 33.0), (0.5, 24.0)])
def test_a_backlog_is_reported_but_never_switches_the_basis(rtf, backlog_wall_s):
    """A backlog keeps tick pairing (clocks_agree) and raises staleness."""
    newest_tick = rtf * backlog_wall_s
    assert clocks_agree(0.0, newest_tick), "same clock — keep pairing on tick"
    assert image_is_stale(0.0, newest_tick), "and say so"
    assert not image_is_stale(0.0, 0.054), "one render lag is healthy"


def test_a_boot_counter_is_the_only_thing_that_switches_the_basis():
    """The 1e9-scale gap no lag can reach: only there is arrival time better."""
    assert not clocks_agree(1.78e9, 4.29e6)  # epoch stamp vs the uint32 ms wrap
    worst_survivable_backlog_s = 0.5 * 24.0
    assert clocks_agree(0.0, worst_survivable_backlog_s)
