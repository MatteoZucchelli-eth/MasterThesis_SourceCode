# MIT License
"""Unit tests for the upper-body interpolation policy."""

import numpy as np
import pytest

from g1_control.policy.interpolation_policy import (
    InterpolationPolicy,
    PoseTrajectoryInterpolator,
)


def test_single_step_interpolator_is_constant():
    poses = np.array([[1.0, 2.0, 3.0]])
    interp = PoseTrajectoryInterpolator(np.array([0.0]), poses)
    assert np.allclose(interp(0.0), poses[0])
    assert np.allclose(interp(100.0), poses[0])  # clamps / holds


def test_interpolator_linear_midpoint():
    times = np.array([0.0, 2.0])
    poses = np.array([[0.0, 0.0], [2.0, 4.0]])
    interp = PoseTrajectoryInterpolator(times, poses)
    # Two knots leave PCHIP nothing to curve with, so it is still the secant.
    assert np.allclose(interp(1.0), [1.0, 2.0])
    # Queries outside the range are clamped to the endpoints.
    assert np.allclose(interp(-5.0), [0.0, 0.0])
    assert np.allclose(interp(99.0), [2.0, 4.0])


def test_interpolator_velocity_matches_a_finite_difference():
    """The analytic derivative is the curve's own, not an approximation of it."""
    times = np.array([0.0, 0.5, 1.0, 2.0])
    poses = np.array([[0.0, 0.0], [0.1, -0.1], [0.3, -0.2], [0.4, -0.2]])
    interp = PoseTrajectoryInterpolator(times, poses)
    for t in (0.25, 0.6, 1.5):
        h = 1e-6
        finite_diff = (interp(t + h) - interp(t - h)) / (2.0 * h)
        assert np.allclose(interp.velocity(t), finite_diff, atol=1e-5)


def test_interpolator_velocity_is_zero_outside_the_trajectory():
    """Past the end __call__ holds the pose, so the derivative is zero.

    The spline's own derivative at a clipped time would report the incoming slope.
    """
    interp = PoseTrajectoryInterpolator(
        np.array([0.0, 1.0]), np.array([[0.0, 0.0], [1.0, 1.0]])
    )
    assert np.allclose(interp.velocity(-1.0), 0.0)
    assert np.allclose(interp.velocity(5.0), 0.0)
    assert not np.allclose(interp.velocity(0.5), 0.0)


def test_single_step_interpolator_velocity_is_zero():
    interp = PoseTrajectoryInterpolator(np.array([0.0]), np.array([[1.0, 2.0]]))
    assert np.allclose(interp.velocity(0.0), 0.0)
    assert np.allclose(interp.velocity(9.0), 0.0)


def test_interpolator_does_not_overshoot_its_waypoints():
    """PCHIP is shape-preserving: no overshoot past the last waypoint (the wall)."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    poses = np.array([[0.0], [0.0], [1.0], [1.0]])
    interp = PoseTrajectoryInterpolator(times, poses)
    dense = interp(np.linspace(0.0, 3.0, 400))
    assert dense.min() >= -1e-12
    assert dense.max() <= 1.0 + 1e-12


# ── Re-planning: splicing a fresh goal onto one already being followed ──────
#
# The controller republishes a chunk ~1 Hz while earlier waypoints are still
# pending, so every chunk is a re-plan; the splice must not step the command.

_CHUNK = 30
_VLA_DT = 1.0 / 30.0
_TICK = 0.02


def _send_chunk(policy, chunk, now):
    """Mirror WholeBodyRLController._send_chunk_to_interpolator."""
    policy.set_goal(
        {
            "pose": list(chunk),
            "target_time": [now + (i + 1) * _VLA_DT for i in range(len(chunk))],
            "interpolation_garbage_collection_time": now,
        }
    )


def _ramp(start, stop):
    return np.linspace(start, stop, _CHUNK).reshape(-1, 1)


def test_replan_does_not_step_the_command():
    """The command at the re-plan instant is what it was an instant before."""
    rate = 0.25
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=rate
    )
    _send_chunk(policy, _ramp(0.0, 1.0), now=0.0)

    replan_at = 0.95  # the client's real republish period, inside the horizon
    before = policy.get_action(time=replan_at - 1e-6)["pose"].copy()
    _send_chunk(policy, _ramp(1.0, 2.0), now=replan_at)
    after = policy.get_action(time=replan_at)["pose"].copy()

    assert np.allclose(before, after, atol=1e-4)


def test_replan_cannot_bypass_the_rate_limit():
    """No tick may move further than max_change_rate allows, re-plans included."""
    rate = 0.25
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=rate
    )

    t, previous, worst_step = 0.0, policy.get_action(time=0.0)["pose"].copy(), 0.0
    for chunk_index in range(4):
        _send_chunk(policy, _ramp(chunk_index, chunk_index + 1.0), now=t)
        for _ in range(int(0.95 / _TICK)):
            t += _TICK
            current = policy.get_action(time=t)["pose"].copy()
            worst_step = max(worst_step, float(np.abs(current - previous).max()))
            previous = current

    # A tick's allowance, with a little slack for the cubic's local excess.
    assert worst_step <= rate * _TICK * 1.6


def test_a_goal_that_fits_the_rate_limit_is_not_late():
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=0.25
    )
    # 30 waypoints over 1.0 s covering 0.2 rad = 0.2 rad/s, inside the limit.
    _send_chunk(policy, _ramp(0.0, 0.2), now=0.0)
    assert policy.last_goal_lateness_s == pytest.approx(0.0, abs=1e-6)


def test_lateness_is_the_seconds_the_limit_adds():
    """1.0 rad at 0.25 rad/s needs 4 s; asked for in 1 s, so ~3 s late."""
    rate = 0.25
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=rate
    )
    _send_chunk(policy, _ramp(0.0, 1.0), now=0.0)
    assert policy.last_goal_lateness_s == pytest.approx(3.0, rel=0.1)


def test_lateness_ignores_a_single_fast_waypoint_the_chunk_absorbs():
    """One brisk step inside a gentle chunk is absorbed by later slack."""
    rate = 0.25
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=rate
    )
    steps = np.full(_CHUNK, 0.10 * _VLA_DT)
    steps[14] = 0.60 * _VLA_DT  # one waypoint at 2.4x the limit
    _send_chunk(policy, np.cumsum(steps).reshape(-1, 1), now=0.0)
    assert policy.last_goal_lateness_s == pytest.approx(0.0, abs=1e-3)


def test_lateness_includes_a_backlog_already_owed():
    """A goal issued while the arm is already behind finishes later than from rest.

    Lateness measures the limit's cost, not the policy's requested speed (see
    the controller's last_chunk_peak_speed).
    """
    rate = 0.25
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=rate
    )
    _send_chunk(policy, _ramp(0.0, 0.5), now=0.0)
    from_rest = policy.last_goal_lateness_s
    _send_chunk(policy, _ramp(0.5, 1.0), now=0.95)  # same speed, arm behind
    while_behind = policy.last_goal_lateness_s

    assert from_rest > 0.5  # the limit really is binding
    assert while_behind > from_rest


def test_a_single_target_stream_holds_its_pose_without_stepping():
    """The per-tick path used by VLA_PLANNER.

    Its first tick after chunk playback stops is a re-plan; the freeze must not twitch.
    """
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(1)}, max_change_rate=0.25
    )
    _send_chunk(policy, _ramp(0.0, 1.0), now=0.0)

    freeze_at = 0.4  # mid-chunk, waypoints still scheduled ahead
    before = policy.get_action(time=freeze_at - 1e-6)["pose"].copy()
    held = policy.get_action(time=freeze_at)["pose"].copy()

    t, previous = freeze_at, before
    for _ in range(50):
        policy.set_goal(
            {
                "pose": held,
                "target_time": t + _TICK,
                "interpolation_garbage_collection_time": t,
            }
        )
        t += _TICK
        current = policy.get_action(time=t)["pose"].copy()
        assert np.abs(current - previous).max() <= 0.25 * _TICK * 1.6
        previous = current

    assert np.allclose(previous, held, atol=1e-6)


def test_policy_velocity_has_the_same_keys_as_the_action():
    init = {"pose": np.zeros(3), "height": np.array([0.74])}
    policy = InterpolationPolicy(
        init_time=0.0, init_values=dict(init), max_change_rate=10.0
    )
    policy.set_goal(
        {
            "pose": np.array([1.0, 1.0, 1.0]),
            "target_time": 1.0,
            "interpolation_garbage_collection_time": -1.0,
        }
    )
    velocity = policy.get_velocity(time=0.5)
    assert sorted(velocity) == sorted(policy.get_action(time=0.5))
    assert np.all(velocity["pose"] > 0.0)  # moving toward the goal
    assert np.allclose(velocity["height"], 0.0)  # untouched key holds still


def test_policy_velocity_is_zero_once_the_goal_is_reached():
    """What makes it usable as a "this joint is holding" test."""
    policy = InterpolationPolicy(
        init_time=0.0, init_values={"pose": np.zeros(2)}, max_change_rate=10.0
    )
    policy.set_goal(
        {
            "pose": np.array([0.5, 0.5]),
            "target_time": 1.0,
            "interpolation_garbage_collection_time": -1.0,
        }
    )
    assert np.allclose(policy.get_action(time=50.0)["pose"], 0.5)
    assert np.allclose(policy.get_velocity(time=50.0)["pose"], 0.0)


def test_policy_initial_action_is_init_values():
    init = {"pose": np.array([0.5, -0.5]), "height": np.array([0.74])}
    policy = InterpolationPolicy(
        init_time=0.0, init_values=dict(init), max_change_rate=1000.0
    )
    action = policy.get_action(time=0.0)
    assert np.allclose(action["pose"], init["pose"])
    assert np.allclose(action["height"], init["height"])


def test_policy_reaches_goal():
    init = {"pose": np.array([0.0, 0.0])}
    policy = InterpolationPolicy(
        init_time=0.0, init_values=dict(init), max_change_rate=1000.0
    )
    goal_pose = np.array([1.0, -1.0])
    policy.set_goal(
        {
            "pose": goal_pose,
            "target_time": 1.0,
            "interpolation_garbage_collection_time": -1.0,
        }
    )
    action = policy.get_action(time=1.0)
    assert np.allclose(action["pose"], goal_pose, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
