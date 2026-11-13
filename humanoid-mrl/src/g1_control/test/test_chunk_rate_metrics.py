# MIT License
"""Unit tests for the chunk-vs-rate-limit metrics.

* ``last_chunk_peak_speed`` -- fastest joint speed the chunk requests (raw waypoints).
* ``last_goal_lateness_s`` -- seconds the limit added to this goal's end.
"""

import numpy as np
import pytest
from g1_common.data import RobotState

from g1_control.controllers import WholeBodyRLController
from g1_definitions import definitions

_FPS = definitions.RECORD_FPS
_VLA_DT = 1.0 / _FPS


class _StubPolicy:
    """Records the goals the controller forwards, ignoring the rest."""

    def __init__(self):
        self.goals = []

    def set_goal(self, goal):
        self.goals.append(goal)


def _controller():
    ctrl = WholeBodyRLController.__new__(WholeBodyRLController)
    ctrl._wbc_policy = _StubPolicy()
    ctrl._last_commanded_target = None
    ctrl._last_chunk_peak_speed = 0.0
    ctrl.robot_state = RobotState()
    return ctrl


def _chunk_at(speed_rad_s, n=30, joints=4):
    """A chunk whose every joint steps at `speed_rad_s`."""
    steps = np.arange(1, n + 1) * speed_rad_s * _VLA_DT
    return np.tile(steps.reshape(-1, 1), (1, joints))


def test_peak_speed_is_the_speed_the_chunk_asks_for():
    ctrl = _controller()
    for speed in (0.1, 0.25, 0.6, 1.0):
        ctrl._send_chunk_to_interpolator(_chunk_at(speed))
        assert ctrl.last_chunk_peak_speed == pytest.approx(speed, rel=1e-6)


def test_peak_speed_is_a_peak_over_joints_and_waypoints():
    """The limit is per joint per waypoint, so one brisk joint sets the peak."""
    ctrl = _controller()
    chunk = _chunk_at(0.1)
    chunk[15:, 2] += 0.6 * _VLA_DT  # a single 0.7 rad/s step on joint 2
    ctrl._send_chunk_to_interpolator(chunk)
    assert ctrl.last_chunk_peak_speed == pytest.approx(0.7, rel=1e-6)


def test_peak_speed_does_not_grow_with_the_backlog():
    """Chunks at one speed, each further out: the peak must not move."""
    ctrl = _controller()
    speeds = []
    for round_index in range(3):
        chunk = _chunk_at(0.6) + round_index * 0.6  # each starts further out
        ctrl._send_chunk_to_interpolator(chunk)
        speeds.append(ctrl.last_chunk_peak_speed)
    assert speeds == pytest.approx([0.6] * 3, rel=1e-6)


def test_peak_speed_ignores_where_the_arm_currently_is():
    """Distance from the arm is a backlog, not a demanded speed."""
    ctrl = _controller()
    near = _chunk_at(0.2)
    ctrl._send_chunk_to_interpolator(near)
    from_here = ctrl.last_chunk_peak_speed

    far = near + 5.0  # identical shape, a long way from the current pose
    ctrl._send_chunk_to_interpolator(far)
    assert ctrl.last_chunk_peak_speed == pytest.approx(from_here, rel=1e-6)


def test_a_single_waypoint_chunk_has_no_measurable_speed():
    ctrl = _controller()
    ctrl._send_chunk_to_interpolator(np.zeros((1, 4)))
    assert ctrl.last_chunk_peak_speed == 0.0


def test_the_chunk_is_still_forwarded_with_its_own_timing():
    """The metric is a side observation; the goal must be unchanged by it."""
    ctrl = _controller()
    ctrl._send_chunk_to_interpolator(_chunk_at(0.2), start_index=3)
    goal = ctrl._wbc_policy.goals[-1]
    times = goal["target_time"]
    assert len(goal["target_upper_body_pose"]) == 30 - 3
    # Skipped waypoints keep their slots, so the remainder still arrives on beat.
    assert np.allclose(np.diff(times), _VLA_DT)
    assert times[0] == pytest.approx(
        goal["interpolation_garbage_collection_time"] + 4 * _VLA_DT
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
