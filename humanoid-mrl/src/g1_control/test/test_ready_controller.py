# MIT License
"""Unit tests for the READY ramp's start pose.

The ramp lerps from the entry pose to the default pose; the start endpoint must
be latched once, not re-read each tick (which would fold drift into the reference).
"""

import numpy as np
import pytest
from g1_common.data import RobotState

from g1_control.controllers.ready_controller import ReadyController
from g1_definitions import definitions

_N = definitions.NUM_MOTOR_G1_BASE
_DT = 0.02
_DURATION = 10.0  # ReadyController._ready_state_transition_duration


def _controller(start: np.ndarray, default: np.ndarray) -> ReadyController:
    rs = RobotState()
    rs.joint_pos = start.copy()
    rs.mode_machine = 0
    ctrl = ReadyController(
        robot_state=rs,
        gains={},
        stiffness_array=np.full(_N, 100.0),
        damping_array=np.full(_N, 1.0),
        default_joint_pos=default,
        control_dt=_DT,
        # No gravity feedforward; build_low_cmd zero-fills tau.
        robot_model=None,
    )
    ctrl.on_activate()
    return ctrl


def _q(cmd) -> np.ndarray:
    return np.array([cmd.motor_cmd[i].q for i in range(_N)])


def test_start_pose_is_latched_not_re_read():
    """A drifting arm must not drag the ramp's start point with it."""
    start = np.zeros(_N)
    default = np.ones(_N)
    ctrl = _controller(start, default)

    ctrl.get_control()  # latches on the first tick

    # Sag the arm while stiffness is still ramping.
    ctrl.robot_state.joint_pos = np.full(_N, -0.5)

    # The command stays on the line from the latched start, not the sagged pose.
    ctrl.get_control()
    cmd = ctrl.get_control()
    ratio = 3 * _DT / _DURATION
    np.testing.assert_allclose(_q(cmd), (1.0 - ratio) * start + ratio * default)


def test_ramp_ends_exactly_on_the_default_pose():
    """Whatever the arm did during the ramp, ratio == 1 commands the default."""
    ctrl = _controller(np.zeros(_N), np.ones(_N))
    ctrl.get_control()
    ctrl.robot_state.joint_pos = np.full(_N, -0.5)  # never converged
    for _ in range(int(_DURATION / _DT) + 5):
        cmd = ctrl.get_control()
    np.testing.assert_allclose(_q(cmd), np.ones(_N))


def test_stiffness_ramps_with_the_ratio():
    """Kp scales 0 -> full over the ramp; Kd is constant throughout."""
    ctrl = _controller(np.zeros(_N), np.ones(_N))
    first = ctrl.get_control()
    assert first.motor_cmd[0].kp == pytest.approx(100.0 * _DT / _DURATION)
    assert first.motor_cmd[0].kd == pytest.approx(1.0)
    for _ in range(int(_DURATION / _DT) + 5):
        last = ctrl.get_control()
    assert last.motor_cmd[0].kp == pytest.approx(100.0)
    assert last.motor_cmd[0].kd == pytest.approx(1.0)


def test_re_entry_latches_the_new_pose():
    """on_activate must re-latch, or a second entry ramps from a stale pose."""
    ctrl = _controller(np.zeros(_N), np.ones(_N))
    for _ in range(int(_DURATION / _DT) + 5):
        ctrl.get_control()

    ctrl.robot_state.joint_pos = np.full(_N, 0.4)
    ctrl.on_activate()
    cmd = ctrl.get_control()
    ratio = _DT / _DURATION
    np.testing.assert_allclose(
        _q(cmd), (1.0 - ratio) * np.full(_N, 0.4) + ratio * np.ones(_N)
    )


def test_no_command_before_the_first_state_and_no_time_advances():
    """A tick without joint_pos must not consume ramp time or latch anything."""
    ctrl = _controller(np.zeros(_N), np.ones(_N))
    ctrl.robot_state.joint_pos = None
    assert ctrl.get_control() is None

    # First real tick: ratio = dt/T from the pose present now.
    ctrl.robot_state.joint_pos = np.full(_N, 0.25)
    cmd = ctrl.get_control()
    ratio = _DT / _DURATION
    np.testing.assert_allclose(
        _q(cmd), (1.0 - ratio) * np.full(_N, 0.25) + ratio * np.ones(_N)
    )
