# MIT License
"""Tests for the dedicated zero-torque controller."""

import pytest

from g1_common.data import RobotState
from g1_control.controllers import ZeroTorqueController
from g1_definitions import definitions


def test_zero_torque_controller_zeros_every_torque_producing_channel():
    state = RobotState()
    state.mode_machine = 3
    command = ZeroTorqueController(state, {}).get_control()

    assert command.mode_machine == 3
    for motor in command.motor_cmd[: definitions.NUM_MOTOR_G1_BASE]:
        assert motor.q == 0.0
        assert motor.dq == 0.0
        assert motor.kp == 0.0
        assert motor.kd == 0.0
        assert motor.tau == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
