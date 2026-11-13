# MIT License
"""Unit tests for the LowCmd builder."""

import numpy as np
import pytest

from g1_control.utils import build_low_cmd, nonfinite_motors
from g1_definitions import definitions


def test_build_low_cmd_maps_arrays_per_motor():
    n = definitions.NUM_MOTOR_G1_BASE
    q = np.arange(n, dtype=float)
    kp = np.full(n, 10.0)
    kd = np.full(n, 2.0)
    tau = np.linspace(0.0, 1.0, n)

    cmd = build_low_cmd(mode_machine=7, q=q, kp=kp, kd=kd, tau=tau)

    assert cmd.mode_machine == 7
    assert cmd.mode_pr == definitions.G1JointControlMode.PR_Mode.value
    assert list(cmd.reserve) == [0, 0, 0, 0]
    for i in range(n):
        assert cmd.motor_cmd[i].mode == 1
        assert cmd.motor_cmd[i].q == q[i]
        assert cmd.motor_cmd[i].dq == 0.0
        assert cmd.motor_cmd[i].kp == kp[i]
        assert cmd.motor_cmd[i].kd == kd[i]
        assert np.isclose(cmd.motor_cmd[i].tau, tau[i])


def test_build_low_cmd_defaults_tau_to_zero():
    n = definitions.NUM_MOTOR_G1_BASE
    cmd = build_low_cmd(0, q=np.zeros(n), kp=np.zeros(n), kd=np.ones(n))
    assert all(cmd.motor_cmd[i].tau == 0.0 for i in range(n))


def test_build_low_cmd_defaults_dq_to_zero():
    n = definitions.NUM_MOTOR_G1_BASE
    cmd = build_low_cmd(0, q=np.zeros(n), kp=np.zeros(n), kd=np.ones(n))
    assert all(cmd.motor_cmd[i].dq == 0.0 for i in range(n))


def test_build_low_cmd_forwards_the_velocity_target():
    """dq is the motor's velocity target, so zero is a brake, not a neutral."""
    n = definitions.NUM_MOTOR_G1_BASE
    dq = np.linspace(-0.5, 0.5, n)
    cmd = build_low_cmd(0, q=np.zeros(n), kp=np.zeros(n), kd=np.ones(n), dq=dq)
    for i in range(n):
        assert np.isclose(cmd.motor_cmd[i].dq, dq[i])


def test_nonfinite_motors_empty_for_a_clean_command():
    n = definitions.NUM_MOTOR_G1_BASE
    cmd = build_low_cmd(0, q=np.zeros(n), kp=np.full(n, 10.0), kd=np.ones(n))
    assert nonfinite_motors(cmd) == []


@pytest.mark.parametrize("field", ["q", "tau", "kp", "kd"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_motors_reports_the_offending_joint(field, bad):
    """Every commanded channel is checked for NaN and +/-inf."""
    n = definitions.NUM_MOTOR_G1_BASE
    arrays = {
        "q": np.zeros(n),
        "kp": np.full(n, 10.0),
        "kd": np.ones(n),
        "tau": np.zeros(n),
    }
    arrays[field][7] = bad

    cmd = build_low_cmd(0, **arrays)

    assert nonfinite_motors(cmd) == [7]


def test_nonfinite_motors_ignores_slots_past_the_base_motors():
    """The G1 LowCmd carries 35 slots; only the 29 base motors are commanded."""
    n = definitions.NUM_MOTOR_G1_BASE
    cmd = build_low_cmd(0, q=np.zeros(n), kp=np.full(n, 10.0), kd=np.ones(n))
    cmd.motor_cmd[n].q = float("nan")
    assert nonfinite_motors(cmd) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
