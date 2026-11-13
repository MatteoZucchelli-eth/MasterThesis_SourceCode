# MIT License
"""Unit tests for the arm feedforward integrator.

The integrator accumulates every joint's error unconditionally (no dead-band or
speed gate); the clamp is its only limit. Covers accumulation, clamp, timebase
and reset.
"""

import numpy as np
import pytest
from g1_common.data import RobotState

from g1_control.controllers import WholeBodyRLController
from g1_definitions import definitions

_N = definitions.NUM_MOTOR_G1_BASE
_RIGHT_ARM = definitions.JOINT_PR_SLICES["right_arm"]
_SHOULDER_PITCH = _RIGHT_ARM[0]  # first right-arm joint in PR order


def _controller(ki=50.0, clamp=5.0):
    """A controller with only the integrator's state wired up (no ONNX load)."""
    ctrl = WholeBodyRLController.__new__(WholeBodyRLController)
    ctrl._control_dt = 0.02
    ctrl.robot_state = RobotState()
    ctrl.robot_state.joint_pos = np.zeros(_N)
    ctrl._arm_int_ki = ki
    ctrl._arm_int_clamp = np.full(_N, clamp)
    ctrl._arm_int_state = np.zeros(_N)
    ctrl._arm_int_last_t = None
    return ctrl


def _run(ctrl, error, seconds, dt=0.02, meas_speed=0.0):
    """Hold a constant error on shoulder pitch and return the torque it built.

    ``meas_speed`` drives the measured joint position at a constant rate while
    keeping the error fixed, i.e. the commanded pose slides with it.
    """
    t = 0.0
    for _ in range(int(seconds / dt)):
        t += dt
        measured = meas_speed * t
        ctrl.robot_state.joint_pos = np.full(_N, measured)
        action = np.full(_N, measured)
        action[_SHOULDER_PITCH] = measured + error
        ctrl._accumulate_arm_error(action, t)
    return ctrl._arm_int_ki * ctrl._arm_int_state[_SHOULDER_PITCH]


def test_a_held_joint_corrects_its_sag():
    """A standing error becomes torque."""
    assert _run(_controller(), error=0.025, seconds=2.0) > 0.5


def test_the_torque_is_ki_times_the_integral():
    """No dead-band, no scaling: exactly Ki * err * t while under the clamp."""
    torque = _run(_controller(ki=50.0), error=0.01, seconds=1.0)
    assert torque == pytest.approx(50.0 * 0.01 * 1.0, rel=1e-6)


# ── The gates, and their absence ─────────────────────────────────────────────


def test_a_tiny_error_is_still_integrated():
    """There is no dead-band."""
    assert _run(_controller(), error=0.0005, seconds=5.0) > 0.0


def test_a_fast_reference_is_integrated_like_any_other():
    """There is no reference-speed gate."""
    fast = _run(_controller(), error=0.025, seconds=2.0)
    held = _run(_controller(), error=0.025, seconds=2.0)
    assert fast == pytest.approx(held)
    assert fast > 0.5


def test_a_sliding_joint_is_integrated_like_any_other():
    """There is no measured-speed gate (on hardware, fight stick-slip with Ki/clamp)."""
    sliding = _run(_controller(), error=0.025, seconds=4.0, meas_speed=0.01)
    parked = _run(_controller(), error=0.025, seconds=4.0, meas_speed=0.0)
    assert sliding == pytest.approx(parked)


def test_every_arm_joint_accumulates_independently():
    """No cross-joint coupling: each joint integrates its own error."""
    ctrl = _controller()
    action = np.zeros(_N)
    action[_SHOULDER_PITCH] = 0.02
    action[_SHOULDER_PITCH + 1] = 0.01
    for i in range(100):
        ctrl._accumulate_arm_error(action, 0.02 * (i + 1))
    assert ctrl._arm_int_state[_SHOULDER_PITCH] == pytest.approx(
        2.0 * ctrl._arm_int_state[_SHOULDER_PITCH + 1]
    )


# ── The clamp ────────────────────────────────────────────────────────────────


def test_the_clamp_bounds_the_torque():
    """Anti-windup, and what keeps the 5 Nm wrists inside their effort limit."""
    assert _run(_controller(clamp=5.0), error=0.5, seconds=30.0) == pytest.approx(5.0)


def test_the_clamp_bounds_a_negative_error_too():
    assert _run(_controller(clamp=5.0), error=-0.5, seconds=30.0) == pytest.approx(-5.0)


def test_a_clamped_integrator_unwinds_when_the_error_reverses():
    """It must not latch: the clamp bounds the state, it does not freeze it."""
    ctrl = _controller(clamp=5.0)
    assert _run(ctrl, error=0.5, seconds=30.0) == pytest.approx(5.0)
    assert _run(ctrl, error=-0.5, seconds=30.0) == pytest.approx(-5.0)


# ── Timebase and lifecycle ───────────────────────────────────────────────────


def test_reset_drops_the_accumulated_torque():
    ctrl = _controller()
    assert _run(ctrl, error=0.025, seconds=2.0) > 0.5
    ctrl._reset_arm_integral()
    assert ctrl._arm_int_state[_SHOULDER_PITCH] == 0.0
    assert ctrl._arm_int_last_t is None


def test_controller_uses_the_injected_control_clock():
    """Simulation timing must follow ROS/Isaac time, not host wall time."""
    ctrl = _controller()
    ctrl._clock = lambda: 12.5
    assert ctrl._now() == pytest.approx(12.5)


def test_reactivating_the_same_controller_preserves_the_integral():
    """A handler transition must not discard the load correction it learned."""

    class _UpperBodyPolicy:
        def reset(self, *_args, **_kwargs):
            pass

    class _Policy:
        upper_body_policy = _UpperBodyPolicy()

        def activate_policy(self):
            pass

    ctrl = _controller()
    ctrl.robot_state.joint_pos = None
    ctrl._wbc_policy = _Policy()
    ctrl._arm_int_state[_SHOULDER_PITCH] = 0.25

    ctrl.on_activate(reset_arm_integral=False)
    assert ctrl._arm_int_state[_SHOULDER_PITCH] == pytest.approx(0.25)

    # Returning after a different controller had control is still a fresh
    # activation and must not revive an old correction.
    ctrl.on_activate()
    assert ctrl._arm_int_state[_SHOULDER_PITCH] == 0.0


def test_a_late_tick_integrates_the_interval_that_elapsed():
    """Real elapsed time is integrated, not the nominal period."""
    action = np.zeros(_N)
    action[_SHOULDER_PITCH] = 0.1

    punctual = _controller()
    punctual._accumulate_arm_error(action, 0.02)
    punctual._accumulate_arm_error(action, 0.04)

    late = _controller()
    late._accumulate_arm_error(action, 0.02)
    late._accumulate_arm_error(action, 0.06)  # 40 ms late

    assert punctual._arm_int_state[_SHOULDER_PITCH] == pytest.approx(0.1 * 0.04)
    assert late._arm_int_state[_SHOULDER_PITCH] == pytest.approx(0.1 * 0.06)


def test_a_very_late_tick_cannot_integrate_its_whole_gap():
    """A resumed session must not dump seconds of error in one step."""
    action = np.zeros(_N)
    action[_SHOULDER_PITCH] = 0.1

    ctrl = _controller()
    ctrl._accumulate_arm_error(action, 0.02)
    ctrl._accumulate_arm_error(action, 10.0)  # a 10 s gap

    # Capped at 4 nominal ticks (80 ms), not 10 s.
    assert ctrl._arm_int_state[_SHOULDER_PITCH] <= 0.1 * 5 * 0.02


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
