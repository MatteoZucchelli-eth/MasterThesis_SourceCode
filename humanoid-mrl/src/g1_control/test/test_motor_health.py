# MIT License
"""Unit tests for the motor-health monitor's window reduction; it must never raise into
the executor."""

from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from g1_control.nodes.motor_health_node import MotorHealthNode  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def _node():
    node = MotorHealthNode()
    # Explicit effort limit instead of the URDF.
    node._effort = np.full(len(node._names), 25.0)
    return node


def _msg(n, *, fault=0, temp=40.0, tau=1.0, vol=48.0):
    ms = [
        SimpleNamespace(
            motorstate=fault, temperature=[temp, temp - 2.0], tau_est=tau, vol=vol
        )
        for _ in range(n)
    ]
    return SimpleNamespace(motor_state=ms)


def test_fault_report_does_not_raise_on_the_window_temperature():
    """A fault word plus accumulated temperatures must log, not TypeError."""
    node = _node()
    try:
        for _ in range(5):
            node._on_low_state(_msg(len(node._names), fault=0x40, temp=61.0))
        node._evaluate()  # must not raise TypeError
        assert node._fault_word[node._idx[0]] == 0x40
    finally:
        node.destroy_node()


def test_fault_report_survives_a_window_with_no_temperature():
    """The slow fields are not refreshed every message; a window may have none."""
    node = _node()
    try:
        node._on_low_state(_msg(len(node._names), fault=0x2))
        for i in node._idx:  # simulate a window that carried no temperature
            node._acc_temp[i].clear()
        node._evaluate()
        assert node._fault_word[node._idx[0]] == 0x2
    finally:
        node.destroy_node()


def test_voltage_is_checked_even_when_temperature_is_absent():
    """A missing temperature must not skip the rest of the motor's checks."""
    node = _node()
    try:
        node._on_low_state(_msg(len(node._names), vol=30.0))
        for i in node._idx:
            node._acc_temp[i].clear()
        node._evaluate()
        assert node._low_volt, "sagging bus went unreported"
    finally:
        node.destroy_node()


def test_temperature_uses_the_window_median_not_a_single_frame():
    """One stale/zero frame must not latch the reported temperature."""
    node = _node()
    try:
        n = len(node._names)
        for t in (71.0, 71.0, 0.0, 72.0, 71.0):
            node._on_low_state(_msg(n, temp=t))
        node._evaluate()
        assert node._hot, "a hot motor was not reported"
        i = node._idx[0]
        assert 70.0 <= node._peak_temp[i] <= 72.0
    finally:
        node.destroy_node()
