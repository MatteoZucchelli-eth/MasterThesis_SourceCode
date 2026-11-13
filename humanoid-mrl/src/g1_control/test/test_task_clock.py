# MIT License
"""Tests for the task clock — which clock a duration is measured on."""

import time
from types import SimpleNamespace

import pytest

from g1_control.utils.task_clock import TaskClock


def _node(ros_seconds, *, param=None):
    """Node stub with a ROS clock and an optional use_sim_time parameter."""
    return SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(ros_seconds["t"] * 1e9))
        ),
        has_parameter=lambda name: param is not None,
        get_parameter=lambda name: SimpleNamespace(value=param),
    )


def test_on_sim_time_it_reads_the_node_clock():
    ros = {"t": 12.5}
    clock = TaskClock(_node(ros), sim_time=True)
    assert clock() == pytest.approx(12.5)
    ros["t"] = 40.0
    assert clock() == pytest.approx(40.0)


def test_off_sim_time_it_ignores_the_node_clock():
    """Off sim time, durations use monotonic time (immune to NTP steps)."""
    ros = {"t": 1e9}  # a system clock, wildly different from monotonic
    clock = TaskClock(_node(ros), sim_time=False)
    first = clock()
    assert first == pytest.approx(time.monotonic(), abs=1.0)
    ros["t"] -= 3600.0  # NTP steps the system clock back an hour
    assert clock() >= first, "monotonic never goes backwards"


def test_it_reads_use_sim_time_from_the_node_by_default():
    assert TaskClock(_node({"t": 5.0}, param=True)).sim_time is True
    assert TaskClock(_node({"t": 5.0}, param=False)).sim_time is False


def test_a_node_without_the_parameter_is_not_on_sim_time():
    """Nothing declares it on the real robot, and there is no /clock there."""
    assert TaskClock(_node({"t": 5.0})).sim_time is False


def test_stalled_flags_sim_time_with_no_clock_publisher():
    """ROS time pinned at 0 under use_sim_time means no /clock publisher."""
    assert TaskClock(_node({"t": 0.0}), sim_time=True).stalled() is True
    assert TaskClock(_node({"t": 0.1}), sim_time=True).stalled() is False


def test_a_stopped_wall_clock_is_never_reported_as_stalled():
    """Off sim time there is no /clock to wait for, so nothing to report."""
    assert TaskClock(_node({"t": 0.0}), sim_time=False).stalled() is False


# ── Pacing work on the robot's clock ─────────────────────────────────────────


def test_off_sim_time_pacing_is_a_plain_sleep():
    clock = TaskClock(_node({"t": 0.0}), sim_time=False)
    start = time.monotonic()
    assert clock.sleep_until(clock() + 0.05) is True
    assert time.monotonic() - start >= 0.045


def test_pacing_waits_for_sim_time_to_reach_the_deadline():
    """A slow sim is waited out: the deadline is in robot seconds, not wall."""
    import threading

    ros = {"t": 10.0}
    clock = TaskClock(_node(ros), sim_time=True)

    def advance():
        for _ in range(10):
            time.sleep(0.01)
            ros["t"] += 0.1

    threading.Thread(target=advance, daemon=True).start()
    assert clock.sleep_until(10.5) is True
    assert ros["t"] >= 10.5


def test_pacing_gives_up_on_a_dead_clock_instead_of_hanging():
    """A stopped sim must not wedge the inference thread forever."""
    clock = TaskClock(_node({"t": 10.0}), sim_time=True)
    start = time.monotonic()
    assert clock.sleep_until(20.0, stall_s=0.2) is False
    assert time.monotonic() - start < 2.0
