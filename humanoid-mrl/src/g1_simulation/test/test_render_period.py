# MIT License
"""Tests for the render-period check (pure logic, no Isaac import)."""

import pytest

from g1_simulation.config import IsaacSimCfg, PHYSICS_DT_S, render_period_s


def test_the_shipped_default_is_a_whole_number_of_physics_steps():
    """Whatever the default becomes, it must not split a physics step."""
    render_dt = render_period_s(IsaacSimCfg().camera_publish_hz)
    steps = render_dt / PHYSICS_DT_S
    assert steps == pytest.approx(round(steps), abs=1e-6)


def test_thirty_hz_is_legal_at_the_shipped_physics_rate():
    """30 Hz (head d435 at 640x480, and RECORD_FPS) must divide the physics rate."""
    assert render_period_s(30.0) == pytest.approx(10 * PHYSICS_DT_S)


@pytest.mark.parametrize(
    "hz, steps",
    [(100.0, 3), (75.0, 4), (50.0, 6), (30.0, 10), (25.0, 12), (15.0, 20), (6.0, 50)],
)
def test_rates_that_divide_physics_evenly_are_accepted(hz, steps):
    assert render_period_s(hz) == pytest.approx(steps * PHYSICS_DT_S)


def test_forty_hz_is_rejected_now_that_physics_runs_at_300():
    """40 Hz is illegal: 25 ms is 7.5 physics steps, so image stamps would drift."""
    with pytest.raises(ValueError, match="not a whole number"):
        render_period_s(40.0)


def test_the_rejection_names_the_drift_and_a_way_out():
    """The error message names the drift and a valid rate."""
    with pytest.raises(ValueError) as exc:
        render_period_s(40.0)
    msg = str(exc.value)
    assert "6.7%" in msg, "the drift the rate would cause"
    assert "30" in msg, "a valid rate to use instead"


@pytest.mark.parametrize("hz", [0.0, -1.0])
def test_non_positive_rates_are_rejected(hz):
    with pytest.raises(ValueError, match="must be > 0"):
        render_period_s(hz)


def test_the_constraint_is_the_ratio_not_either_number():
    """Each rate is legal under some physics step, but not both under one."""
    # 30 Hz, illegal at 1/200 (6.667 steps), is 8 steps at 1/240.
    assert render_period_s(30.0, 1 / 240) == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match="not a whole number"):
        render_period_s(30.0, 1 / 200)
    # 40 Hz, illegal at the current 1/300, is 5 steps at 1/200.
    assert render_period_s(40.0, 1 / 200) == pytest.approx(1 / 40)
