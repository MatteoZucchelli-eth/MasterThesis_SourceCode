# MIT License
"""Unit tests for the slew-limited target correction."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from g1_control.utils.target_correction import TargetCorrection


def _correction(**overrides) -> TargetCorrection:
    kwargs = dict(
        max_rate_m_s=0.005,
        max_pos_m=0.030,
        max_rate_rad_s=np.deg2rad(3.0),
        max_ang_rad=np.deg2rad(3.0),
    )
    kwargs.update(overrides)
    return TargetCorrection(**kwargs)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> pin.SE3:
    return pin.SE3(np.eye(3), np.array([x, y, z]))


def test_identity_before_any_reference() -> None:
    """A correction that was never anchored must not move anything."""
    correction = _correction()
    pose = pin.SE3(pin.exp3(np.array([0.1, 0.2, 0.3])), np.array([1.0, 2.0, 3.0]))
    applied = correction.apply(pose)
    assert np.allclose(applied.translation, pose.translation)
    assert np.allclose(applied.rotation, pose.rotation)


def test_static_target_leaves_the_command_alone() -> None:
    """The whole point: no target motion, no correction."""
    correction = _correction()
    correction.reset(_pose(0.6, 0.0, 0.0))
    for _ in range(100):
        correction.update(_pose(0.6, 0.0, 0.0), 0.02)
    assert correction.applied == pytest.approx((0.0, 0.0), abs=1e-9)
    assert np.allclose(correction.apply(_pose(0.5)).translation, [0.5, 0.0, 0.0])


def test_converges_to_the_offset_the_target_moved_by() -> None:
    """15 mm of drift is eventually applied in full, and displaces the command."""
    correction = _correction()
    correction.reset(_pose(0.600))
    for _ in range(500):  # 10 s at 50 Hz; 5 mm/s needs 3 s for 15 mm
        correction.update(_pose(0.615), 0.02)

    assert correction.applied[0] == pytest.approx(15.0, abs=1e-6)
    assert correction.raw[0] == pytest.approx(15.0, abs=1e-6)
    moved = correction.apply(_pose(0.400, 0.1, 0.2))
    assert np.allclose(moved.translation, [0.415, 0.1, 0.2])


def test_slew_limit_caps_how_fast_the_command_may_move() -> None:
    """A step in the estimate must not become a step in the command."""
    correction = _correction(max_rate_m_s=0.005)
    correction.reset(_pose(0.600))

    correction.update(_pose(0.900), 0.02)  # 300 mm step in one tick
    assert correction.applied[0] == pytest.approx(0.1, abs=1e-9)  # 5 mm/s * 0.02 s

    correction.update(_pose(0.900), 0.02)
    assert correction.applied[0] == pytest.approx(0.2, abs=1e-9)


def test_position_clamp_bounds_the_total_correction() -> None:
    """Lost tracking saturates rather than flying the arm somewhere wrong."""
    correction = _correction(max_pos_m=0.030)
    correction.reset(_pose(0.600))
    for _ in range(5000):
        correction.update(_pose(1.600), 0.02)

    assert correction.applied[0] == pytest.approx(30.0, abs=1e-6)
    # The raw measurement is left untouched, so the log can say how far off it was.
    assert correction.raw[0] == pytest.approx(1000.0, abs=1e-6)


def test_rotation_is_corrected_and_clamped() -> None:
    """The wall normal feeds the drill alignment, so orientation is tracked too."""
    correction = _correction(max_ang_rad=np.deg2rad(3.0))
    reference = pin.SE3(np.eye(3), np.zeros(3))
    turned = pin.SE3(pin.exp3(np.array([0.0, 0.0, np.deg2rad(10.0)])), np.zeros(3))

    correction.reset(reference)
    for _ in range(5000):
        correction.update(turned, 0.02)

    assert correction.applied[1] == pytest.approx(3.0, abs=1e-6)
    assert correction.raw[1] == pytest.approx(10.0, abs=1e-6)
    # Applied on the left, in the shared (pelvis) frame: the goal turns the same
    # way the target did.
    rotated = correction.apply(pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])))
    assert np.degrees(np.linalg.norm(pin.log3(rotated.rotation))) == pytest.approx(
        3.0, abs=1e-6
    )


def test_reset_drops_the_accumulated_correction() -> None:
    """Each plan re-anchors, so a new episode never inherits the last one's offset."""
    correction = _correction()
    correction.reset(_pose(0.600))
    for _ in range(500):
        correction.update(_pose(0.615), 0.02)
    assert correction.applied[0] > 1.0

    correction.reset(_pose(0.615))
    assert correction.applied == pytest.approx((0.0, 0.0), abs=1e-9)


def test_a_tick_without_an_estimate_holds_the_correction() -> None:
    """A dropped mocap frame must hold, not snap back or extrapolate."""
    correction = _correction()
    correction.reset(_pose(0.600))
    for _ in range(50):
        correction.update(_pose(0.615), 0.02)
    held = correction.applied

    # Caller skips update() when the TF is missing; time passing changes nothing.
    assert correction.applied == pytest.approx(held, abs=1e-12)


def test_zero_dt_does_not_advance() -> None:
    """The first tick of an approach has no dt to slew over."""
    correction = _correction()
    correction.reset(_pose(0.600))
    correction.update(_pose(0.615), 0.0)
    assert correction.applied == pytest.approx((0.0, 0.0), abs=1e-9)
