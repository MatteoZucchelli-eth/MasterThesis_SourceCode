# MIT License
"""Unit tests for the VLA_PLANNER pure primitives (numpy only, no ROS)."""

import numpy as np
import pytest

from g1_control.utils.tip_trajectory import aligned_drill_rotation
from g1_control.vla.planner import (
    DEPTH_SOURCES,
    DepthBiasedSource,
    drill_entry_point,
    finalize_knots,
    wall_normal_of,
    FinalizeResidual,
    TargetPoseAccumulator,
    VlaPlannerConfig,
)


def _pose_9d(position, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return np.concatenate([position, rotation[:, 0], rotation[:, 1]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _anchored(pred_position, seq, anchor_9d=None, pred_rotation=None):
    anchor_9d = _pose_9d(np.zeros(3)) if anchor_9d is None else anchor_9d
    return (
        _pose_9d(np.asarray(pred_position, dtype=float), pred_rotation),
        anchor_9d,
        0.0,
        seq,
    )


# ── TargetPoseAccumulator ────────────────────────────────────────────────────


def test_accumulator_anchors_predictions_into_the_static_frame():
    acc = TargetPoseAccumulator()
    # Tip at x=0.3 rotated 90 deg about z: a target "0.1 ahead of the tip"
    # (tip frame +x) lands at tip + 0.1 * (pelvis +y).
    anchor = _pose_9d(np.array([0.3, 0.0, 0.0]), _rot_z(np.pi / 2))
    assert acc.add(_anchored([0.1, 0.0, 0.0], seq=1, anchor_9d=anchor))
    np.testing.assert_allclose(acc.position(), [0.3, 0.1, 0.0], atol=1e-12)


def test_accumulator_deduplicates_by_sequence_number():
    acc = TargetPoseAccumulator()
    sample = _anchored([0.1, 0.0, 0.0], seq=1)
    assert acc.add(sample)
    assert not acc.add(sample)
    assert not acc.add(None)
    assert acc.count == 1


def test_accumulator_mean_and_spread():
    acc = TargetPoseAccumulator()
    acc.add(_anchored([0.10, 0.01, 0.0], seq=1))
    acc.add(_anchored([0.10, -0.01, 0.0], seq=2))
    np.testing.assert_allclose(acc.position(), [0.10, 0.0, 0.0], atol=1e-12)
    assert np.isclose(acc.position_spread(), 0.01)
    assert acc.rotation_spread_deg() == 0.0


def test_accumulator_rotation_spread():
    acc = TargetPoseAccumulator()
    acc.add(_anchored([0.1, 0.0, 0.0], seq=1, pred_rotation=_rot_z(0.1)))
    acc.add(_anchored([0.1, 0.0, 0.0], seq=2, pred_rotation=_rot_z(-0.1)))
    assert np.isclose(acc.rotation_spread_deg(), np.degrees(0.1), atol=0.1)


def test_accumulator_convergence_needs_samples_and_spread():
    acc = TargetPoseAccumulator()
    acc.add(_anchored([0.10, 0.0, 0.0], seq=1))
    assert not acc.converged(min_samples=2, max_spread_m=0.012)
    acc.add(_anchored([0.10, 0.001, 0.0], seq=2))
    assert acc.converged(min_samples=2, max_spread_m=0.012)
    # A wild sample blows the spread past the gate again.
    acc.add(_anchored([0.10, 0.08, 0.0], seq=3))
    assert not acc.converged(min_samples=2, max_spread_m=0.012)


def test_accumulator_empty_reports_nothing():
    acc = TargetPoseAccumulator()
    assert acc.position() is None
    assert acc.pose_9d() is None
    assert acc.position_spread() == 0.0
    assert not acc.converged(min_samples=1, max_spread_m=1.0)


def test_accumulator_pose_9d_mean_rotation_is_orthonormal():
    acc = TargetPoseAccumulator()
    acc.add(_anchored([0.1, 0.0, 0.0], seq=1, pred_rotation=_rot_z(0.2)))
    acc.add(_anchored([0.1, 0.0, 0.0], seq=2, pred_rotation=_rot_z(0.4)))
    pose = acc.pose_9d()
    x, y = pose[3:6], pose[6:9]
    assert np.isclose(np.linalg.norm(x), 1.0)
    assert np.isclose(np.linalg.norm(y), 1.0)
    assert np.isclose(np.dot(x, y), 0.0, atol=1e-12)
    # Mean of ±0.1 rad about the midpoint: x-axis of rot_z(0.3).
    np.testing.assert_allclose(x, _rot_z(0.3)[:, 0], atol=1e-3)


# ── DepthBiasedSource ────────────────────────────────────────────────────────


class _FakeFine:
    """The two accessors the state reads a fine estimate through."""

    def __init__(self, anchored):
        self._anchored = anchored
        self.detection_count = 7

    def get_tip_target(self):
        return self._anchored[0]

    def get_tip_target_anchored(self):
        return self._anchored


def test_depth_bias_moves_the_target_along_its_own_ray():
    """Same pixel, more depth: the bearing is untouched.

    Camera 100 mm behind the tip along x: tip-frame (-0.06, 0, 0.10) is camera
    (0.04, 0, 0.10); 15 mm of bias scales that ray by 1.15.
    """
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [-0.1, 0.0, 0.0]
    biased = DepthBiasedSource(
        _FakeFine(_anchored([-0.06, 0.0, 0.10], seq=3)), 0.015, lambda: extrinsic
    )
    pred_9d, _anchor, _stamp, seq = biased.get_tip_target_anchored()
    np.testing.assert_allclose(pred_9d[:3], [0.046 - 0.1, 0.0, 0.115], atol=1e-12)
    # The camera-frame point stays on its ray.
    np.testing.assert_allclose(
        np.cross(pred_9d[:3] - extrinsic[:3, 3], [0.04, 0.0, 0.10]),
        np.zeros(3),
        atol=1e-12,
    )
    assert seq == 3
    # Non-range accessors pass through.
    assert biased.detection_count == 7
    np.testing.assert_allclose(biased.get_tip_target()[:3], pred_9d[:3], atol=1e-12)


def test_depth_bias_waits_for_the_extrinsic_and_leaves_bad_depths_alone():
    """No extrinsic or a point behind the lens: the estimate passes through
    unchanged."""
    fine = _FakeFine(_anchored([-0.06, 0.0, 0.10], seq=1))
    np.testing.assert_allclose(
        DepthBiasedSource(fine, 0.015, lambda: None).get_tip_target_anchored()[0],
        fine.get_tip_target_anchored()[0],
    )
    behind = _FakeFine(_anchored([0.0, 0.0, -0.10], seq=1))
    np.testing.assert_allclose(
        DepthBiasedSource(behind, 0.015, lambda: np.eye(4))
        .get_tip_target_anchored()[0],
        behind.get_tip_target_anchored()[0],
    )


def test_the_bias_is_a_ros_parameter_with_no_bias_as_its_default():
    """Fields become ``vla_planner_<field>`` parameters; the default applies no bias."""
    assert VlaPlannerConfig().fine_depth_bias_mm == 0.0


# ── finalize_knots ───────────────────────────────────────────────────────────


def test_finalize_knots_places_standoff_and_aim_along_the_wall_normal():
    tip = np.array([0.2, 0.0, 0.1])
    estimate = _pose_9d(np.array([0.6, 0.1, 0.2]))  # wall normal = pelvis +x
    knots = finalize_knots(tip, estimate, standoff_m=0.075, wall_offset_m=0.005)
    np.testing.assert_allclose(knots[0], tip)
    np.testing.assert_allclose(knots[1], [0.525, 0.1, 0.2])
    np.testing.assert_allclose(knots[2], [0.595, 0.1, 0.2])


def test_finalize_knots_follows_the_estimated_normal():
    tip = np.zeros(3)
    rotation = _rot_z(np.pi / 2)  # wall normal = pelvis +y
    estimate = _pose_9d(np.array([0.0, 0.5, 0.0]), rotation)
    knots = finalize_knots(tip, estimate, standoff_m=0.1, wall_offset_m=0.01)
    np.testing.assert_allclose(knots[1], [0.0, 0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(knots[2], [0.0, 0.49, 0.0], atol=1e-12)


def test_finalize_knots_applies_all_three_wall_local_target_offsets():
    """Normal, right and up controls translate the aim in wall coordinates."""
    tip = np.zeros(3)
    target = np.array([0.6, 0.1, 0.2])
    estimate = _pose_9d(target)  # normal +x, wall-lateral +y, wall-up +z
    knots = finalize_knots(
        tip,
        estimate,
        standoff_m=0.075,
        wall_offset_m=0.005,
        wall_lateral_offset_m=0.012,
        wall_up_offset_m=-0.008,
    )
    np.testing.assert_allclose(knots[1], [0.525, 0.112, 0.192])
    np.testing.assert_allclose(knots[2], [0.595, 0.112, 0.192])


def test_in_plane_target_offsets_follow_a_rotated_wall():
    rotation = _rot_z(np.pi / 2)  # normal +y, wall-lateral -x, wall-up +z
    estimate = _pose_9d(np.array([0.0, 0.5, 0.2]), rotation)
    knots = finalize_knots(
        np.zeros(3),
        estimate,
        standoff_m=0.1,
        wall_offset_m=0.01,
        wall_lateral_offset_m=0.02,
        wall_up_offset_m=0.03,
    )
    np.testing.assert_allclose(knots[1], [-0.02, 0.4, 0.23], atol=1e-12)
    np.testing.assert_allclose(knots[2], [-0.02, 0.49, 0.23], atol=1e-12)


def test_a_raised_standoff_lifts_the_knot_without_backing_it_off_the_wall():
    """The rise is in the wall plane, so the knot stays ``standoff_m`` off the wall."""
    tip = np.zeros(3)
    target = np.array([0.6, 0.1, 0.2])
    estimate = _pose_9d(target)  # wall normal = pelvis +x
    knots = finalize_knots(
        tip, estimate, standoff_m=0.025, wall_offset_m=0.0, rise_m=0.005
    )
    np.testing.assert_allclose(knots[1], [0.575, 0.1, 0.205], atol=1e-12)
    np.testing.assert_allclose(knots[2], target, atol=1e-12)
    normal = wall_normal_of(estimate)
    assert float(np.dot(target - knots[1], normal)) == pytest.approx(0.025)


def test_the_rise_is_up_the_wall_not_up_the_world():
    """On a tilted wall the rise is along the surface, not world +z."""
    tilt = np.pi / 6
    # Normal tipped up out of horizontal by `tilt`, so the wall leans back.
    rotation = np.array(
        [
            [np.cos(tilt), 0.0, -np.sin(tilt)],
            [0.0, 1.0, 0.0],
            [np.sin(tilt), 0.0, np.cos(tilt)],
        ]
    )
    target = np.array([0.6, 0.0, 0.2])
    estimate = _pose_9d(target, rotation)
    knots = finalize_knots(
        np.zeros(3), estimate, standoff_m=0.03, wall_offset_m=0.0, rise_m=0.01
    )
    normal = wall_normal_of(estimate)
    rise = knots[1] - (target - 0.03 * normal)
    assert float(np.linalg.norm(rise)) == pytest.approx(0.01)
    assert float(np.dot(rise, normal)) == pytest.approx(0.0, abs=1e-12)
    assert rise[2] > 0.0, "still upward, just along the surface"


def test_a_horizontal_wall_falls_back_to_the_estimates_own_axis():
    """On a horizontal wall, pelvis +z has no in-plane projection; the estimate's z-axis
    is used."""
    # Columns: normal = +z, then +x, then their cross product +y.
    rotation = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    target = np.array([0.4, 0.0, 0.9])
    estimate = _pose_9d(target, rotation)
    rise = finalize_knots(
        np.zeros(3), estimate, standoff_m=0.03, wall_offset_m=0.0, rise_m=0.01
    )[1] - (target - 0.03 * wall_normal_of(estimate))
    np.testing.assert_allclose(rise, 0.01 * rotation[:, 2], atol=1e-12)


def test_a_zero_rise_is_the_plain_straight_in_standoff():
    tip = np.zeros(3)
    estimate = _pose_9d(np.array([0.6, 0.1, 0.2]))
    np.testing.assert_allclose(
        finalize_knots(tip, estimate, 0.05, 0.0, rise_m=0.0)[1],
        finalize_knots(tip, estimate, 0.05, 0.0)[1],
    )


def test_the_wall_normal_comes_off_the_estimate_not_the_path():
    """The drill axis follows the estimate's normal, not the oblique standoff -> aim
    chord."""
    rotation = _rot_z(np.pi / 2)
    estimate = _pose_9d(np.array([0.0, 0.5, 0.0]), rotation)
    np.testing.assert_allclose(wall_normal_of(estimate), rotation[:, 0], atol=1e-12)

    knots = finalize_knots(
        np.zeros(3), estimate, standoff_m=0.05, wall_offset_m=0.0, rise_m=0.01
    )
    chord = knots[2] - knots[1]
    chord /= np.linalg.norm(chord)
    assert float(np.dot(chord, wall_normal_of(estimate))) < 0.99, "obliquely"


def test_aligned_rotation_aligns_drill_axis():
    axis = np.array([0.3, -0.1, 0.1])
    rotation = aligned_drill_rotation(axis, np.eye(3), blend=1.0)
    np.testing.assert_allclose(rotation[:, 0], axis / np.linalg.norm(axis))
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_aligned_rotation_blend_fades_from_the_reference():
    axis = np.array([0.0, 1.0, 0.0])  # 90 deg off the reference x-axis
    r0 = aligned_drill_rotation(axis, np.eye(3), blend=0.0)
    np.testing.assert_allclose(r0[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    r_half = aligned_drill_rotation(axis, np.eye(3), blend=0.5)
    # Halfway through a 90 deg turn: 45 deg from both.
    assert np.isclose(np.degrees(np.arccos(r_half[0, 0])), 45.0)


def test_aligned_rotation_keeps_roll_anchored():
    # Axis along the reference's own x-axis: nothing to turn, frame unchanged.
    reference = _rot_z(0.3)
    rotation = aligned_drill_rotation(reference[:, 0], reference, blend=1.0)
    np.testing.assert_allclose(rotation, reference, atol=1e-12)


def test_accumulator_rejects_samples_that_do_not_fit_the_cluster():
    """With the arm frozen, a reading 200 mm from the cluster is an outlier."""
    accumulator = TargetPoseAccumulator(max_outlier_m=0.05)
    for seq in range(1, 4):
        assert accumulator.add(_anchored([0.30, 0.0, 0.0], seq=seq))
    assert not accumulator.add(_anchored([0.30, 0.20, 0.0], seq=4))
    assert accumulator.count == 3 and accumulator.rejected == 1
    np.testing.assert_allclose(accumulator.position(), [0.30, 0.0, 0.0], atol=1e-12)


def test_accumulator_needs_a_cluster_before_it_rejects_anything():
    """With one or two samples there is nothing to be an outlier from."""
    accumulator = TargetPoseAccumulator(max_outlier_m=0.05)
    assert accumulator.add(_anchored([0.30, 0.0, 0.0], seq=1))
    assert accumulator.add(_anchored([0.30, 0.50, 0.0], seq=2))
    assert accumulator.rejected == 0


def test_accumulator_outlier_reference_is_the_median_not_the_mean():
    """The median is robust to the outliers being excluded."""
    accumulator = TargetPoseAccumulator(max_outlier_m=0.05)
    for seq, y in enumerate([0.0, 0.30, 0.0, 0.0], start=1):
        accumulator.add(_anchored([0.30, y, 0.0], seq=seq))
    # A 300 mm outlier admitted before the cluster formed does not shift the median.
    assert accumulator.add(_anchored([0.30, 0.001, 0.0], seq=9))
    # A second far reading is still rejected.
    assert not accumulator.add(_anchored([0.30, 0.20, 0.0], seq=10))


def test_accumulator_exposes_the_sequence_it_stopped_at():
    """Lets the live estimator skip the batch's last sample."""
    accumulator = TargetPoseAccumulator()
    assert accumulator.last_seq is None
    accumulator.add(_anchored([0.1, 0.0, 0.0], seq=4))
    accumulator.add(_anchored([0.1, 0.0, 0.0], seq=5))
    assert accumulator.last_seq == 5
    accumulator.reset()
    assert accumulator.last_seq is None


def test_config_defaults_are_consistent():
    cfg = VlaPlannerConfig()
    # Approach gate: above wall offset + standoff, below the camera's trusted range.
    assert cfg.finalize_wall_offset_m < cfg.finalize_standoff_m
    assert cfg.finalize_standoff_m <= cfg.approach_distance * 2
    assert cfg.accumulate_min_samples >= 1
    assert cfg.accumulate_settle_tol_mm == pytest.approx(2.0)
    assert cfg.accumulate_settle_hold_s > 0.0
    # Settle floor must leave room inside the timeout.
    assert cfg.accumulate_settle_min_s < cfg.accumulate_settle_timeout_s
    # The accumulator must be able to reject, not only time out.
    assert cfg.accumulate_outlier_mm > cfg.accumulate_max_spread_mm
    # Live correction is off by default.
    assert cfg.finalize_correct_target is False
    # Clamp covers the viewpoint bias and is reachable within the finalize.
    assert cfg.finalize_correction_max_mm >= cfg.accumulate_max_spread_mm
    reachable_mm = cfg.finalize_correction_rate_mm_s * (
        cfg.finalize_transit_duration_s + cfg.finalize_insert_duration_s
    )
    assert reachable_mm > cfg.finalize_correction_max_mm
    # Detections arrive at ~1 Hz.
    assert cfg.finalize_correction_max_age_s > 1.0
    assert 0.0 < cfg.finalize_correction_ema_alpha <= 1.0
    assert cfg.depth_source in DEPTH_SOURCES
    # The driver replans at ~1 s.
    assert cfg.depth_max_age_s > 1.0
    # Accept the driver's range error, reject the far mode.
    assert cfg.depth_max_shift_mm > cfg.accumulate_max_spread_mm
    assert cfg.depth_max_shift_mm < cfg.approach_distance * 1000.0


def test_every_config_attribute_the_state_reads_actually_exists():
    """Every ``config.<field>`` / ``self._cfg.<field>`` read in vla_planner exists on
    the dataclass."""
    import ast
    import inspect

    from g1_control.states import vla_planner

    source = inspect.getsource(vla_planner)
    fields = set(vars(VlaPlannerConfig()))
    read = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and (
            (isinstance(node.value, ast.Name) and node.value.id == "config")
            or (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "_cfg"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
            )
        )
    }
    assert read, "found no config reads — the AST shapes above must have drifted"
    missing = read - fields
    assert not missing, f"vla_planner reads config fields that do not exist: {missing}"


# ── The finalize correction only listens to plausible predictions ────────────


def test_the_correction_gate_sits_above_what_it_legitimately_tracks():
    """The gate admits the correctable bias and the clamp range."""
    cfg = VlaPlannerConfig()
    assert cfg.finalize_correction_gate_mm > cfg.finalize_correction_max_mm
    assert cfg.finalize_correction_gate_mm > cfg.accumulate_max_spread_mm
    # Far mode is hundreds of mm.
    assert cfg.finalize_correction_gate_mm < 150.0


# ── The miss a finalize achieved, split about the drill axis ─────────────────


def test_the_two_halves_of_a_miss_do_not_cancel():
    """5 mm lateral and 4 mm too deep must not score as a ~0 mm plain distance."""
    offset = 0.005
    residual = FinalizeResidual.of([offset - 0.004, 0.005, 0.0], offset)
    naive = np.linalg.norm([offset - 0.004, 0.005, 0.0]) - offset
    assert abs(naive) < 0.0002, "a plain distance sees nothing wrong"
    assert abs(residual.lateral_m - 0.005) < 1e-9
    assert abs(residual.along_m + 0.004) < 1e-9


def test_a_flawless_round_reads_zero_on_both_axes():
    """Landing ``finalize_wall_offset_m`` short is on target."""
    offset = 0.005
    residual = FinalizeResidual.of([offset, 0.0, 0.0], offset)
    assert residual.lateral_m == 0.0 and residual.along_m == 0.0
    # The raw distance still includes the offset.
    assert abs(residual.distance_m - offset) < 1e-9


def test_the_sign_along_the_axis_says_which_way_the_tip_missed():
    offset = 0.005
    short = FinalizeResidual.of([offset + 0.02, 0.0, 0.0], offset)
    past = FinalizeResidual.of([offset - 0.02, 0.0, 0.0], offset)
    assert short.along_m > 0.0, "positive = still short of the wall"
    assert past.along_m < 0.0, "negative = at it, or into it"


def test_lateral_is_measured_across_both_axes_perpendicular_to_the_bit():
    residual = FinalizeResidual.of([0.005, 0.003, 0.004], 0.005)
    assert abs(residual.lateral_m - 0.005) < 1e-9


def test_deliberate_in_plane_offsets_are_not_scored_as_a_miss():
    desired = np.array([0.005, -0.003, 0.004])
    residual = FinalizeResidual.of(
        desired,
        wall_offset_m=0.005,
        desired_target_tip=desired,
    )
    assert residual.lateral_m == pytest.approx(0.0)
    assert residual.along_m == pytest.approx(0.0)


def test_the_pixel_figure_is_optional_because_the_camera_geometry_can_be_absent():
    assert FinalizeResidual.of([0.005, 0.0, 0.0], 0.005).lateral_px is None
    with_px = FinalizeResidual.of([0.005, 0.0, 0.0], 0.005, lateral_px=3.2)
    assert with_px.lateral_px == 3.2


def test_the_entry_point_is_the_targets_station_on_the_drill_axis():
    """The target's projection onto the drill axis."""
    assert np.allclose(drill_entry_point([0.005, 0.002, -0.003]), [0.005, 0.0, 0.0])
