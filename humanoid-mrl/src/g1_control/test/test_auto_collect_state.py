# MIT License
"""Tests for the AUTO_COLLECT state's trajectory generation and IK precision."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from g1_control.utils.task_clock import TaskClock

from g1_control.utils.tip_trajectory import (
    aligned_drill_rotation as _aligned_drill_rotation,
    CubicBezier as _CubicBezier,
    levelled_drill_rotation as _levelled_drill_rotation,
    quintic as _quintic,
    quintic_hermite as _quintic_hermite,
    TipTrajectory as _TipTrajectory,
)

from g1_definitions.definitions import AutoCollectPhase

# Pelvis-frame geometry of the Isaac scene: the pelvis is pinned at
# IsaacSimCfg.reset_position and the target squares sit on the wall face.
_WALL_X = 3.309 - 0.001 / 2
_PELVIS = np.array([2.80, 0.0, 0.7283520102500916])

# The uniform box IsaacSimCfg samples the squares from.
_TARGET_Y_RANGE = (-0.32, -0.02)
_TARGET_Z_RANGE = (0.88, 1.10)

_STANDOFF_M = 0.12
_TRANSIT_DURATION_S = 8.0
_INSERT_DURATION_S = 7.0

# Wall faces the robot square-on: target normal = pelvis +x, up = pelvis +z.
_WALL_NORMAL = np.array([1.0, 0.0, 0.0])
_WALL_UP = np.array([0.0, 0.0, 1.0])


# ``drill_robot_model`` is a session-scoped fixture in conftest.py.


# ── Phase enum ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("approaching", AutoCollectPhase.APPROACHING),
        ("APPROACHING", AutoCollectPhase.APPROACHING),
        (" Holding ", AutoCollectPhase.HOLDING),
        ("trajectory_generation", AutoCollectPhase.TRAJECTORY_GENERATION),
        ("2", AutoCollectPhase.GOING_TO_DEFAULT),
    ],
)
def test_phase_parse_accepts_names_and_values(text, expected):
    assert AutoCollectPhase.parse(text) == expected


@pytest.mark.parametrize("text", ["", "sprinting", "9"])
def test_phase_parse_rejects_unknown(text):
    with pytest.raises((KeyError, ValueError)):
        AutoCollectPhase.parse(text)


# ── Time scaling / trajectory ─────────────────────────────────────────────────


def test_quintic_is_a_minimum_jerk_scaling():
    assert _quintic(0.0) == pytest.approx(0.0)
    assert _quintic(1.0) == pytest.approx(1.0)
    assert _quintic(0.5) == pytest.approx(0.5)
    # Zero velocity at both ends.
    eps = 1e-6
    assert _quintic(eps) / eps == pytest.approx(0.0, abs=1e-8)
    assert (1.0 - _quintic(1.0 - eps)) / eps == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("tau", [0.0, 0.13, 0.5, 0.87, 1.0])
def test_quintic_hermite_reduces_to_the_plain_quintic_at_zero_endpoint_speed(tau):
    assert _quintic_hermite(tau, 1.0, 0.0, 0.0) == pytest.approx(_quintic(tau))


def test_quintic_hermite_honours_its_endpoint_speeds():
    v0, v1 = 0.3, 0.7
    h = 1e-6
    assert _quintic_hermite(0.0, 2.0, v0, v1) == pytest.approx(0.0)
    assert _quintic_hermite(1.0, 2.0, v0, v1) == pytest.approx(2.0)
    slope0 = (_quintic_hermite(h, 2.0, v0, v1) - 0.0) / h
    slope1 = (2.0 - _quintic_hermite(1.0 - h, 2.0, v0, v1)) / h
    assert slope0 == pytest.approx(v0, abs=1e-4)
    assert slope1 == pytest.approx(v1, abs=1e-4)


def _straight_trajectory(segment_durations=(6.0, 4.0)):
    knots = [np.zeros(3), np.array([0.4, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]
    return _TipTrajectory(knots, np.eye(3), list(segment_durations))


def test_trajectory_starts_and_ends_exactly_on_its_knots():
    traj = _straight_trajectory()
    assert traj.length == pytest.approx(0.5)
    np.testing.assert_allclose(traj.pose_at(0.0).translation, traj.knots[0], atol=1e-12)
    # Any residual here would be a permanent tip-to-target offset.
    end = traj.pose_at(1.0).translation
    np.testing.assert_allclose(end, traj.knots[-1], atol=1e-12)


def test_trajectory_clamps_out_of_range_time():
    traj = _straight_trajectory()
    np.testing.assert_allclose(traj.pose_at(-3.0).translation, traj.knots[0])
    np.testing.assert_allclose(traj.pose_at(7.5).translation, traj.knots[-1])


def test_trajectory_is_monotonic_along_the_path():
    traj = _straight_trajectory()
    xs = [traj.pose_at(u).translation[0] for u in np.linspace(0.0, 1.0, 50)]
    assert np.all(np.diff(xs) >= -1e-12)


def test_trajectory_holds_orientation_constant():
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    traj = _TipTrajectory([np.zeros(3), np.array([0.3, 0.0, 0.0])], rotation, [5.0])
    for u in (0.0, 0.37, 1.0):
        np.testing.assert_allclose(traj.pose_at(u).rotation, rotation, atol=1e-12)


def test_degenerate_trajectory_stays_on_the_target():
    """A plan generated with the tip already on target must not divide by zero."""
    point = np.array([0.5, -0.2, 0.2])
    traj = _TipTrajectory([point, point], np.eye(3), [3.0])
    assert traj.length == pytest.approx(0.0)
    np.testing.assert_allclose(traj.pose_at(0.5).translation, point)


def test_zero_duration_trajectory_stays_on_the_target():
    traj = _TipTrajectory([np.zeros(3), np.array([0.1, 0.0, 0.0])], np.eye(3), [0.0])
    np.testing.assert_allclose(traj.pose_at(0.5).translation, traj.knots[-1])


def test_segment_durations_must_match_knots():
    with pytest.raises(ValueError):
        _TipTrajectory([np.zeros(3), np.ones(3), 2 * np.ones(3)], np.eye(3), [1.0])


def test_each_segment_gets_its_own_time_budget():
    """u is normalised over total time, so a knot sits at its cumulative duration."""
    traj = _straight_trajectory((6.0, 4.0))
    assert traj.duration == pytest.approx(10.0)
    # The standoff knot is reached exactly when the transit leg's time is up.
    np.testing.assert_allclose(traj.pose_at(6.0 / 10.0).translation, traj.knots[1])


def _speed_profile(traj, samples=4001):
    us = np.linspace(0.0, 1.0, samples)
    pos = np.array([traj.pose_at(u).translation for u in us])
    dt = (us[1] - us[0]) * traj.duration
    return us[:-1], np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt


def test_tip_is_at_rest_only_at_the_endpoints():
    """Zero speed at start and end so the arm eases in and settles; not at the knot."""
    traj = _straight_trajectory((6.0, 4.0))
    _, speed = _speed_profile(traj)
    peak = speed.max()
    assert speed[0] < 0.02 * peak
    assert speed[-1] < 0.02 * peak


def test_tip_does_not_stop_at_the_standoff():
    """The tip flows through the interior knot instead of stopping on it."""
    traj = _corner_trajectory()
    us, speed = _speed_profile(traj)
    u_knot = traj.segment_durations[0] / traj.duration
    at_knot = speed[np.argmin(np.abs(us - u_knot))]

    assert at_knot > 0.2 * speed.max(), f"tip stalls at the standoff ({at_knot} m/s)"
    # Knot speed equals the insertion leg's mean speed.
    insertion_mean = traj.segments[1].length / traj.segment_durations[1]
    assert at_knot == pytest.approx(traj.knot_speeds[1], rel=0.02)
    assert at_knot <= insertion_mean * 1.02


def test_speed_is_continuous_across_the_standoff():
    """C1 in time: no step in |v| where the two Hermite schedules meet."""
    traj = _corner_trajectory()
    us, speed = _speed_profile(traj, samples=20001)
    u_knot = traj.segment_durations[0] / traj.duration
    k = int(np.argmin(np.abs(us - u_knot)))
    before, after = speed[k - 3], speed[k + 3]
    assert abs(after - before) < 0.01 * speed.max(), (before, after)


def test_residual_curvature_step_at_the_knot_stays_negligible():
    """The path is G1, not G2: bound the centripetal step v^2*kappa at the standoff."""
    traj = _corner_trajectory()
    transit = traj.segments[0]
    v_knot = traj.knot_speeds[1]

    p = transit.control_points
    numerator = np.linalg.norm(np.cross(p[3] - p[2], p[1] - p[2]))
    kappa_end = (2.0 / 3.0) * numerator / np.linalg.norm(p[3] - p[2]) ** 3
    # The straight insertion has zero curvature, so this is the whole jump.
    centripetal_step = v_knot**2 * kappa_end
    assert centripetal_step < 5e-3, f"{centripetal_step * 1e3:.2f} mm/s^2"


def test_tangential_acceleration_is_continuous_across_the_standoff():
    """Both Hermite schedules pin zero acceleration at their shared endpoint."""
    traj = _corner_trajectory()
    t0, t1 = traj.segment_durations
    v_knot = traj.knot_speeds[1]
    h = 1e-5

    def accel(length, v0, v1, span, tau):
        def f(t):
            return _quintic_hermite(t, length, v0, v1)

        return (f(tau + h) - 2 * f(tau) + f(tau - h)) / h**2 / span**2

    before = accel(traj.segments[0].length, 0.0, v_knot * t0, t0, 1.0 - h)
    after = accel(traj.segments[1].length, v_knot * t1, 0.0, t1, h)
    assert abs(before) < 1e-5 and abs(after) < 1e-5, (before, after)


def test_arclength_schedule_is_monotone():
    """The Hermite knot velocity must never make the tip back up along the path."""
    for durations in ((6.0, 4.0), (8.0, 12.0), (12.0, 2.0), (1.0, 20.0)):
        traj = _TipTrajectory(
            [np.zeros(3), np.array([0.4, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
            np.eye(3),
            list(durations),
        )
        xs = np.array([traj.pose_at(u).translation[0] for u in np.linspace(0, 1, 3000)])
        assert np.all(np.diff(xs) >= -1e-12), f"non-monotone for {durations}"


# ── Tangent continuity (G1) ───────────────────────────────────────────────────


def _tangent_angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _corner_trajectory():
    """An L-shaped plan: the standoff turns the path hard into the insertion axis."""
    tip0 = np.array([0.45, -0.19, 0.10])
    standoff = np.array([0.41, -0.21, 0.24])
    target = np.array([0.51, -0.21, 0.24])
    return _TipTrajectory([tip0, standoff, target], np.eye(3), [8.0, 8.0])


def test_path_is_tangent_continuous_at_the_standoff():
    """The transit arrives at the standoff along the insertion axis (no kink)."""
    traj = _corner_trajectory()
    transit, insertion = traj.segments

    exit_dir = transit.tangent_at(1.0)
    entry_dir = insertion.tangent_at(0.0)
    assert _tangent_angle_deg(exit_dir, entry_dir) < 1e-6

    # ...and that shared tangent is the insertion axis itself.
    insertion_axis = traj.knots[2] - traj.knots[1]
    insertion_axis = insertion_axis / np.linalg.norm(insertion_axis)
    assert _tangent_angle_deg(exit_dir, insertion_axis) < 1e-6

    # Guard: the underlying corner is sharp.
    chord = traj.knots[1] - traj.knots[0]
    chord = chord / np.linalg.norm(chord)
    assert _tangent_angle_deg(chord, insertion_axis) > 90.0


def test_final_segment_stays_exactly_straight():
    """The straight-in push must not bow: it has no successor to bend toward."""
    traj = _corner_trajectory()
    standoff, target = traj.knots[1], traj.knots[2]
    insertion = traj.segments[-1]

    fractions = np.linspace(0.0, 1.0, 201)
    points = np.array([insertion.point_at_arclength_fraction(f) for f in fractions])
    ideal = standoff + np.outer(fractions, target - standoff)
    np.testing.assert_allclose(points, ideal, atol=1e-9)
    assert insertion.length == pytest.approx(np.linalg.norm(target - standoff))


def test_curved_transit_bows_away_from_the_wall():
    """Bending the transit must not push the tip past the target plane (the wall)."""
    traj = _corner_trajectory()
    wall_x = traj.knots[2][0]
    points = np.array([traj.pose_at(u).translation for u in np.linspace(0, 1, 500)])
    assert points[:, 0].max() <= wall_x + 1e-9


def test_curve_still_passes_exactly_through_its_knots():
    traj = _corner_trajectory()
    np.testing.assert_allclose(traj.pose_at(0.0).translation, traj.knots[0], atol=1e-12)
    np.testing.assert_allclose(traj.pose_at(0.5).translation, traj.knots[1], atol=1e-9)
    np.testing.assert_allclose(traj.pose_at(1.0).translation, traj.knots[2], atol=1e-12)


def test_bezier_is_sampled_by_arclength_not_by_parameter():
    """Equal fractions must cover equal distance, or the tip speeds up in the bend."""
    curve = _CubicBezier(
        np.zeros(3),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.2, 0.2, 0.0]),
        np.array([0.0, 0.2, 0.0]),
    )
    fractions = np.linspace(0.0, 1.0, 401)
    points = np.array([curve.point_at_arclength_fraction(f) for f in fractions])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Uniform within 1%; the raw Bézier parameter is not.
    assert steps.std() / steps.mean() < 0.01

    raw = np.array([curve._at(s) for s in fractions])
    raw_steps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    assert raw_steps.std() / raw_steps.mean() > 0.05


def test_degenerate_bezier_has_no_length_and_holds_its_endpoint():
    point = np.array([0.3, 0.1, -0.2])
    curve = _CubicBezier(point, point, point, point)
    assert curve.length == pytest.approx(0.0)
    np.testing.assert_allclose(curve.point_at_arclength_fraction(0.4), point)
    np.testing.assert_allclose(curve.tangent_at(0.5), np.zeros(3))


# ── End-to-end precision ──────────────────────────────────────────────────────


def _replay(model, target_pelvis, steps=81, up=None):
    """Replay a planned approach as AutoCollectState does; return the final q.

    Mirrors ``_command_tip_pose`` (IK warm-started from the previous commanded q).
    With ``up`` also mirrors ``_planned_pose``'s levelled orientation, faded in
    over the transit leg.
    """
    import pinocchio as pin

    q = model.default_body_pose.copy()
    model.cache_forward_kinematics(q, auto_clip=False)
    tip0 = model.frame_placement("drill_tip_link")

    traj = _TipTrajectory(
        knots=[
            tip0.translation.copy(),
            target_pelvis - np.array([_STANDOFF_M, 0.0, 0.0]),
            target_pelvis,
        ],
        rotation=tip0.rotation.copy(),
        segment_durations=[_TRANSIT_DURATION_S, _INSERT_DURATION_S],
    )
    for u in np.linspace(0.0, 1.0, steps):
        pose = traj.pose_at(u)
        if up is not None:
            blend = _quintic(min(u * traj.duration / _TRANSIT_DURATION_S, 1.0))
            pose = pin.SE3(
                _aligned_drill_rotation(_WALL_NORMAL, tip0.rotation, blend, up=up),
                pose.translation,
            )
        q = model.solve_ik("drill_tip_link", pose, q)
    return q


def _tip_error_mm(model, q, target_pelvis):
    model.cache_forward_kinematics(q, auto_clip=False)
    achieved = model.frame_placement("drill_tip_link").translation
    return float(np.linalg.norm(achieved - target_pelvis)) * 1e3


def test_drill_points_into_the_wall_at_the_default_pose(drill_robot_model):
    """The plan reuses the default-pose tip orientation, so it must face +x."""
    model = drill_robot_model
    model.cache_forward_kinematics(model.default_body_pose.copy(), auto_clip=False)
    tip_x_axis = model.frame_placement("drill_tip_link").rotation[:, 0]
    np.testing.assert_allclose(tip_x_axis, [1.0, 0.0, 0.0], atol=1e-3)


@pytest.mark.parametrize("y", _TARGET_Y_RANGE)
@pytest.mark.parametrize("z", _TARGET_Z_RANGE)
def test_approach_lands_on_every_corner_of_the_sampling_box(drill_robot_model, y, z):
    """Each corner of IsaacSimCfg's target box is reachable sub-mm."""
    target = np.array([_WALL_X, y, z]) - _PELVIS
    q = _replay(drill_robot_model, target)
    assert _tip_error_mm(drill_robot_model, q, target) < 1.0


def test_approach_is_precise_across_the_sampling_box(drill_robot_model):
    """Uniform samples from the box (not just its corners) all land on target."""
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(6):
        y = rng.uniform(*_TARGET_Y_RANGE)
        z = rng.uniform(*_TARGET_Z_RANGE)
        target = np.array([_WALL_X, y, z]) - _PELVIS
        q = _replay(drill_robot_model, target)
        errors.append(_tip_error_mm(drill_robot_model, q, target))
    assert max(errors) < 1.0, f"worst tip error {max(errors):.2f} mm"


# ── Drill roll about the wall normal ──────────────────────────────────────────
#
# Aligning the drill axis to the wall normal leaves the roll free; these cover
# levelling it against the wall's up-direction.


def _rot_x(angle_rad):
    """Roll about the drill axis — what levelling has to take out."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _turn_deg(a, b):
    """Angle of the rotation taking ``a`` to ``b`` [deg]."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_levelled_rotation_puts_the_drill_upright_on_the_normal():
    rotation = _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP)
    np.testing.assert_allclose(rotation[:, 0], _WALL_NORMAL, atol=1e-12)
    # y perpendicular to up, z up (handle along tip -z hangs straight down).
    assert rotation[:, 1] @ _WALL_UP == pytest.approx(0.0, abs=1e-12)
    assert rotation[:, 2] @ _WALL_UP > 0.0
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_levelled_roll_does_not_depend_on_the_pose_it_started_from():
    """The jittered start pose does not set the roll."""
    levelled = [
        _aligned_drill_rotation(_WALL_NORMAL, _rot_x(angle), blend=1.0, up=_WALL_UP)
        for angle in np.radians([-40.0, 0.0, 25.0, 60.0])
    ]
    for rotation in levelled[1:]:
        np.testing.assert_allclose(rotation, levelled[0], atol=1e-9)
    # Anchored to the start pose instead, those same poses land 100 deg apart.
    anchored = [
        _aligned_drill_rotation(_WALL_NORMAL, _rot_x(angle), blend=1.0)
        for angle in np.radians([-40.0, 60.0])
    ]
    assert _turn_deg(anchored[0], anchored[1]) == pytest.approx(100.0, abs=1e-6)


def test_levelling_fades_in_from_the_start_pose_without_a_step():
    start = _rot_x(np.radians(50.0))
    np.testing.assert_allclose(
        _aligned_drill_rotation(_WALL_NORMAL, start, 0.0, up=_WALL_UP),
        start,
        atol=1e-12,
    )
    # Constant-rate turn: equal rotation per blend step.
    steps = []
    previous = start
    for blend in np.linspace(0.0, 1.0, 51)[1:]:
        current = _aligned_drill_rotation(_WALL_NORMAL, start, blend, up=_WALL_UP)
        steps.append(_turn_deg(previous, current))
        previous = current
    assert max(steps) - min(steps) < 1e-6
    np.testing.assert_allclose(
        previous, _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP), atol=1e-9
    )


def test_roll_offset_cants_about_the_drill_axis_and_nothing_else():
    canted = _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP, np.radians(30.0))
    np.testing.assert_allclose(canted[:, 0], _WALL_NORMAL, atol=1e-12)
    level = _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP)
    assert _turn_deg(level, canted) == pytest.approx(30.0, abs=1e-9)


def test_levelling_declines_a_normal_it_cannot_level_against():
    """Roll is undefined when the drill points along `up`; the caller falls back."""
    assert _levelled_drill_rotation(_WALL_UP, _WALL_UP) is None
    start = _rot_x(np.radians(20.0))
    np.testing.assert_allclose(
        _aligned_drill_rotation(_WALL_UP, start, 1.0, up=_WALL_UP),
        _aligned_drill_rotation(_WALL_UP, start, 1.0),
        atol=1e-12,
    )


def test_orientation_is_fully_turned_by_the_standoff_not_by_the_target():
    """The turn completes at the standoff; the insertion holds a fixed rotation.

    Mirrors ``_planned_pose``: the blend is clocked against the transit leg only.
    """
    start = _rot_x(np.radians(50.0))
    total = _TRANSIT_DURATION_S + _INSERT_DURATION_S

    def planned_rotation(elapsed):
        blend = _quintic(min(elapsed / _TRANSIT_DURATION_S, 1.0))
        return _aligned_drill_rotation(_WALL_NORMAL, start, blend, up=_WALL_UP)

    goal = _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP)
    np.testing.assert_allclose(planned_rotation(_TRANSIT_DURATION_S), goal, atol=1e-12)
    # Still turning right up to the standoff, and rigid from there to the target.
    assert _turn_deg(planned_rotation(_TRANSIT_DURATION_S * 0.99), goal) > 0.0
    for elapsed in np.linspace(_TRANSIT_DURATION_S, total, 21):
        np.testing.assert_allclose(planned_rotation(elapsed), goal, atol=1e-12)


def test_upright_drill_is_reachable_across_the_sampling_box(drill_robot_model):
    """IK reaches every corner of IsaacSimCfg's box with the drill upright."""
    model = drill_robot_model
    goal = _levelled_drill_rotation(_WALL_NORMAL, _WALL_UP)
    worst_mm = worst_deg = 0.0
    for y in _TARGET_Y_RANGE:
        for z in _TARGET_Z_RANGE:
            target = np.array([_WALL_X, y, z]) - _PELVIS
            q = _replay(model, target, up=_WALL_UP)
            model.cache_forward_kinematics(q, auto_clip=False)
            achieved = model.frame_placement("drill_tip_link")
            worst_mm = max(
                worst_mm, float(np.linalg.norm(achieved.translation - target)) * 1e3
            )
            worst_deg = max(worst_deg, _turn_deg(goal, achieved.rotation))
    # Measured: 0.04 mm and 0.01 deg at every corner.
    assert worst_mm < 1.0, f"worst tip error {worst_mm:.2f} mm"
    assert worst_deg < 0.5, f"worst tip orientation error {worst_deg:.2f} deg"


# ── Start-pose arrival gate ───────────────────────────────────────────────────
#
# GOING_TO_DEFAULT holds until the measured joints (not just the command) have
# reached the start pose.

_N_UPPER = 6
_TOL_DEG = 2.0
_HOLD_S = 0.3
_TIMEOUT_S = 10.0


class _FakeLogger:
    def __init__(self):
        self.errors = []

    def info(self, *_a, **_k):
        pass

    def warn(self, *_a, **_k):
        pass

    def error(self, msg, *_a, **_k):
        self.errors.append(msg)


class _FakeCtx:
    def __init__(self, q):
        self.full_q = np.asarray(q, dtype=np.float64)
        self.upper_body_indices = list(range(len(self.full_q)))


class _FakeClockNode:
    """Node stub exposing the ROS clock read by a real :class:`TaskClock`."""

    def __init__(self, state):
        self._state = state

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self._state["t"] * 1e9))
        )


def _gate(measured_deg, _clock):
    """A handler carrying only what the arrival gate reads."""
    from g1_control.states.auto_collect import AutoCollectHandler

    handler = object.__new__(AutoCollectHandler)
    handler._ctx = _FakeCtx(np.deg2rad(measured_deg))
    handler._return_target = np.zeros(_N_UPPER)
    handler._upper_body_labels = [f"right_arm[{i}]" for i in range(_N_UPPER)]
    handler._return_duration_s = 4.0
    handler._return_settle_deg = _TOL_DEG
    handler._return_settle_hold_s = _HOLD_S
    handler._return_settle_timeout_s = _TIMEOUT_S
    handler._return_settled_since = None
    handler._logger = _FakeLogger()
    handler._clock = TaskClock(_FakeClockNode(_clock), sim_time=True)
    return handler


@pytest.fixture
def fake_clock(monkeypatch):
    """Drive g1_control.states.auto_collect's clock from the test."""
    pytest.importorskip("rclpy")
    from g1_control.states import auto_collect as module

    state = {"t": 100.0}
    # ROS clock (via _FakeClockNode) and wall clock (liveness watchdog).
    monkeypatch.setattr(module.time, "monotonic", lambda: state["t"])
    return state


def _measure(handler, deg):
    handler._ctx.full_q = np.deg2rad(np.asarray(deg, dtype=np.float64))


def test_gate_blocks_while_a_joint_is_short_of_the_start_pose(fake_clock):
    """One joint outside the band is enough to hold the phase, however long."""
    off = [0.0] * _N_UPPER
    off[3] = 12.0  # the arm is still travelling
    gate = _gate(off, fake_clock)
    assert gate._start_pose_reached(elapsed=4.0) is False
    fake_clock["t"] += 5.0
    assert gate._start_pose_reached(elapsed=9.0) is False


def test_gate_waits_out_the_hold_before_declaring_arrival(fake_clock):
    """Being inside the band for an instant is not arrival — it must be held."""
    gate = _gate([0.5] * _N_UPPER, fake_clock)
    assert gate._start_pose_reached(elapsed=4.0) is False, "passed on first touch"
    fake_clock["t"] += _HOLD_S / 2
    assert gate._start_pose_reached(elapsed=4.2) is False
    fake_clock["t"] += _HOLD_S
    assert gate._start_pose_reached(elapsed=4.5) is True


def test_gate_restarts_its_clock_when_the_arm_leaves_the_band(fake_clock):
    """An arm swinging through the start pose must not be counted as arrived."""
    gate = _gate([0.5] * _N_UPPER, fake_clock)
    gate._start_pose_reached(elapsed=4.0)
    fake_clock["t"] += _HOLD_S / 2
    _measure(gate, [9.0] * _N_UPPER)  # swung past
    assert gate._start_pose_reached(elapsed=4.2) is False
    _measure(gate, [0.5] * _N_UPPER)  # came back
    fake_clock["t"] += _HOLD_S / 2
    assert gate._start_pose_reached(elapsed=4.4) is False, "reused the stale clock"
    fake_clock["t"] += 2 * _HOLD_S
    assert gate._start_pose_reached(elapsed=5.0) is True


def test_gate_gives_up_after_the_timeout_and_says_so(fake_clock):
    """A headless worker must not hang on one unreachable joint — but it logs."""
    off = [0.0] * _N_UPPER
    off[1] = 30.0
    gate = _gate(off, fake_clock)
    assert gate._start_pose_reached(elapsed=4.0 + _TIMEOUT_S - 0.1) is False
    assert not gate._logger.errors
    assert gate._start_pose_reached(elapsed=4.0 + _TIMEOUT_S) is True
    assert len(gate._logger.errors) == 1
    assert "right_arm[1]" in gate._logger.errors[0]
    assert "30.00 deg" in gate._logger.errors[0]


def test_gate_reports_the_worst_joint_not_the_first(fake_clock):
    gate = _gate([1.0, 4.0, 0.0, 7.5, 2.0, 0.0], fake_clock)
    worst_deg, label = gate._start_pose_error()
    assert label == "right_arm[3]"
    assert worst_deg == pytest.approx(7.5)


# ── Which clock governs what ──────────────────────────────────────────────────
#
# Task durations run on the ROS clock (independent of real-time factor);
# liveness runs on the wall clock, since a dead sim also stops the ROS clock.


def test_task_durations_follow_the_ros_clock_not_the_wall(monkeypatch):
    """Sim time advancing is enough to complete a gate hold, with wall frozen."""
    from g1_control.states import auto_collect as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 500.0)  # wall stopped
    ros = {"t": 100.0}
    gate = _gate([0.5] * _N_UPPER, ros)

    assert gate._start_pose_reached(elapsed=0.0) is False
    ros["t"] += _HOLD_S * 2  # only the robot's own clock moves
    assert gate._start_pose_reached(elapsed=0.0) is True


def test_the_stall_watchdog_still_fires_when_sim_time_freezes(monkeypatch):
    """A sim that dies stops /clock; the watchdog must notice on wall time."""
    from g1_control.states import auto_collect as module
    from g1_control.states.auto_collect import AutoCollectHandler

    wall = {"t": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: wall["t"])

    resets = []
    handler = object.__new__(AutoCollectHandler)
    # ROS clock frozen: the sim is gone.
    handler._clock = TaskClock(_FakeClockNode({"t": 100.0}), sim_time=True)
    handler._loop = True
    handler._capture_active = False
    handler._last_progress = wall["t"]
    handler._loop_stall_timeout_s = 30.0
    handler._logger = _FakeLogger()
    handler._reset_request_pub = SimpleNamespace(publish=lambda msg: resets.append(msg))

    handler._maybe_kick_stalled_loop()
    assert resets == [], "nothing stalled yet"

    wall["t"] += 31.0  # sim time still frozen; only the wall has moved
    handler._maybe_kick_stalled_loop()
    assert len(resets) == 1, "the watchdog must fire on a frozen ROS clock"


def test_off_sim_time_durations_use_the_monotonic_clock(monkeypatch):
    """Without sim time, durations use time.monotonic() (immune to NTP steps)."""
    from g1_control.states import auto_collect as module

    wall = {"t": 42.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: wall["t"])
    ros = {"t": 100.0}  # a system clock that will jump
    gate = _gate([0.5] * _N_UPPER, ros)
    gate._clock = TaskClock(_FakeClockNode(ros), sim_time=False)

    assert gate._start_pose_reached(elapsed=0.0) is False
    ros["t"] -= 3600.0  # NTP steps the system clock backwards an hour
    wall["t"] += _HOLD_S * 2  # monotonic keeps counting
    assert gate._start_pose_reached(elapsed=0.0) is True


# ── Standoff release logging ──────────────────────────────────────────────────


class _RclpyStrictLogger:
    """Logger enforcing rclpy's rule: one severity per call site.

    rclpy raises ``ValueError`` if the same (function, file, line) logs at a
    second severity.
    """

    def __init__(self):
        self.messages = []
        self._severity_at = {}

    def _log(self, severity, message):
        frame = inspect.currentframe().f_back.f_back
        site = (frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)
        if self._severity_at.setdefault(site, severity) != severity:
            raise ValueError("Logger severity cannot be changed between calls.")
        self.messages.append((severity, message))

    def info(self, message, *_a, **_k):
        self._log("info", message)

    def warning(self, message, *_a, **_k):
        self._log("warning", message)

    def warn(self, message, *_a, **_k):
        self._log("warning", message)

    def error(self, message, *_a, **_k):
        self._log("error", message)


def _standoff_handler():
    """A handler carrying only what the standoff release reads."""
    from g1_control.states.auto_collect import AutoCollectHandler

    handler = object.__new__(AutoCollectHandler)
    handler._standoff_settle_mm = 1.0
    handler._standoff_wait_start = 100.0
    handler._logger = _RclpyStrictLogger()
    return handler


def test_a_timed_out_standoff_can_follow_a_nulled_one():
    """Both release outcomes in one process (loop mode reuses the handler)."""
    handler = _standoff_handler()
    handler._release_standoff(103.0, 0.4, -0.2, "nulled")
    handler._standoff_wait_start = 200.0
    handler._release_standoff(206.0, 7.1, -0.3, "timed out")

    messages = handler._logger.messages
    assert [severity for severity, _ in messages] == ["info", "warning"]
    assert "nulled after 3.0 s" in messages[0][1]
    assert "timed out after 6.0 s" in messages[1][1]
    assert "7.10 mm" in messages[1][1]
