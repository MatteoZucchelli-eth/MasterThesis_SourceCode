# MIT License
"""Unit tests for the wall estimator — the fit, without ROS around it.

A head camera pitched down at a vertical wall, with a larger floor below it
that a fit without the verticality prior would return.
"""

import numpy as np
import pytest

from g1_common.camera import PinholeIntrinsics
from g1_sensors.depth.wall_plane import WallEstimatorConfig, WallPlaneEstimator

# The head d435 as both the sim and the robot describe it (camera_specs).
HEAD_K = [
    606.370361328125, 0.0, 308.32464599609375,
    0.0, 606.370361328125, 246.02682495117188,
    0.0, 0.0, 1.0,
]  # fmt: skip
WALL_X_M = 1.2
PITCH_DEG = 47.6  # the head camera's pitch in the sim scene
CAMERA_Z_M = 1.202


def intrinsics() -> PinholeIntrinsics:
    return PinholeIntrinsics.from_k(HEAD_K, 640, 480)


def pelvis_from_camera() -> np.ndarray:
    """T(pelvis <- d435_color_optical_frame), pitched down like the real mount."""
    pitch = np.deg2rad(PITCH_DEG)
    # optical (+z fwd, +x right, +y down) -> body (+x fwd, +y left, +z up)
    body = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    pitch_down = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = pitch_down @ body
    transform[:3, 3] = [0.0, 0.0, CAMERA_Z_M]
    return transform


def render_depth(
    *, wall_x_m: float | None = WALL_X_M, floor_z_m: float | None = 0.0
) -> np.ndarray:
    """Depth image of a vertical wall and/or a horizontal floor, in metres.

    Ray-casts each pixel against the requested surfaces and keeps the nearer hit.
    """
    intr = intrinsics()
    transform = pelvis_from_camera()
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    rays = np.column_stack(
        (
            ((u - intr.cx) / intr.fx).ravel(),
            ((v - intr.cy) / intr.fy).ravel(),
            np.ones(u.size),
        )
    )
    # Unit-z rays: the hit scale is the depth along the optical axis.
    in_pelvis = rays @ transform[:3, :3].T
    origin = transform[:3, 3]

    best = np.full(len(rays), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        if wall_x_m is not None:
            t = (wall_x_m - origin[0]) / in_pelvis[:, 0]
            best = np.where((t > 0) & (t < best), t, best)
        if floor_z_m is not None:
            t = (floor_z_m - origin[2]) / in_pelvis[:, 2]
            best = np.where((t > 0) & (t < best), t, best)
    return np.where(np.isfinite(best), best, 0.0).reshape(480, 640)


def estimate(config: WallEstimatorConfig | None = None, **kwargs):
    depth = kwargs.pop("depth", None)
    if depth is None:
        depth = render_depth(**kwargs)
    return WallPlaneEstimator(config).estimate(
        depth, intrinsics(), pelvis_from_camera()
    )


# ── The happy path ───────────────────────────────────────────────────────────


def test_a_wall_in_view_is_measured_where_it_is():
    result = estimate()
    assert result.plane is not None, result.reason
    assert result.plane.point[0] == pytest.approx(WALL_X_M, abs=1e-3)
    assert abs(result.plane.normal[0]) == pytest.approx(1.0, abs=1e-3)
    # Not zero: the floor seam joins the consensus and tilts the fit ~0.04 deg.
    # Centimetres here would indicate a ray-length depth stream.
    assert result.plane.rms_m < 1e-3


def test_the_normal_points_out_of_the_wall_towards_the_camera():
    """Every consumer means the outward normal; the drill travels along its
    negation. An SVD has no opinion about the sign, so this is the node's."""
    result = estimate()
    assert result.plane is not None
    # Camera at x = 0, wall at x = +1.2: outward is -x.
    assert result.plane.normal[0] < -0.99


def test_the_floor_does_not_win_even_though_it_is_bigger():
    """The reason the verticality prior exists. In this view the floor covers
    more pixels than the wall, so 'the largest plane' is the floor."""
    result = estimate()
    assert result.plane is not None
    assert abs(result.plane.normal[2]) < 0.01


def test_without_the_prior_the_floor_is_what_comes_back():
    """Documents what the prior buys, so losing it cannot look harmless."""
    result = estimate(WallEstimatorConfig(up_axis=(0.0, 0.0, 0.0)))
    assert result.plane is not None
    assert abs(result.plane.normal[2]) == pytest.approx(1.0, abs=1e-3)


def test_the_cloud_comes_back_alongside_the_plane():
    """So a caller can draw the measurement next to the surface fitted to it."""
    result = estimate()
    assert result.points.shape[1] == 3
    assert len(result.points) > result.plane.inliers


def test_a_tilted_wall_is_still_found_and_measured():
    """The prior is there to separate a wall from a floor, not to assume the
    wall is perfectly plumb."""
    depth = render_depth(wall_x_m=None, floor_z_m=None)
    # A wall rolled 12 deg about the pelvis y axis, well inside the 20 deg limit.
    angle = np.deg2rad(12.0)
    normal = np.array([np.cos(angle), 0.0, np.sin(angle)])
    intr = intrinsics()
    transform = pelvis_from_camera()
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    rays = np.column_stack(
        (
            ((u - intr.cx) / intr.fx).ravel(),
            ((v - intr.cy) / intr.fy).ravel(),
            np.ones(u.size),
        )
    )
    in_pelvis = rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    offset = float(normal @ np.array([WALL_X_M, 0.0, 0.8]))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - normal @ origin) / (in_pelvis @ normal)
    depth = np.where((t > 0) & np.isfinite(t), t, 0.0).reshape(480, 640)

    result = estimate(depth=depth)
    assert result.plane is not None, result.reason
    np.testing.assert_allclose(np.abs(result.plane.normal), np.abs(normal), atol=2e-3)


# ── Refusals: a missing plane must say which kind of missing ─────────────────


def test_an_empty_depth_frame_is_refused_and_says_so():
    """A stream that is not arriving and a scene with no wall need opposite
    fixes, so the reason has to distinguish them."""
    result = estimate(depth=np.zeros((480, 640)))
    assert result.plane is None
    assert "no depth readings" in result.reason
    assert result.points.shape == (0, 3)


def test_a_scene_with_no_wall_is_refused_with_the_cloud_intact():
    """The other half of the pair above: plenty of points, no wall among them."""
    result = estimate(wall_x_m=None)
    assert result.plane is None
    assert "vertical" in result.reason
    assert len(result.points) > 1000


def test_a_view_too_small_to_vouch_for_a_plane_is_refused():
    """min_inliers is consensus, not quality: three points fit any plane
    exactly and come back with an rms of zero."""
    result = estimate(WallEstimatorConfig(min_inliers=10**6))
    assert result.plane is None
    assert "1000000" in result.reason


def test_the_depth_band_excludes_what_it_says_it_does():
    """A band that cannot reach the wall must produce no wall, rather than a
    plane fitted to whatever was inside it."""
    result = estimate(WallEstimatorConfig(max_depth_m=0.5))
    assert result.plane is None


# ── Properties the consumer relies on ────────────────────────────────────────


def test_the_measurement_does_not_drift_between_identical_frames():
    """A range estimate that moves when the scene did not cannot be debugged
    from a log, and this one feeds a trajectory."""
    first, second = estimate(), estimate()
    np.testing.assert_allclose(first.plane.point, second.plane.point)
    np.testing.assert_allclose(first.plane.normal, second.plane.normal)


def test_decimation_changes_the_cost_not_the_answer():
    """stride exists to make the fit cheap; it must not move the wall."""
    fine = estimate(WallEstimatorConfig(stride=2))
    coarse = estimate(WallEstimatorConfig(stride=8, min_inliers=100))
    assert fine.plane is not None and coarse.plane is not None
    assert coarse.plane.point[0] == pytest.approx(fine.plane.point[0], abs=1e-3)
    assert len(coarse.points) < len(fine.points) / 10


def test_sampled_is_the_whole_view_so_inlier_share_is_meaningful():
    result = estimate()
    assert result.plane.sampled == len(result.points)
    assert 0.2 < result.plane.inliers / result.plane.sampled < 0.9


def test_a_ray_from_the_wrist_camera_meets_the_measured_wall():
    """The estimate's actual job: give a monocular detection a range.

    A detection along a ray that ends at the wall is moved to the wall with
    its bearing unchanged.
    """
    plane = estimate().plane
    # Wrist camera 0.9 m from the wall; detection 0.6 m along the ray (too near).
    origin = np.array([WALL_X_M - 0.9, 0.05, 0.9])
    ray = np.array([0.6, 0.04, -0.02])
    scale = plane.intersect_ray(ray, origin=origin)
    assert scale is not None
    hit = origin + scale * ray
    assert hit[0] == pytest.approx(WALL_X_M, abs=1e-3)
    # The bearing survived: the correction moved the point along the ray only.
    np.testing.assert_allclose(np.cross(hit - origin, ray), np.zeros(3), atol=1e-9)


# ── The measured depth bias ──────────────────────────────────────────────────


def test_the_depth_bias_pushes_the_measured_wall_further_away():
    """What ``depth_bias_m`` is for: the sensor reads the wall short.

    Each reading moves along its own ray, so the wall shifts by only the
    normal component of the 15 mm bias (camera pitched 47.6 deg).
    """
    plain = estimate()
    biased = estimate(WallEstimatorConfig(depth_bias_m=0.015))
    assert plain.plane is not None and biased.plane is not None, biased.reason
    shift = biased.plane.point[0] - plain.plane.point[0]
    assert 0.008 < shift < 0.015
    # The bias must not degrade the fit's flatness.
    assert biased.plane.normal[0] < -0.99
    assert biased.plane.rms_m < 1e-3


def test_the_bias_does_not_turn_a_missing_reading_into_a_surface():
    """A zero is 'no match', not a surface at the lens.

    Biased, it must stay below ``min_depth_m`` and be dropped.
    """
    result = WallPlaneEstimator(WallEstimatorConfig(depth_bias_m=0.015)).estimate(
        np.zeros((480, 640)), intrinsics(), pelvis_from_camera()
    )
    assert result.plane is None
    assert len(result.points) == 0
