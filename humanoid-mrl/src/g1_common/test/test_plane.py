# MIT License
"""Unit tests for g1_common.plane and g1_common.depth (wall plane measurement).

Scene: head d435 pitched down at a vertical wall above a larger floor plane.
"""

import numpy as np
import pytest

from g1_common.camera import PinholeIntrinsics
from g1_common.depth import deproject, depth_scale, transform_points
from g1_common.plane import fit_plane, fit_plane_ransac, Plane

# Head d435 intrinsics (g1_simulation camera_specs).
HEAD_K = [
    606.370361328125, 0.0, 308.32464599609375,
    0.0, 606.370361328125, 246.02682495117188,
    0.0, 0.0, 1.0,
]  # fmt: skip
UP = np.array([0.0, 0.0, 1.0])


def head_intrinsics() -> PinholeIntrinsics:
    return PinholeIntrinsics.from_k(HEAD_K, 640, 480)


def wall_points(x_m: float = 1.2, n: int = 40, noise_m: float = 0.0) -> np.ndarray:
    """A vertical wall at x = ``x_m`` in a pelvis-like frame (+x forward, +z up)."""
    rng = np.random.default_rng(7)
    y, z = np.meshgrid(np.linspace(-0.7, 0.7, n), np.linspace(0.3, 1.6, n))
    points = np.column_stack((np.full(y.size, x_m), y.ravel(), z.ravel()))
    if noise_m:
        points[:, 0] += rng.normal(0.0, noise_m, len(points))
    return points


def floor_points(z_m: float = 0.0, n: int = 60) -> np.ndarray:
    """A horizontal floor, larger than the wall; stops 10 cm short so no points are
    shared."""
    x, y = np.meshgrid(np.linspace(0.2, 1.1, n), np.linspace(-0.9, 0.9, n))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z_m)))


# ── The fit ──────────────────────────────────────────────────────────────────


def test_a_clean_wall_is_fitted_to_its_own_position_and_normal():
    plane = fit_plane(wall_points(x_m=1.2))
    assert plane is not None
    assert plane.point[0] == pytest.approx(1.2)
    assert abs(plane.normal[0]) == pytest.approx(1.0)
    assert plane.rms_m == pytest.approx(0.0, abs=1e-12)


def test_fewer_than_three_points_define_no_plane():
    assert fit_plane(np.zeros((2, 3))) is None


def test_identical_points_define_no_plane():
    """Degenerate case that would otherwise report rms 0.0."""
    assert fit_plane(np.ones((10, 3))) is None


def test_collinear_points_define_no_plane():
    """Collinear points leave the plane free to rotate about the line."""
    t = np.linspace(0.0, 1.0, 50)
    line = np.column_stack((t, 2.0 * t, -0.5 * t)) + np.array([1.2, 0.0, 0.4])
    assert fit_plane(line) is None


def test_the_fit_minimises_perpendicular_distance_not_depth_error():
    """Tilted wall x = z: normal is 45 deg off both axes."""
    t = np.linspace(-1.0, 1.0, 30)
    y, s = np.meshgrid(t, t)
    points = np.column_stack((s.ravel(), y.ravel(), s.ravel()))
    plane = fit_plane(points)
    assert plane is not None
    np.testing.assert_allclose(
        np.abs(plane.normal), [0.5**0.5, 0.0, 0.5**0.5], atol=1e-9
    )


def test_rms_reports_the_noise_it_was_given():
    plane = fit_plane(wall_points(noise_m=0.004))
    assert plane is not None
    assert plane.rms_m == pytest.approx(0.004, rel=0.2)


# ── RANSAC and the verticality prior ─────────────────────────────────────────


def test_ransac_picks_the_wall_over_a_larger_floor():
    """The up-axis prior selects the wall over the floor (3600 vs 1600 points)."""
    cloud = np.vstack((wall_points(x_m=1.2), floor_points()))
    plane = fit_plane_ransac(cloud, up_axis=UP, min_inliers=100)
    assert plane is not None
    assert plane.point[0] == pytest.approx(1.2, abs=1e-6)
    assert abs(plane.normal[2]) < 0.02


def test_without_the_prior_the_floor_wins():
    """Without the up-axis prior the larger floor is returned."""
    cloud = np.vstack((wall_points(x_m=1.2), floor_points()))
    plane = fit_plane_ransac(cloud, up_axis=None, min_inliers=100)
    assert plane is not None
    assert abs(plane.normal[2]) == pytest.approx(1.0, abs=1e-6)


def test_a_wall_tilted_past_the_limit_is_refused():
    """A surface outside the verticality prior is rejected regardless of inliers."""
    tilted = wall_points()
    # Roll the wall 40 deg about y, well past the 20 deg default.
    angle = np.deg2rad(40.0)
    rot = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    assert fit_plane_ransac(tilted @ rot.T, up_axis=UP, min_inliers=100) is None


def test_a_cloud_too_small_to_vouch_for_a_plane_returns_none():
    """Three points fit any plane exactly; min_inliers is what rejects that."""
    assert fit_plane_ransac(wall_points(n=6), up_axis=UP, min_inliers=200) is None


def test_sampled_counts_the_whole_cloud_not_the_consensus():
    """``sampled`` is the full input size, ``inliers`` the consensus."""
    cloud = np.vstack((wall_points(x_m=1.2), floor_points()))
    plane = fit_plane_ransac(cloud, up_axis=UP, min_inliers=100)
    assert plane is not None
    assert plane.sampled == len(cloud)
    assert plane.inliers == len(wall_points())


def test_the_same_cloud_gives_the_same_plane_twice():
    """RANSAC is deterministic for a given cloud."""
    cloud = np.vstack((wall_points(x_m=1.2, noise_m=0.003), floor_points()))
    first = fit_plane_ransac(cloud, up_axis=UP, min_inliers=100)
    second = fit_plane_ransac(cloud, up_axis=UP, min_inliers=100)
    assert first is not None and second is not None
    np.testing.assert_allclose(first.normal, second.normal)
    np.testing.assert_allclose(first.point, second.point)


# ── Using the plane ──────────────────────────────────────────────────────────


def test_intersect_ray_returns_a_multiple_of_the_ray():
    """``scale * ray`` hits the plane with the bearing unchanged."""
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 100, 100, 0.0)
    ray = np.array([0.6, 0.1, 0.0])  # a target the detector put at x = 0.6
    scale = plane.intersect_ray(ray)
    assert scale == pytest.approx(2.0)
    hit = scale * ray
    assert hit[0] == pytest.approx(1.2)
    # The bearing survived: the hit is along the original ray.
    np.testing.assert_allclose(np.cross(hit, ray), np.zeros(3), atol=1e-12)


def test_a_ray_parallel_to_the_wall_never_meets_it():
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 100, 0.0)
    assert plane.intersect_ray(np.array([0.0, 1.0, 0.0])) is None
    assert plane.ray_cos(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_ray_cos_measures_conditioning_not_direction():
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 100, 0.0)
    assert plane.ray_cos(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert plane.ray_cos(np.array([-1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert plane.ray_cos(np.zeros(3)) == pytest.approx(0.0)


def test_intersect_ray_respects_a_non_origin_start():
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 100, 0.0)
    scale = plane.intersect_ray(np.array([1.0, 0.0, 0.0]), origin=np.array([0.2, 0, 0]))
    assert scale == pytest.approx(1.0)


def test_facing_turns_the_normal_back_towards_the_viewer():
    """An SVD normal has an arbitrary sign; every consumer means the outward one."""
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 100, 0.0)
    looking_forward = np.array([1.0, 0.0, 0.0])
    assert plane.facing(looking_forward).normal[0] == pytest.approx(-1.0)
    # Already facing the viewer: returned untouched.
    assert plane.facing(-looking_forward).normal[0] == pytest.approx(1.0)


def test_facing_preserves_the_fit_quality():
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 42, 99, 0.003)
    flipped = plane.facing(np.array([1.0, 0.0, 0.0]))
    assert (flipped.inliers, flipped.sampled, flipped.rms_m) == (42, 99, 0.003)


def test_signed_distance_is_positive_on_the_normal_side():
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 100, 100, 0.0)
    assert plane.signed_distance(np.array([0.0, 0.0, 0.0])) == pytest.approx(1.2)
    assert plane.signed_distance(np.array([2.0, 0.0, 0.0])) == pytest.approx(-0.8)


def test_transformed_moves_the_point_and_only_rotates_the_normal():
    """A translation must not move a normal; a rotation must move both."""
    plane = Plane(np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10, 10, 0.0)
    shift = np.eye(4)
    shift[:3, 3] = [0.5, -0.2, 0.1]
    moved = plane.transformed(shift)
    np.testing.assert_allclose(moved.normal, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(moved.point, [1.7, -0.2, 0.1])

    yaw = np.eye(4)
    yaw[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    turned = plane.transformed(yaw)
    np.testing.assert_allclose(turned.normal, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(turned.point, [0.0, 1.2, 0.0], atol=1e-12)


def test_offset_is_the_d_in_n_dot_x_equals_d():
    plane = Plane(np.array([1.2, 0.3, 0.0]), np.array([1.0, 0.0, 0.0]), 10, 10, 0.0)
    assert plane.offset == pytest.approx(1.2)


# ── Deprojection ─────────────────────────────────────────────────────────────


def test_a_constant_depth_image_deprojects_to_a_plane_at_that_depth():
    intr = head_intrinsics()
    cloud = deproject(np.full((480, 640), 1.5), intr, stride=8)
    assert len(cloud) == 60 * 80
    np.testing.assert_allclose(cloud[:, 2], 1.5)
    plane = fit_plane(cloud)
    assert plane is not None
    assert plane.point[2] == pytest.approx(1.5)
    assert abs(plane.normal[2]) == pytest.approx(1.0)


def test_the_principal_point_deprojects_to_the_optical_axis():
    intr = head_intrinsics()
    depth = np.zeros((480, 640))
    # cx, cy rounded to the nearest pixel: 308, 246.
    depth[246, 308] = 2.0
    cloud = deproject(depth, intr, stride=1)
    assert len(cloud) == 1
    np.testing.assert_allclose(cloud[0], [0.0, 0.0, 2.0], atol=2e-3)


def test_zeros_and_out_of_band_readings_are_dropped_not_deprojected():
    """Zero (no stereo match) and out-of-range depths are dropped."""
    intr = head_intrinsics()
    depth = np.zeros((480, 640))
    depth[100:110, 100:110] = 1.5
    depth[200:210, 200:210] = 9.0  # past max_depth_m
    depth[300:310, 300:310] = 0.05  # nearer than min_depth_m
    cloud = deproject(depth, intr, stride=1, min_depth_m=0.1, max_depth_m=5.0)
    assert len(cloud) == 100
    np.testing.assert_allclose(cloud[:, 2], 1.5)


def test_nan_readings_are_dropped():
    intr = head_intrinsics()
    depth = np.full((480, 640), np.nan)
    depth[10, 10] = 1.0
    cloud = deproject(depth, intr, stride=1)
    assert len(cloud) == 1


def test_an_all_invalid_frame_yields_an_empty_cloud_of_the_right_shape():
    """Empty cloud has shape (0, 3) so it stacks without special-casing."""
    cloud = deproject(np.zeros((480, 640)), head_intrinsics())
    assert cloud.shape == (0, 3)


def test_intrinsics_that_describe_a_different_image_are_refused():
    """Depth image size must match the intrinsics."""
    with pytest.raises(ValueError, match="does not describe one image"):
        deproject(np.full((240, 320), 1.0), head_intrinsics())


def test_a_non_2d_depth_array_is_refused():
    with pytest.raises(ValueError, match="must be 2-D"):
        deproject(np.full((480, 640, 3), 1.0), head_intrinsics())


def test_stride_decimates_uniformly_without_rescaling_the_intrinsics():
    """Stride keeps every n-th pixel over the same field of view."""
    intr = head_intrinsics()
    full = deproject(np.full((480, 640), 2.0), intr, stride=1)
    every_fourth = deproject(np.full((480, 640), 2.0), intr, stride=4)
    assert len(every_fourth) == len(full) // 16
    # Same field of view, not a cropped or zoomed one.
    assert every_fourth[:, 0].min() == pytest.approx(full[:, 0].min())
    assert every_fourth[:, 1].min() == pytest.approx(full[:, 1].min())


def test_depth_scale_knows_the_two_encodings_in_use():
    assert depth_scale("32FC1") == pytest.approx(1.0)
    assert depth_scale("16UC1") == pytest.approx(0.001)


def test_an_unknown_depth_encoding_is_refused_rather_than_assumed_metres():
    """Unknown encodings raise instead of assuming metres."""
    with pytest.raises(ValueError, match="unsupported depth encoding"):
        depth_scale("mono16")


def test_transform_points_matches_the_homogeneous_form():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(50, 3))
    transform = np.eye(4)
    angle = 0.4
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = [0.1, -0.2, 0.3]
    expected = (transform @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    np.testing.assert_allclose(transform_points(points, transform), expected)


def test_a_deprojected_wall_survives_the_trip_into_another_frame():
    """End to end: render a wall into a pitched camera, deproject, transform, fit."""
    intr = head_intrinsics()
    pitch = np.deg2rad(47.6)  # the head camera's pitch in the sim scene
    # T(pelvis <- optical): optical +z forward maps to pelvis +x, pitched down.
    optical_to_pelvis = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch), 1.202],
        ]
    )
    body = np.array(
        [
            [0.0, 0.0, 1.0],  # optical +z -> body +x (forward)
            [-1.0, 0.0, 0.0],  # optical +x -> body -y (left)
            [0.0, -1.0, 0.0],  # optical +y -> body -z (down)
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = optical_to_pelvis[:3, :3] @ body
    transform[:3, 3] = optical_to_pelvis[:3, 3]

    # Wall at x = 1.2 in the pelvis frame, rendered as a depth image.
    cam_from_pelvis = np.linalg.inv(transform)
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    # Ray through each pixel, in the pelvis frame.
    dirs = np.column_stack(
        (
            ((u - intr.cx) / intr.fx).ravel(),
            ((v - intr.cy) / intr.fy).ravel(),
            np.ones(u.size),
        )
    )
    dirs_pelvis = dirs @ transform[:3, :3].T
    origin = transform[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (1.2 - origin[0]) / dirs_pelvis[:, 0]
    z = np.where((t > 0) & np.isfinite(t), t, 0.0).reshape(480, 640)

    cloud = deproject(z, intr, stride=6, max_depth_m=8.0)
    assert len(cloud) > 1000
    plane = fit_plane_ransac(
        transform_points(cloud, transform), up_axis=UP, min_inliers=500
    )
    assert plane is not None
    assert plane.point[0] == pytest.approx(1.2, abs=1e-6)
    assert abs(plane.normal[0]) == pytest.approx(1.0, abs=1e-6)
    assert cam_from_pelvis.shape == (4, 4)
