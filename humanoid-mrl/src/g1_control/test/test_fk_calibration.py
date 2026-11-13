"""Geometry tests for the FK Calibration state."""

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from g1_control.utils.fk_calibration import (
    average_tag_poses,
    CameraModel,
    point_offset_from_tag,
    project_camera_point,
)


def test_average_tag_poses_uses_so3_mean():
    translations = [np.array([0.1, 0.2, 1.0]), np.array([0.3, 0.4, 1.2])]
    rotations = Rotation.from_euler("z", [0.0, 20.0], degrees=True).as_matrix()

    averaged = average_tag_poses(translations, list(rotations))

    np.testing.assert_allclose(averaged.translation, [0.2, 0.3, 1.1])
    np.testing.assert_allclose(
        averaged.rotation,
        Rotation.from_euler("z", 10.0, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_project_camera_point_uses_camera_info_intrinsics():
    model = CameraModel(
        matrix=np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0, 0, 1]]),
        distortion=np.zeros(5),
        frame="camera_optical_frame",
    )

    assert project_camera_point(np.array([0.1, -0.2, 1.0]), model) == (330, 220)


def test_project_camera_point_rejects_points_behind_camera():
    model = CameraModel(np.eye(3), np.zeros(0), "camera_optical_frame")

    assert project_camera_point(np.array([0.0, 0.0, 0.0]), model) is None
    assert project_camera_point(np.array([0.0, 0.0, -1.0]), model) is None


def test_point_offset_from_tag_follows_tag_local_image_down_axis():
    tag_pose = pin.SE3(
        Rotation.from_euler("x", 90.0, degrees=True).as_matrix(),
        np.array([0.4, -0.2, 0.8]),
    )

    point = point_offset_from_tag(tag_pose, 0.030)

    # A +90 degree rotation about x maps tag +y onto pelvis +z.
    np.testing.assert_allclose(point, [0.4, -0.2, 0.83], atol=1e-12)


def test_point_offset_from_tag_applies_signed_normal_offset():
    tag_pose = pin.SE3(
        Rotation.from_euler("x", 90.0, degrees=True).as_matrix(),
        np.array([0.4, -0.2, 0.8]),
    )

    point = point_offset_from_tag(tag_pose, 0.030, -0.050)

    # The same rotation maps tag +z onto pelvis -y, so a negative normal
    # offset moves the point towards pelvis +y.
    np.testing.assert_allclose(point, [0.4, -0.15, 0.83], atol=1e-12)
