# MIT License
"""Frame conventions for the wrist camera calibration, checked against the assets.

Checks the tag -> camera pose round trip and that the calibration output matches
the shipped URDF and Isaac USD. After a remount, re-run the calibration and
update the URDF, the USD and USD_* below together.
"""

import os
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
from g1_cam_drill_interface.extrinsics import (
    average_poses,
    camera_pose_in_drill_body,
    DEFAULT_TAG_XYZ_IN_DRILL_BODY,
    DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY,
    R_BODY_TO_OPTICAL,
    R_DRILL_TAG,
    urdf_joint_origin,
    usd_xform,
)
from scipy.spatial.transform import Rotation

# Snapshot of prim hand_drill_hilti/hand_cam_link_01 in
# g1_simulation/assets/isaacsim/g1/g1_29dof_rev_1_0_w_hilti_drill.usd (optical
# convention, quaternion wxyz). Refresh from the output of
# g1_simulation/scripts/sync_camera_extrinsics_to_usd.py when the camera moves.
USD_TRANSLATE = np.array(
    [-0.08088563655839826, 0.0005694106822838885, 0.07248831866672231]
)
USD_ORIENT_WXYZ = np.array(
    [
        0.004672240619819663,
        0.845526637838796,
        -0.0008907724613153745,
        0.5339120539911365,
    ]
)

URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "g1_state_publisher",
    "urdf",
    "g1_29_dof_with_hand_drill.urdf.xml",
)


def _usd_rotation() -> Rotation:
    w, x, y, z = USD_ORIENT_WXYZ
    return Rotation.from_quat([x, y, z, w])


def _synthetic_detection(t_drill_cam, r_drill_cam):
    """The detection a camera at that pose would report for the drill's tag (inverse of
    the node)."""
    tag_xyz = np.asarray(DEFAULT_TAG_XYZ_IN_DRILL_BODY)
    r_cam_drill = r_drill_cam.as_matrix().T
    return r_cam_drill @ (tag_xyz - t_drill_cam), r_cam_drill @ R_DRILL_TAG


def _urdf_joint(name):
    if not os.path.exists(URDF_PATH):
        pytest.skip(f"{URDF_PATH} not available")
    root = ElementTree.parse(URDF_PATH).getroot()
    for joint in root.iter("joint"):
        if joint.get("name") == name:
            origin = joint.find("origin")
            return (
                np.array([float(v) for v in origin.get("xyz").split()]),
                np.array([float(v) for v in origin.get("rpy").split()]),
            )
    raise AssertionError(f"joint {name} not found in {URDF_PATH}")


def test_detection_round_trips_to_the_camera_pose():
    t_drill_cam, r_drill_cam = USD_TRANSLATE, _usd_rotation()
    t_cam_tag, r_cam_tag = _synthetic_detection(t_drill_cam, r_drill_cam)

    translation, rotation = camera_pose_in_drill_body(
        t_cam_tag, r_cam_tag, DEFAULT_TAG_XYZ_IN_DRILL_BODY
    )

    np.testing.assert_allclose(translation, t_drill_cam, atol=1e-12)
    np.testing.assert_allclose(rotation, r_drill_cam.as_matrix(), atol=1e-12)


def test_urdf_output_matches_the_urdf_as_shipped():
    """The calibration output for the USD pose matches the URDF joint (URDF/USD
    agreement)."""
    xyz, rpy = urdf_joint_origin(
        USD_TRANSLATE, _usd_rotation(), DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY
    )
    urdf_xyz, urdf_rpy = _urdf_joint("hand_camera_joint")

    # Tolerance covers the URDF drill body being rotated 0.0119 deg relative to
    # the USD one (~2.7e-5 m / 2.1e-4 rad at the camera).
    np.testing.assert_allclose(xyz, urdf_xyz, atol=5e-5)
    np.testing.assert_allclose(rpy, urdf_rpy, atol=3e-4)


def test_urdf_rpy_is_extrinsic_and_composes_with_the_optical_joint():
    """The two URDF joints compose to the measured optical pose.

    URDF rpy is extrinsic ``"xyz"``; Isaac's Orient field is intrinsic ``"XYZ"``.
    """
    _, rpy = urdf_joint_origin(
        USD_TRANSLATE, _usd_rotation(), DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY
    )
    _, optical_rpy = _urdf_joint("hand_camera_optical_joint")

    composed = Rotation.from_euler("xyz", rpy) * Rotation.from_euler("xyz", optical_rpy)
    np.testing.assert_allclose(
        composed.as_matrix(), _usd_rotation().as_matrix(), atol=1e-9
    )
    # The optical joint is exactly the change of basis the module applies.
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", optical_rpy).as_matrix(),
        R_BODY_TO_OPTICAL,
        atol=1e-12,
    )


def test_usd_output_round_trips_to_the_asset():
    translate, orient_wxyz = usd_xform(USD_TRANSLATE, _usd_rotation())

    np.testing.assert_allclose(translate, USD_TRANSLATE, atol=1e-12)
    # Quaternion sign is free; compare the rotations, not the four numbers.
    w, x, y, z = orient_wxyz
    np.testing.assert_allclose(
        Rotation.from_quat([x, y, z, w]).as_matrix(),
        _usd_rotation().as_matrix(),
        atol=1e-12,
    )


def test_average_poses_reports_the_scatter_it_averaged_over():
    r_drill_cam = _usd_rotation()
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1e-3, size=(200, 3))  # 1 mm, 1-sigma

    translations = [USD_TRANSLATE + n for n in noise]
    rotations = [r_drill_cam.as_matrix()] * len(translations)

    mean, rotation, spread_mm, max_deg = average_poses(translations, rotations)

    np.testing.assert_allclose(mean, USD_TRANSLATE, atol=3e-4)
    np.testing.assert_allclose(rotation.as_matrix(), r_drill_cam.as_matrix(), atol=1e-9)
    np.testing.assert_allclose(spread_mm, [1.0, 1.0, 1.0], rtol=0.2)
    assert max_deg == pytest.approx(0.0, abs=1e-9)
