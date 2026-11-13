# MIT License
"""Geometry behind the wrist camera calibration, kept free of ROS and OpenCV.

Frames D, L, C and T are defined in calibrate_hand_camera_node's module docstring.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Tag centre in the drill body frame, from the USD ``dummy_drill_Tag_link`` prim
# (g1_simulation/assets/isaacsim/g1/configuration/hand_drill_hilti.usd).
# Override if the tag is re-stuck.
DEFAULT_TAG_XYZ_IN_DRILL_BODY = [
    -0.054371295851537806,
    0.0,
    0.031427397329002946,
]

# Origin of URDF right_drill_link in the drill body frame (pure translation):
# URDF 0.0833 m vs USD 0.2112550 m ahead of right_wrist_yaw_link.
DEFAULT_URDF_LINK_XYZ_IN_DRILL_BODY = [0.0833 - 0.2112549984204769, 0.0, 0.0]

# Tag frame in the drill body frame: drill = (x, -y, -z) of tag. Self-inverse.
R_DRILL_TAG = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

# Body -> optical change of basis, rpy = (-pi/2, 0, -pi/2), as in the URDF's
# hand_camera_optical_joint.
R_BODY_TO_OPTICAL = Rotation.from_euler(
    "xyz", [-np.pi / 2, 0.0, -np.pi / 2]
).as_matrix()


def camera_pose_in_drill_body(
    t_cam_tag: Sequence[float],
    r_cam_tag: np.ndarray,
    tag_xyz_in_drill_body: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn one tag detection into the camera's optical pose in the drill body.

    :param t_cam_tag: tag position in the camera optical frame, as AprilTag
        reports it.
    :param r_cam_tag: tag orientation in the camera optical frame, 3x3.
    :param tag_xyz_in_drill_body: where the tag is stuck on the drill.
    :returns: ``(translation, rotation)`` of the camera optical frame in the
        drill body frame.
    """
    t_cam_tag = np.asarray(t_cam_tag, dtype=float).reshape(3)
    r_cam_tag = np.asarray(r_cam_tag, dtype=float)

    # Camera in tag coordinates, then re-expressed in the drill body frame.
    r_tag_cam = r_cam_tag.T
    t_tag_cam = -r_tag_cam @ t_cam_tag

    translation = (
        np.asarray(tag_xyz_in_drill_body, dtype=float) + R_DRILL_TAG @ t_tag_cam
    )
    return translation, R_DRILL_TAG @ r_tag_cam


def tag_pose_in_camera(
    t_drill_cam: np.ndarray,
    r_drill_cam: np.ndarray,
    tag_xyz_in_drill_body: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Tag pose in the camera optical frame for a given camera pose.

    Inverse of :func:`camera_pose_in_drill_body`.

    :returns: ``(t_cam_tag, r_cam_tag)``, matching what pupil_apriltags reports.
    """
    t_drill_cam = np.asarray(t_drill_cam, dtype=float).reshape(3)
    r_drill_cam = np.asarray(r_drill_cam, dtype=float)
    tag_xyz = np.asarray(tag_xyz_in_drill_body, dtype=float).reshape(3)

    # R_DRILL_TAG is self-inverse.
    r_tag_cam = R_DRILL_TAG @ r_drill_cam
    t_tag_cam = R_DRILL_TAG @ (t_drill_cam - tag_xyz)

    r_cam_tag = r_tag_cam.T
    return -r_cam_tag @ t_tag_cam, r_cam_tag


def tag_corners_in_tag_frame(tag_size_m: float) -> np.ndarray:
    """The tag's four corners, ordered to match ``Detection.corners``.

    Ordering verified against pupil_apriltags (reprojects to ~1 px).
    """
    half = 0.5 * float(tag_size_m)
    return np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ]
    )


def average_poses(
    translations: List[np.ndarray], rotations: List[np.ndarray]
) -> Tuple[np.ndarray, Rotation, np.ndarray, float]:
    """Average a sample set and report its spread.

    :returns: ``(translation, rotation, per_axis_spread_mm, max_rotation_deg)``,
        the last being the largest angle any sample sits from the mean.
    """
    stacked = np.stack(translations)
    sampled = Rotation.from_matrix(np.stack(rotations))
    mean_rotation = sampled.mean()
    return (
        stacked.mean(axis=0),
        mean_rotation,
        stacked.std(axis=0) * 1e3,
        float(np.degrees((mean_rotation.inv() * sampled).magnitude().max())),
    )


def urdf_joint_origin(
    t_drill_cam: np.ndarray,
    r_drill_cam: Rotation,
    urdf_link_xyz_in_drill_body: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Express the pose as the URDF's ``hand_camera_joint`` origin.

    Translation is shifted into right_drill_link and rotation converted from
    optical to body frame. rpy is extrinsic xyz (URDF), not Isaac's intrinsic XYZ.
    """
    xyz = t_drill_cam - np.asarray(urdf_link_xyz_in_drill_body, dtype=float)
    body = Rotation.from_matrix(r_drill_cam.as_matrix() @ R_BODY_TO_OPTICAL.T)
    return xyz, body.as_euler("xyz")


def usd_xform(
    t_drill_cam: np.ndarray, r_drill_cam: Rotation
) -> Tuple[np.ndarray, List[float]]:
    """Express the pose as the USD ``hand_cam_link_01`` xform ops.

    The prim is parented to hand_drill_hilti in the optical convention, so only
    the quaternion is reordered to (w, x, y, z).
    """
    quat_xyzw = r_drill_cam.as_quat()
    return t_drill_cam, [float(quat_xyzw[3]), *(float(v) for v in quat_xyzw[:3])]
