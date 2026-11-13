"""ROS 2 / TF2 utility helpers for SE(3) broadcasting."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def mat4_to_se3(mat4: np.ndarray) -> pin.SE3:
    """Convert a 4×4 homogeneous numpy matrix to a Pinocchio SE3 object."""
    return pin.SE3(mat4[:3, :3].copy(), mat4[:3, 3].copy())


def se3_to_transform_stamped(
    se3: pin.SE3,
    parent_frame: str,
    child_frame: str,
    stamp,
) -> TransformStamped:
    """Build a ``TransformStamped`` message from a Pinocchio SE3 pose."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = float(se3.translation[0])
    t.transform.translation.y = float(se3.translation[1])
    t.transform.translation.z = float(se3.translation[2])

    quat = pin.Quaternion(se3.rotation)
    quat.normalize()
    t.transform.rotation.x = float(quat.x)
    t.transform.rotation.y = float(quat.y)
    t.transform.rotation.z = float(quat.z)
    t.transform.rotation.w = float(quat.w)

    return t


def broadcast_se3(
    broadcaster: TransformBroadcaster,
    se3: pin.SE3,
    parent_frame: str,
    child_frame: str,
    clock,
) -> None:
    """Broadcast a Pinocchio SE3 pose as a TF transform."""
    t = se3_to_transform_stamped(se3, parent_frame, child_frame, clock.now().to_msg())
    broadcaster.sendTransform(t)


def broadcast_mat4(
    broadcaster: TransformBroadcaster,
    mat4: np.ndarray,
    parent_frame: str,
    child_frame: str,
    clock,
) -> None:
    """Broadcast a raw 4×4 homogeneous numpy matrix as a TF transform."""
    broadcast_se3(broadcaster, mat4_to_se3(mat4), parent_frame, child_frame, clock)
