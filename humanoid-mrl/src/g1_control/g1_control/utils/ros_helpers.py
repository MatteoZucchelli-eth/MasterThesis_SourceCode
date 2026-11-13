# MIT License
"""Small ROS helpers shared across g1_control: QoS, TF lookups, RViz markers."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException
from visualization_msgs.msg import Marker, MarkerArray

# Latched QoS so a subscriber that attaches late (e.g. RViz) still receives the
# last published message.
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def lookup_se3(
    tf_buffer: Buffer, target_frame: str | None, source_frame: str
) -> pin.SE3 | None:
    """Latest ``target_frame`` pose expressed in ``source_frame``, or None.

    Also returns None when ``target_frame`` is None (unresolved task target).
    """
    if target_frame is None:
        return None
    try:
        tf = tf_buffer.lookup_transform(source_frame, target_frame, rclpy.time.Time())
    except TransformException:
        return None
    t = tf.transform.translation
    r = tf.transform.rotation
    return pin.SE3(
        pin.Quaternion(r.w, r.x, r.y, r.z).normalized().matrix(),
        np.array([t.x, t.y, t.z]),
    )


def lookup_position(
    tf_buffer: Buffer, target_frame: str | None, source_frame: str
) -> np.ndarray | None:
    """Latest ``target_frame`` position in ``source_frame`` as (3,), or None."""
    se3 = lookup_se3(tf_buffer, target_frame, source_frame)
    return None if se3 is None else se3.translation.copy()


def pose_error(a: pin.SE3, b: pin.SE3) -> tuple[float, float]:
    """Return (position error [m], orientation error [rad]) between two SE3."""
    pos = float(np.linalg.norm(a.translation - b.translation))
    ang = float(np.linalg.norm(pin.log3(a.rotation.T @ b.rotation)))
    return pos, ang


def sphere_marker(
    ns: str,
    marker_id: int,
    position,
    color: tuple[float, float, float, float],
    diameter: float,
    frame: str = "pelvis",
) -> Marker:
    """An RViz sphere marker with a zero stamp (RViz uses the latest TF)."""
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = rclpy.time.Time().to_msg()
    m.ns = ns
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = diameter
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def delete_all_markers(ns: str) -> MarkerArray:
    """A MarkerArray that clears every marker previously published under ``ns``."""
    delete_all = Marker()
    delete_all.ns = ns
    delete_all.action = Marker.DELETEALL
    markers = MarkerArray()
    markers.markers.append(delete_all)
    return markers
