# MIT License
"""Broadcast one mocap rigid body's pose relative to another as a TF edge.

The Qualisys driver publishes bodies in the ``qualisys`` lab frame with no TF.
This node publishes the child relative to the parent, so the target hangs off a
frame the URDF already broadcasts::

    pelvis -> ... -> drill_chuck_link -> red_square_target
             (robot_state_publisher)      (this node)

Only the child edge is broadcast; the parent must be a URDF frame.

* The target inherits the arm's FK error; AUTO_COLLECT re-reads it every tick
  (``g1_control.utils.target_correction``) so that error cancels against IK.
* The child's x-axis is the wall normal and must point into the wall (set by the
  QTM body definition).

Each body is median-filtered in the lab frame with its own window: the static
target over a long window, the moving parent pass-through by default.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from mocap4r2_msgs.msg import RigidBodies
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from ``[x, y, z, w]``."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """``[x, y, z, w]`` from a rotation matrix, via the largest-diagonal branch."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        return np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
                0.25 * s,
            ]
        )
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def _position(pose) -> np.ndarray:
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def _quaternion(pose) -> np.ndarray:
    return np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )


class PoseFilter:
    """Sliding-window median of one body's pose, in whatever frame it is fed.

    A window of 1 is an exact pass-through.
    """

    def __init__(self, window: int) -> None:
        self._window = max(1, int(window))
        self._translations: deque = deque(maxlen=self._window)
        self._quaternions: deque = deque(maxlen=self._window)

    @property
    def window(self) -> int:
        return self._window

    def update(
        self, translation: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add a sample and return the filtered ``(translation, quaternion)``."""
        # Align to the oldest sample's hemisphere (q and -q are the same rotation).
        if self._quaternions and float(quaternion @ self._quaternions[0]) < 0.0:
            quaternion = -quaternion
        self._translations.append(translation)
        self._quaternions.append(quaternion)

        # Component-wise median, renormalised; valid only for small spreads.
        rotation = np.median(np.array(self._quaternions), axis=0)
        return (
            np.median(np.array(self._translations), axis=0),
            rotation / np.linalg.norm(rotation),
        )


class MocapTfBridge(Node):
    """Publish ``child_body`` relative to ``parent_body`` as a TF transform."""

    def __init__(self) -> None:
        super().__init__("mocap_tf_bridge")

        self.declare_parameter("parent_body", "drill_chuck_link")
        self.declare_parameter("child_body", "red_square_target")
        # Moving parent: pass-through by default to avoid group delay.
        self.declare_parameter("parent_median_window", 1)
        # Static wall target: 101 samples ~= 2 s at 50 Hz.
        self.declare_parameter("child_median_window", 101)

        self._parent = self.get_parameter("parent_body").value
        self._child = self.get_parameter("child_body").value
        self._parent_filter = PoseFilter(
            self.get_parameter("parent_median_window").value
        )
        self._child_filter = PoseFilter(self.get_parameter("child_median_window").value)

        self._broadcaster = TransformBroadcaster(self)
        # BEST_EFFORT to match the driver.
        self.create_subscription(
            RigidBodies,
            "/rigid_bodies",
            self._callback,
            QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            ),
        )
        self.get_logger().info(
            f"Mocap TF bridge: {self._parent} -> {self._child}, lab-frame median "
            f"over {self._parent_filter.window} samples (parent) and "
            f"{self._child_filter.window} (child)."
        )

    def _callback(self, msg: RigidBodies) -> None:
        bodies = {rb.rigid_body_name: rb.pose for rb in msg.rigidbodies}
        if self._parent not in bodies or self._child not in bodies:
            self.get_logger().warn(
                f"Mocap TF bridge: waiting for both bodies (have "
                f"{sorted(bodies)}, need {self._parent} and {self._child}).",
                throttle_duration_sec=5.0,
            )
            return

        parent, child = bodies[self._parent], bodies[self._child]
        # Filter each body in the lab frame, then compose.
        p_parent, q_parent = self._parent_filter.update(
            _position(parent), _quaternion(parent)
        )
        p_child, q_child = self._child_filter.update(
            _position(child), _quaternion(child)
        )

        r_parent = _quat_to_matrix(q_parent)
        translation = r_parent.T @ (p_child - p_parent)
        rotation = _matrix_to_quat(r_parent.T @ _quat_to_matrix(q_child))

        tf = TransformStamped()
        # QTM capture time mapped onto the ROS clock.
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = self._parent
        tf.child_frame_id = self._child
        tf.transform.translation.x = float(translation[0])
        tf.transform.translation.y = float(translation[1])
        tf.transform.translation.z = float(translation[2])
        tf.transform.rotation.x = float(rotation[0])
        tf.transform.rotation.y = float(rotation[1])
        tf.transform.rotation.z = float(rotation[2])
        tf.transform.rotation.w = float(rotation[3])
        self._broadcaster.sendTransform(tf)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MocapTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
