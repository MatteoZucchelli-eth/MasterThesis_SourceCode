# MIT License
"""Publish a drill-tip-relative xyz translation for the GOAL_POSE state.

The translation is resolved once against the measured ``drill_tip_link`` pose,
in the tip's axes; the tip orientation is preserved.

Examples
--------
Move 5 cm forward along the drill axis::

    ros2 run g1_control goal_pose_publisher_node --x 0.05

Move 2 cm along the tip's +y axis and 1 cm along its -z axis::

    ros2 run g1_control goal_pose_publisher_node --y 0.02 --z -0.01
"""
from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def _build_translation(args) -> Vector3:
    msg = Vector3()
    msg.x = args.x
    msg.y = args.y
    msg.z = args.z
    return msg


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.0, help="Tip-frame x offset (m)")
    parser.add_argument("--y", type=float, default=0.0, help="Tip-frame y offset (m)")
    parser.add_argument("--z", type=float, default=0.0, help="Tip-frame z offset (m)")
    parser.add_argument("--topic", type=str, default="/goal_pose", help="Goal topic")
    # Ignore ROS-injected args (e.g. --ros-args) so `ros2 run` still works.
    known, _ = parser.parse_known_args(argv)
    return known


def main(argv=None) -> None:
    args = _parse_args(argv)
    rclpy.init()
    node = Node("goal_pose_publisher")

    # Latched for late subscribers. The controller clears commands on entering
    # GOAL_POSE, so publish after entering the state.
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    pub = node.create_publisher(Vector3, args.topic, qos)

    msg = _build_translation(args)
    pub.publish(msg)
    node.get_logger().info(
        f"Published drill-tip-relative translation on '{args.topic}': "
        f"[{msg.x:+.3f}, {msg.y:+.3f}, {msg.z:+.3f}] m"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
