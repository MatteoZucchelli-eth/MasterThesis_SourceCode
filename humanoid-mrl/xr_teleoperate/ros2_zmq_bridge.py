#!/usr/bin/env python3
"""
ros2_zmq_bridge.py  —  Python 3.12 / ROS2 Jazzy

Subscribes to a ROS2 sensor_msgs/Image topic, JPEG-encodes each frame, and
pushes it over ZMQ to the teleimager ZMQSourceCamera.

Requirements (Python 3.12 env with ROS2 sourced):
    source /opt/ros/jazzy/setup.bash
    pip install pyzmq opencv-python

Usage:
    python3 ros2_zmq_bridge.py
    python3 ros2_zmq_bridge.py --topic /camera/d435/color/image_raw --host 127.0.0.1 --port 55590
"""
import argparse
import sys

import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(
        description="Bridge a ROS2 image topic to teleimager via ZMQ"
    )
    parser.add_argument(
        "--topic",
        default="/camera/d435/color/image_raw",
        help="ROS2 sensor_msgs/Image topic to subscribe to",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host running teleimager (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=55590,
        help="zmq_source_port configured in cam_config_server.yaml (default: 55590)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="JPEG encode quality 1-100 (default: 80)",
    )
    args = parser.parse_args()

    # ZMQ PUSH socket — connects to the teleimager PULL socket
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)  # drop stale frames instead of buffering
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(
        f"[ros2_zmq_bridge] ZMQ PUSH connected to tcp://{args.host}:{args.port}",
        flush=True,
    )

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError as e:
        print(
            f"[ros2_zmq_bridge] Import error: {e}\n"
            "Make sure you have sourced ROS2 and are running with Python 3.12.",
            file=sys.stderr,
        )
        sock.close()
        ctx.term()
        sys.exit(1)

    bridge = CvBridge()

    rclpy.init()
    node = Node("ros2_zmq_bridge")

    def callback(msg):
        try:
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(".jpg", bgr, encode_params)
            if ok:
                sock.send(buf.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass  # teleimager not ready yet — skip frame
        except Exception as e:
            node.get_logger().warning(f"Bridge error: {e}")

    node.create_subscription(Image, args.topic, callback, 1)
    node.get_logger().info(
        f"[ros2_zmq_bridge] Subscribed to '{args.topic}' → tcp://{args.host}:{args.port}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
