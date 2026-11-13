from __future__ import annotations

import os

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from g1_definitions import definitions
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class FakeCameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_camera_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("topic_name", definitions.TOPIC_CAMERA)
        self.declare_parameter("publish_rate", 20.0)

        video_path = self.get_parameter("video_path").value
        topic_name = self.get_parameter("topic_name").value
        publish_rate = self.get_parameter("publish_rate").value

        if not video_path:
            video_path = os.path.join(
                get_package_share_directory("g1_sensors"),
                "videos",
                "fake_camera.mp4",
            )

        self._video_path = str(video_path)
        self._cap = cv2.VideoCapture(self._video_path)
        self._bridge = CvBridge()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(Image, str(topic_name), qos)

        self.create_timer(1.0 / publish_rate, self._publish_frame)

        if not self._cap.isOpened():
            self.get_logger().error(
                f"Failed to open video file: {self._video_path}. "
                "Place a video there or pass video_path via launch."
            )
        else:
            self.get_logger().info(
                f"Publishing fake camera stream from {self._video_path} "
                f"to topic {topic_name} at {publish_rate} Hz"
            )

    def _publish_frame(self) -> None:
        if not self._cap.isOpened():
            return

        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                self.get_logger().warn("Failed to read frame from video source")
                return

        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        self._pub.publish(msg)

    def destroy_node(self) -> bool:
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = FakeCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
