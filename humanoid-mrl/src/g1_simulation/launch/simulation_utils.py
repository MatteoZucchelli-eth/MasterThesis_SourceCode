# MIT License
#
# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
#
# Portions of this file are derived from:
# Livox ROS2 Driver (https://github.com/tu-darmstadt-ros-pkg/livox_ros_driver2/tree/jazzy)
# Copyright (c) Livox. Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from g1_definitions import definitions

from launch import LaunchDescription
from launch_ros.actions import Node

# JPEG quality for the sim's compressed mirrors; matches the real cameras
# (docker/head_camera.yaml, g1_cam_drill_interface/config/hand_camera.yaml).
_JPEG_QUALITY = 60


def _compressed_mirror(raw_topic: str, name: str) -> Node:
    """Mirror a raw sim camera stream onto its ``/compressed`` transport.

    For remote viewers; local consumers keep reading raw. The republisher is
    lazy (no raw subscription until a compressed subscriber exists). Both
    endpoints are BEST_EFFORT / KEEP_LAST(1): the bridge publishes BEST_EFFORT,
    which a RELIABLE subscription would never match.
    """
    compressed_topic = raw_topic + "/compressed"
    return Node(
        package="image_transport",
        executable="republish",
        name=name,
        output="screen",
        parameters=[
            {
                "in_transport": "raw",
                "out_transport": "compressed",
                "out.compressed.jpeg_quality": _JPEG_QUALITY,
                f"qos_overrides.{raw_topic}.subscription.reliability": "best_effort",
                f"qos_overrides.{raw_topic}.subscription.depth": 1,
                f"qos_overrides.{compressed_topic}.publisher.reliability": "best_effort",
                f"qos_overrides.{compressed_topic}.publisher.depth": 1,
            }
        ],
        # republish ignores a remap of `out`; remap `out/compressed` instead.
        remappings=[
            ("in", raw_topic),
            ("out/compressed", compressed_topic),
        ],
    )


def generate_launch_description():
    lf_lowstate_node = Node(
        package="topic_tools",
        executable="throttle",
        name="lowstate_throttle",
        output="screen",
        arguments=[
            "messages",
            definitions.TOPIC_LOWSTATE,
            "10.0",
            definitions.TOPIC_LOWSTATE_THROTTLED,
        ],
    )

    head_camera_compressed = _compressed_mirror(
        definitions.TOPIC_CAMERA, "head_camera_compressed"
    )
    wrist_camera_compressed = _compressed_mirror(
        definitions.TOPIC_CAMERA_WRIST, "wrist_camera_compressed"
    )

    return LaunchDescription(
        [
            lf_lowstate_node,
            head_camera_compressed,
            wrist_camera_compressed,
        ]
    )
