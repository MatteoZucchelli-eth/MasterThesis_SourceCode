# MIT License
#
# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
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

import logging
from dataclasses import dataclass

import rclpy

import tyro

from g1_definitions import definitions
from rclpy.node import Node
from sensor_msgs.msg import JointState
from unitree_hg.msg import LowState


@dataclass
class StatePublisherArgs:
    """Republishes LowState messages into the JointState format expected by a lot of ROS packages."""

    publishing_rate: float = 100.0
    """Rate at which the LowState msg is republished into a JointState one."""

    input_topic: str = definitions.TOPIC_LOWSTATE
    """LowState topic to subscribe to"""


class StatePublisher(Node):

    def __init__(self, args: StatePublisherArgs):
        super().__init__("g1_state_publisher")

        self._joint_pub = self.create_publisher(JointState, "joint_states", 1)
        self._low_state_sub = self.create_subscription(
            LowState, args.input_topic, self._lowstate_cb, 1
        )

        self._pub_timer = self.create_timer(1 / args.publishing_rate, self._pub_cb)

        self._joint_state = JointState()
        self._joint_state.name = definitions.G1_URDF_JOINT_NAMES
        self._joint_state.position = [0] * len(self._joint_state.name)
        self._joint_state.velocity = [0] * len(self._joint_state.name)
        self._joint_state.effort = [0] * len(self._joint_state.name)

    def _lowstate_cb(self, msg: LowState) -> None:
        """Republish G1 joint data as JointState msg"""

        self._joint_state.header.stamp = self.get_clock().now().to_msg()

        for idx, motor_state in enumerate(msg.motor_state):
            self._joint_state.position[idx] = motor_state.q
            self._joint_state.velocity[idx] = motor_state.dq
            self._joint_state.effort[idx] = motor_state.tau_est

    def _pub_cb(self):
        """Subscribe to G1 LowState and save contents to member variable"""

        self._joint_pub.publish(self._joint_state)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] [%(asctime)s]: %(message)s",
    )

    # NOTE(@naefjo): set return_unknown_args to True because ros launch appends additional flags to cmd.
    args, unparsed_args = tyro.cli(StatePublisherArgs, return_unknown_args=True)

    logging.debug(unparsed_args)

    rclpy.init(args=unparsed_args)

    state_publisher_node = StatePublisher(args)

    try:
        while rclpy.ok():
            rclpy.spin_once(state_publisher_node)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
