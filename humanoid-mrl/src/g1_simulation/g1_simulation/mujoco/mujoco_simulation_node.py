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

import rclpy
from rclpy.executors import ExternalShutdownException

from .g1_mujoco_simulation import SimulationNode, SimulatorCliArgs


def main():
    args = SimulatorCliArgs().parse_args(known_only=True)

    rclpy.init(args=args.extra_args)

    multi_threaded_executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    g1_simulator = SimulationNode(args)

    multi_threaded_executor.add_node(g1_simulator)

    try:
        while rclpy.ok() and g1_simulator.is_active():
            multi_threaded_executor.spin_once()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        multi_threaded_executor.remove_node(g1_simulator)
        g1_simulator.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
