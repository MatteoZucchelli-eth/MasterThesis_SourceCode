import logging
import sys
import threading

import rclpy

from g1_definitions import definitions
from rclpy.node import Node
from std_msgs.msg import String


class TaskInstructionNode(Node):
    """
    ROS2 node to handle task publishing from terminal input.
    """

    def __init__(self):
        super().__init__(
            "task_prompt_node",
            automatically_declare_parameters_from_overrides=True,
        )

        # ── ROS Interfaces ────────────────────────────────────────
        self.task_pub = self.create_publisher(
            String, definitions.TOPIC_TASKINSTRUCTION, 1
        )

        # ── Threading ─────────────────────────────────────────────
        # Daemon thread so input() does not block rclpy.spin().
        self.input_thread = threading.Thread(target=self.listen_for_task, daemon=True)
        self.input_thread.start()

    def listen_for_task(self):
        self.get_logger().info(
            "Ready. Type a task and press Enter to publish (or 'quit' to exit)."
        )

        while rclpy.ok():
            try:
                user_input = input()

                if user_input.strip().lower() == "quit":
                    self.get_logger().info("Shutting down node...")
                    rclpy.shutdown()
                    sys.exit(0)

                if user_input.strip():
                    msg = String()
                    msg.data = user_input
                    self.task_pub.publish(msg)
                    self.get_logger().info(f"Published task: '{msg.data}'")

            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"Error reading input: {e}")


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)
    node = TaskInstructionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
