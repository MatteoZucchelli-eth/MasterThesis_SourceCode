import rclpy
from rclpy.executors import ExternalShutdownException

from .fsm import G1FSM


def main():
    rclpy.init()
    fsm = G1FSM()

    try:
        rclpy.spin(fsm)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        fsm.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
