# MIT License
# Base Controller Interface for Unitree G1

from __future__ import annotations

from abc import ABC, abstractmethod

from g1_common.data import RobotState
from unitree_hg.msg import LowCmd


class BaseController(ABC):
    """
    Abstract base class for all FSM state controllers.
    Enforces a strict interface that all child controllers must implement.
    """

    def __init__(self, robot_state: RobotState, gains: dict):
        self.robot_state = robot_state
        self.gains = gains

    @abstractmethod
    def on_activate(self) -> None:
        """
        Called exactly once when the FSM transitions into this controller's state.
        Use this to reset timers, target interpolation, or clear old buffers.
        """
        pass

    @abstractmethod
    def get_control(self) -> LowCmd | None:
        """
        Called at the high-frequency control rate (e.g., 50Hz).
        Reads self.robot_state and returns the motor commands.

        Returns:
            LowCmd message to be published, or None if no command should be sent.
        """
        pass
