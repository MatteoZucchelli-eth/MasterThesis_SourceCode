# MIT License
# Damping Controller for Unitree G1

from __future__ import annotations

import numpy as np
from g1_common.data import RobotState
from g1_definitions import definitions
from unitree_hg.msg import LowCmd

from g1_control.base import BaseController
from g1_control.utils import build_low_cmd


class DampingController(BaseController):
    """
    Sets all motors to zero stiffness and applies only damping.
    Used as the safe default / fallback state.
    """

    def __init__(self, robot_state: RobotState, gains: dict, damping_array: np.ndarray):
        super().__init__(robot_state, gains)
        self.damping_array = damping_array

    def on_activate(self) -> None:
        """Called when transitioning into DAMPING state."""
        pass

    def get_control(self) -> LowCmd | None:
        """Generates the zero-stiffness, pure-damping command."""
        n = definitions.NUM_MOTOR_G1_BASE
        return build_low_cmd(
            self.robot_state.mode_machine,
            q=np.zeros(n),
            kp=np.zeros(n),
            kd=self.damping_array,
        )
