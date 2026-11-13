# MIT License
"""Zero-torque controller for the unpowered control-stack state."""

from __future__ import annotations

import numpy as np
from g1_common.data import RobotState
from g1_definitions import definitions
from unitree_hg.msg import LowCmd

from g1_control.base import BaseController
from g1_control.utils import build_low_cmd


class ZeroTorqueController(BaseController):
    """Command zero stiffness, damping, velocity, and feedforward torque."""

    def __init__(self, robot_state: RobotState, gains: dict):
        super().__init__(robot_state, gains)

    def on_activate(self) -> None:
        """The stateless controller needs no activation work."""

    def get_control(self) -> LowCmd:
        """Return a command that requests no actuator torque."""
        zeros = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        return build_low_cmd(
            self.robot_state.mode_machine,
            q=zeros,
            dq=zeros,
            kp=zeros,
            kd=zeros,
            tau=zeros,
        )
