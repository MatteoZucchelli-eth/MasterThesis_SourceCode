from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RobotState:
    """Data class storing relevant robot data to compute the observations."""

    joint_pos: np.ndarray | None = None
    """Current robot joint positions"""

    joint_vel: np.ndarray | None = None
    """Current robot joint velocities"""

    joint_torque: np.ndarray | None = None
    """Reported ``tau_est`` values from LowState, in PR order.

    Isaac Sim supplies projected joint forces, MuJoCo supplies joint actuator
    forces, and the real G1 supplies Unitree's motor-torque estimate.
    """

    quat: Sequence[float] | None = None
    """Current robot base quaternion in [w, x, y, z] convention"""

    gyro: Sequence[float] | None = None
    """Current robot base angular velocity in body frame"""

    last_action: np.ndarray | None = None
    """Last action applied to the robot"""

    twist_command: np.ndarray | None = None
    """Command to the robot"""

    height_command: float | None = None
    """Height command to the robot"""

    upper_body_target: np.ndarray | None = None
    """Target upper-body joint positions (set by demo/VLA nodes)"""

    upper_body_target_chunk: np.ndarray | None = None
    """Target upper-body joint positions chunk (set by VLA nodes)"""

    mode_machine: int = 5
    """Mode machine value echoed from LowState and forwarded to LowCmd"""
