# MIT License
# Ready Controller for Unitree G1

from __future__ import annotations

import numpy as np
import pinocchio as pin
from g1_common.data import RobotState
from g1_definitions import definitions
from unitree_hg.msg import LowCmd

from g1_control.base import BaseController
from g1_control.utils import build_low_cmd
from g1_control.utils.wbc_robot_model import RobotModel


class ReadyController(BaseController):
    """
    Smoothly interpolates the robot from its current pose into the default
    starting pose over a fixed duration, gradually ramping up stiffness.
    """

    def __init__(
        self,
        robot_state: RobotState,
        gains: dict,
        stiffness_array: np.ndarray,
        damping_array: np.ndarray,
        default_joint_pos: np.ndarray,
        control_dt: float,
        robot_model: RobotModel | None = None,
    ):
        super().__init__(robot_state, gains)

        self.stiffness_array = stiffness_array
        self.damping_array = damping_array
        self.default_joint_pos = default_joint_pos
        self._control_dt = control_dt

        # Arm gravity feedforward, matching WholeBodyRLController, so the arms
        # do not sag g(q)/kp below the default pose.
        self._robot_model = robot_model
        self._grav_comp_mask = np.zeros(definitions.NUM_MOTOR_G1_BASE, dtype=bool)
        for key in ("left_arm", "right_arm"):
            s, e = definitions.JOINT_PR_SLICES[key]
            self._grav_comp_mask[s:e] = True

        self._ready_state_transition_timer = 0.0
        self._ready_state_transition_duration = 10.0  # [s]
        # Ramp start pose, latched on the first tick of each entry.
        self._start_pos: np.ndarray | None = None

    def on_activate(self) -> None:
        """Called when transitioning into READY state to reset the ramp."""
        self._ready_state_transition_timer = 0.0
        # Latched in get_control: joint_pos may still be None here.
        self._start_pos = None

    def get_control(self) -> LowCmd | None:
        """Generates the interpolated motor commands."""

        if self.robot_state.joint_pos is None:
            return None

        # Latch the start pose once per entry; re-reading joint_pos every tick
        # would turn the ramp into a ratio^2 P-controller that absorbs drift.
        if self._start_pos is None:
            self._start_pos = np.asarray(
                self.robot_state.joint_pos, dtype=np.float64
            ).copy()

        self._ready_state_transition_timer += self._control_dt
        ratio = float(
            np.clip(
                self._ready_state_transition_timer
                / self._ready_state_transition_duration,
                0.0,
                1.0,
            )
        )

        # Linear joint-space ramp to the default pose; kp ramps, kd is constant.
        target_pos = (
            1.0 - ratio
        ) * self._start_pos + ratio * self.default_joint_pos

        tau = None
        if self._robot_model is not None:
            q_full = self._robot_model.get_configuration_from_actuated_joints(
                self.robot_state.joint_pos
            )
            g_full = pin.computeGeneralizedGravity(
                self._robot_model.pinocchio_wrapper.model,
                self._robot_model.pinocchio_wrapper.data,
                q_full,
            )
            g_PR = self._robot_model.get_body_actuated_joints(g_full)
            tau = np.where(self._grav_comp_mask, g_PR, 0.0)

        return build_low_cmd(
            self.robot_state.mode_machine,
            q=target_pos,
            kp=ratio * self.stiffness_array,
            kd=self.damping_array,
            tau=tau,
        )
