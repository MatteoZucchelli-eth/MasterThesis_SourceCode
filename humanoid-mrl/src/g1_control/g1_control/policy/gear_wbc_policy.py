import collections
import logging
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from ..utils.utils import get_gravity_orientation

logger = logging.getLogger(__name__)


class G1GearWbcPolicy:
    """Lower-body locomotion policy using two ONNX models (balance + walk)."""

    def __init__(
        self,
        robot_model,
        config: dict,
        balance_model_path: str,
        walk_model_path: str,
    ):
        self.config = config
        self.robot_model = robot_model

        self.balance_policy = self._load_onnx_policy(balance_model_path)
        self.walk_policy = self._load_onnx_policy(walk_model_path)

        self.observation = None
        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)

        self.use_policy_action = False
        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
        self.target_dof_pos = self.config["default_angles"].copy()
        self.cmd = self.config["cmd_init"].copy()
        self.height_cmd = float(self.config["height_cmd"])
        self.freq_cmd = float(self.config["freq_cmd"])
        self.roll_cmd = float(self.config["rpy_cmd"][0])
        self.pitch_cmd = float(self.config["rpy_cmd"][1])
        self.yaw_cmd = float(self.config["rpy_cmd"][2])
        self.gait_indices = np.zeros(1, dtype=np.float32)

    def _load_onnx_policy(self, model_path: str):
        logger.info("Loading ONNX policy from %s", model_path)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def run_inference(input_tensor):
            ort_inputs = {input_name: input_tensor}
            ort_outs = session.run(None, ort_inputs)
            return ort_outs[0]

        logger.info("Successfully loaded ONNX policy from %s", model_path)
        return run_inference

    def set_observation(self, observation: Dict[str, Any]):
        self.observation = observation

        self.gait_indices = np.remainder(self.gait_indices + 0.02 * self.freq_cmd, 1.0)
        durations = np.full_like(self.gait_indices, 0.5)
        phases = 0.5
        foot_indices = [
            self.gait_indices + phases,
            self.gait_indices,
        ]
        self.foot_indices = np.remainder(np.stack(foot_indices, axis=1), 1.0)
        for fi in foot_indices:
            stance = fi < durations
            swing = fi >= durations
            fi[stance] = fi[stance] * (0.5 / durations[stance])
            fi[swing] = 0.5 + (fi[swing] - durations[swing]) * (
                0.5 / (1 - durations[swing])
            )

        body_indices = self.robot_model.get_joint_group_indices("body")
        n_joints = len(body_indices)

        qj = observation["q"][body_indices].copy()
        dqj = observation["dq"][body_indices].copy()
        quat = observation["floating_base_pose"][3:7].copy()
        omega = observation["floating_base_vel"][3:6].copy()

        if len(self.config["default_angles"]) < n_joints:
            padded_defaults = np.zeros(n_joints, dtype=np.float32)
            padded_defaults[: len(self.config["default_angles"])] = self.config[
                "default_angles"
            ]
        else:
            padded_defaults = self.config["default_angles"][:n_joints]

        qj_scaled = (qj - padded_defaults) * self.config["dof_pos_scale"]
        dqj_scaled = dqj * self.config["dof_vel_scale"]
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.config["ang_vel_scale"]

        single_obs_dim = 86
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)
        single_obs[0:3] = self.cmd[:3] * self.config["cmd_scale"]
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = np.array([self.roll_cmd, self.pitch_cmd, self.yaw_cmd])
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity_orientation
        single_obs[13 : 13 + n_joints] = qj_scaled
        single_obs[13 + n_joints : 13 + 2 * n_joints] = dqj_scaled
        single_obs[13 + 2 * n_joints : 13 + 2 * n_joints + 15] = self.action

        self.obs_history.append(single_obs)
        while len(self.obs_history) < self.config["obs_history_len"]:
            # Warm-start with the live frame; zero-padding is out of distribution.
            self.obs_history.appendleft(single_obs.copy())

        for i, hist_obs in enumerate(self.obs_history):
            start = i * single_obs_dim
            self.obs_buffer[start : start + single_obs_dim] = hist_obs

        self.obs_tensor = self.obs_buffer[np.newaxis, :]  # shape (1, N)

    def set_goal(self, goal: Dict[str, Any]):
        if "toggle_policy_action" in goal and goal["toggle_policy_action"]:
            self.use_policy_action = not self.use_policy_action

    def get_action(
        self,
        time: Optional[float] = None,
        arms_target_pose: Optional[np.ndarray] = None,
        base_height_command: Optional[np.ndarray] = None,
        torso_orientation_rpy: Optional[np.ndarray] = None,
        interpolated_navigate_cmd: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if np.linalg.norm(self.cmd) < 0.05:
            policy = self.balance_policy
        else:
            policy = self.walk_policy
        self.action = policy(self.obs_tensor).squeeze()

        if self.use_policy_action:
            cmd_q = (
                self.action * self.config["action_scale"]
                + self.config["default_angles"]
            )
        else:
            cmd_q = self.observation["q"][
                self.robot_model.get_joint_group_indices("lower_body")
            ]

        cmd_dq = np.zeros(self.config["num_actions"])
        cmd_tau = np.zeros(self.config["num_actions"])
        return {"body_action": (cmd_q, cmd_dq, cmd_tau)}

    def handle_keyboard_button(self, key: str):
        if key == "]":
            self.use_policy_action = True
        elif key == "o":
            self.use_policy_action = False
        elif key == "w":
            self.cmd[0] += 0.2
        elif key == "s":
            self.cmd[0] -= 0.2
        elif key == "a":
            self.cmd[1] += 0.2
        elif key == "d":
            self.cmd[1] -= 0.2
        elif key == "q":
            self.cmd[2] += 0.2
        elif key == "e":
            self.cmd[2] -= 0.2
        elif key == "z":
            self.cmd[:] = 0.0
        elif key == "1":
            self.height_cmd += 0.1
        elif key == "2":
            self.height_cmd -= 0.1
