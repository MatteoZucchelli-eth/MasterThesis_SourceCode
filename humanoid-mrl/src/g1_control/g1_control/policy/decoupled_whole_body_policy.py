import time as time_module
from typing import Optional

import numpy as np
from pinocchio import rpy

_DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]


class G1DecoupledWholeBodyPolicy:
    """
    Combines a lower-body locomotion policy (G1GearWbcPolicy) and an upper-body
    interpolation policy into a single whole-body action.
    """

    def __init__(self, robot_model, lower_body_policy, upper_body_policy):
        self.robot_model = robot_model
        self.lower_body_policy = lower_body_policy
        self.upper_body_policy = upper_body_policy
        self.last_goal_time = time_module.monotonic()

    def set_observation(self, observation):
        self.lower_body_policy.set_observation(observation)

    def set_goal(self, goal: dict):
        self.last_goal_time = time_module.monotonic()

        upper_body_goal = {}
        lower_body_goal = {}

        upper_body_keys = [
            "target_upper_body_pose",
            "base_height_command",
            "target_time",
            "interpolation_garbage_collection_time",
            "navigate_cmd",
        ]
        for key in upper_body_keys:
            if key in goal:
                upper_body_goal[key] = goal[key]

        if "navigate_cmd" not in upper_body_goal:
            if "target_time" in goal and isinstance(goal["target_time"], list):
                upper_body_goal["navigate_cmd"] = [np.array(_DEFAULT_NAV_CMD)] * len(
                    goal["target_time"]
                )
            else:
                upper_body_goal["navigate_cmd"] = np.array(_DEFAULT_NAV_CMD)

        lower_body_keys = ["toggle_stand_command", "toggle_policy_action"]
        for key in lower_body_keys:
            if key in goal:
                lower_body_goal[key] = goal[key]

        self.upper_body_policy.set_goal(upper_body_goal)
        self.lower_body_policy.set_goal(lower_body_goal)

    def get_action(self, time: Optional[float] = None) -> dict:
        current_time = time if time is not None else time_module.monotonic()

        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        q = np.zeros(self.robot_model.num_dofs)

        upper_body_action = self.upper_body_policy.get_action(current_time)
        q[upper_body_indices] = upper_body_action["target_upper_body_pose"]
        # Upper-body reference velocity from the spline; the lower-body RL policy
        # emits positions only, so its entries stay zero.
        dq = np.zeros(self.robot_model.num_dofs)
        dq[upper_body_indices] = self.upper_body_policy.get_velocity(current_time)[
            "target_upper_body_pose"
        ]
        q_arms = q[self.robot_model.get_joint_group_indices("arms")]
        base_height_command = upper_body_action.get("base_height_command", None)
        interpolated_navigate_cmd = upper_body_action.get("navigate_cmd", None)

        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        torso_orientation = self.robot_model.frame_placement("torso_link").rotation
        waist_orientation = self.robot_model.frame_placement("pelvis").rotation
        waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
        waist_yaw_only_rotation = rpy.rpyToMatrix(0, 0, waist_yaw)
        yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
        torso_orientation_rpy = rpy.matrixToRpy(yaw_only_waist_from_torso)

        lower_body_action = self.lower_body_policy.get_action(
            current_time,
            q_arms,
            base_height_command,
            torso_orientation_rpy,
            interpolated_navigate_cmd,
        )

        q[lower_body_indices] = lower_body_action["body_action"][0][
            : len(lower_body_indices)
        ]
        return {"q": q, "dq": dq}

    def activate_policy(self):
        """Enable the RL locomotion policy (called on FSM transition to MOVE)."""
        self.lower_body_policy.handle_keyboard_button("]")
