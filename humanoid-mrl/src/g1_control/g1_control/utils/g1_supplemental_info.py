# TODO: redundant with other robot info in the repo; consolidate.

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .robot_supplemental_info import RobotSupplementalInfo


class WaistLocation(Enum):
    LOWER_BODY = "lower_body"
    UPPER_BODY = "upper_body"
    LOWER_AND_UPPER_BODY = "lower_and_upper_body"


class ElbowPose(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class G1SupplementalInfo(RobotSupplementalInfo):
    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
    ):
        name = "G1_G1ThreeFinger"

        body_actuated_joints = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        left_hand_actuated_joints = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
        ]

        # No right-hand finger joints: the drill is mounted at the right wrist
        # (g1_29_dof_with_hand_drill.urdf.xml). The right palm frame is kept for teleop
        # IK.
        right_hand_actuated_joints = []

        joint_limits = {
            "left_hip_pitch_joint": [-2.5307, 2.8798],
            "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576],
            "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236],
            "left_ankle_roll_joint": [-0.2618, 0.2618],
            "right_hip_pitch_joint": [-2.5307, 2.8798],
            "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576],
            "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236],
            "right_ankle_roll_joint": [-0.2618, 0.2618],
            "waist_yaw_joint": [-2.618, 2.618],
            "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [0.19, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618],
            "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, -0.19],
            "right_shoulder_yaw_joint": [-2.618, 2.618],
            "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
            "left_hand_thumb_0_joint": [-1.04719755, 1.04719755],
            "left_hand_thumb_1_joint": [-0.72431163, 1.04719755],
            "left_hand_thumb_2_joint": [0, 1.74532925],
            "left_hand_index_0_joint": [-1.57079632, 0],
            "left_hand_index_1_joint": [-1.74532925, 0],
            "left_hand_middle_0_joint": [-1.57079632, 0],
            "left_hand_middle_1_joint": [-1.74532925, 0],
        }

        joint_groups = {
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            "left_hand": {
                "joints": [
                    "left_hand_index_0_joint",
                    "left_hand_index_1_joint",
                    "left_hand_middle_0_joint",
                    "left_hand_middle_1_joint",
                    "left_hand_thumb_0_joint",
                    "left_hand_thumb_1_joint",
                    "left_hand_thumb_2_joint",
                ],
                "groups": [],
            },
            # Drill mounted at the right wrist; no finger joints.
            "right_hand": {"joints": [], "groups": []},
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        joint_name_mapping = {
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            "wrist_pitch": {
                "left": "left_wrist_pitch_joint",
                "right": "right_wrist_pitch_joint",
            },
            "wrist_roll": {
                "left": "left_wrist_roll_joint",
                "right": "right_wrist_roll_joint",
            },
            "wrist_yaw": {
                "left": "left_wrist_yaw_joint",
                "right": "right_wrist_yaw_joint",
            },
        }

        root_frame_name = "pelvis"
        hand_frame_names = {
            "left": "left_wrist_yaw_link",
            "right": "right_wrist_yaw_link",
        }
        elbow_calibration_joint_angles = {"left": 0.0, "right": 0.0}
        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
            }
        else:
            default_joint_q = {
                "shoulder_roll": {"left": 0.2, "right": -0.2},
            }

        teleop_upper_body_motion_scale = 1.0

        modified_joint_groups = joint_groups.copy()
        if waist_location == WaistLocation.UPPER_BODY:
            modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }
        elif waist_location == WaistLocation.LOWER_AND_UPPER_BODY:
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }

        t_pose = {
            "shoulder_roll": {"left": np.pi / 2, "right": -np.pi / 2},
            "elbow_pitch": {"left": np.pi / 2, "right": np.pi / 2},
        }

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=modified_joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            elbow_calibration_joint_angles=elbow_calibration_joint_angles,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
            t_pose_joint_q=t_pose,
        )
