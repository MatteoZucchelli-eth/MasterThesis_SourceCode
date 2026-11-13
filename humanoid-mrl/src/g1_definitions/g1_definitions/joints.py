# MIT License
"""G1 kinematic constants: joint counts, names, orderings, indices and limits."""

from enum import Enum

MOTOR_SENSOR_NUM = 3
"""Motor-state fields per joint: q, dq and tau_est."""

NUM_MOTOR_G1_BASE = 29
"""#DoF of the G1 base"""

NUM_MOTOR_DEX3_1 = 7
"""#DoF of the Dex3-1 hand"""

NUM_MOTOR_G1_TOTAL = NUM_MOTOR_G1_BASE + 2 * NUM_MOTOR_DEX3_1
LEFT_HAND_START_IDX = 29
"""Start index of the left hand data in the mj_data struct"""

RIGHT_HAND_START_IDX = 36
"""Start index of the right hand data in the mj_data struct"""

JOINT_PR_SLICES: dict[str, tuple[int, int]] = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
}
"""Start/end indices into the 29-dim PR joint position vector for each body part."""


class G1JointControlMode(Enum):
    """
    Values for the mode_pr setting in the unitree LowCmd message.

    For details refer to the [documentation](https://support.unitree.com/home/en/G1_developer/basic_motion_routine)
    """

    PR_Mode = 0
    AB_Mode = 1


G1_BASE_PR_JOINT_NAMES = [
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
"""List of base G1 joint names in PR mode"""
G1_DEX3_1_JOINT_NAMES = [
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "middle_0_joint",
    "middle_1_joint",
    "index_0_joint",
    "index_1_joint",
]
G1_DEX3_1_LEFT_JOINT_NAMES = [
    f"left_hand_{joint_name}" for joint_name in G1_DEX3_1_JOINT_NAMES
]
"""List of Joint names of G1's left hand"""
G1_DEX3_1_RIGHT_JOINT_NAMES = [
    f"right_hand_{joint_name}" for joint_name in G1_DEX3_1_JOINT_NAMES
]
"""List of Joint names of G1's right hand"""
G1_URDF_JOINT_NAMES = (
    G1_BASE_PR_JOINT_NAMES
    + ["left_hand_palm_joint"]
    + G1_DEX3_1_LEFT_JOINT_NAMES
    + ["right_hand_palm_joint"]
    + G1_DEX3_1_RIGHT_JOINT_NAMES
)
"""List of all Joint names in the G1 URDF"""

DOF_POS_LOWER_LIMIT_LIST = [
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -2.9671,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.61443,
    -1.61443,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.61443,
    -1.61443,
]
"""Lower position limits for the joints in the ordering of G1_BASE_PR_JOINT_NAMES

source: https://github.com/amazon-far/holosoma/blob/e1490cb7ac7aedc2c372ce9d0eda0551f1fbace9/src/holosoma/holosoma/config_values/robot.py#L119-L180
"""

DOF_POS_UPPER_LIMIT_LIST = [
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    0.5236,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.972222054,
    1.61443,
    1.61443,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.972222054,
    1.61443,
    1.61443,
]
"""Upper position limits for the joints in the ordering of G1_BASE_PR_JOINT_NAMES

source: https://github.com/amazon-far/holosoma/blob/e1490cb7ac7aedc2c372ce9d0eda0551f1fbace9/src/holosoma/holosoma/config_values/robot.py#L119-L180
"""


class G1JointIndexPR(Enum):
    """G1 Joint Indices for PR mode"""

    LEFT_HIP_PITCH = 0
    LEFT_HIP_ROLL = 1
    LEFT_HIP_YAW = 2
    LEFT_KNEE = 3
    LEFT_ANKLE_PITCH = 4
    LEFT_ANKLE_ROLL = 5
    RIGHT_HIP_PITCH = 6
    RIGHT_HIP_ROLL = 7
    RIGHT_HIP_YAW = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE_PITCH = 10
    RIGHT_ANKLE_ROLL = 11
    WAIST_YAW = 12
    WAIST_ROLL = 13
    WAIST_PITCH = 14
    LEFT_SHOULDER_PITCH = 15
    LEFT_SHOULDER_ROLL = 16
    LEFT_SHOULDER_YAW = 17
    LEFT_ELBOW = 18
    LEFT_WRIST_ROLL = 19
    LEFT_WRIST_PITCH = 20
    LEFT_WRIST_YAW = 21
    RIGHT_SHOULDER_PITCH = 22
    RIGHT_SHOULDER_ROLL = 23
    RIGHT_SHOULDER_YAW = 24
    RIGHT_ELBOW = 25
    RIGHT_WRIST_ROLL = 26
    RIGHT_WRIST_PITCH = 27
    RIGHT_WRIST_YAW = 28


class G1JointIndexAB(Enum):
    """G1 Joint Indices for AB mode"""

    LEFT_HIP_PITCH = 0
    LEFT_HIP_ROLL = 1
    LEFT_HIP_YAW = 2
    LEFT_KNEE = 3
    LEFT_ANKLE_B = 4
    LEFT_ANKLE_A = 5
    RIGHT_HIP_PITCH = 6
    RIGHT_HIP_ROLL = 7
    RIGHT_HIP_YAW = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE_B = 10
    RIGHT_ANKLE_A = 11
    WAIST_YAW = 12
    WAIST_A = 13
    WAIST_B = 14
    LEFT_SHOULDER_PITCH = 15
    LEFT_SHOULDER_ROLL = 16
    LEFT_SHOULDER_YAW = 17
    LEFT_ELBOW = 18
    LEFT_WRIST_ROLL = 19
    LEFT_WRIST_PITCH = 20
    LEFT_WRIST_YAW = 21
    RIGHT_SHOULDER_PITCH = 22
    RIGHT_SHOULDER_ROLL = 23
    RIGHT_SHOULDER_YAW = 24
    RIGHT_ELBOW = 25
    RIGHT_WRIST_ROLL = 26
    RIGHT_WRIST_PITCH = 27
    RIGHT_WRIST_YAW = 28


class HandIndices(Enum):
    """G1 Hand joint indices"""

    THUMB_0 = 0
    THUMB_1 = 1
    THUMB_2 = 2
    MIDDLE_0 = 3
    MIDDLE_1 = 4
    INDEX_0 = 5
    INDEX_1 = 6
