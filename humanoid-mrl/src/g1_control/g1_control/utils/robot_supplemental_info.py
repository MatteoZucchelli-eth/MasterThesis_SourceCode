from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np


@dataclass
class RobotSupplementalInfo:
    """
    Base class for robot-specific information not easily extractable from URDF.
    """

    name: str
    body_actuated_joints: List[str]
    left_hand_actuated_joints: List[str]
    right_hand_actuated_joints: List[str]
    joint_groups: Dict[str, Dict[str, List[str]]]
    root_frame_name: str
    hand_frame_names: Dict[str, str]
    joint_limits: Dict[str, List[float]]
    elbow_calibration_joint_angles: Dict[str, float]
    joint_name_mapping: Dict[str, Union[str, Dict[str, str]]]
    default_joint_q: Dict[str, Union[float, Dict[str, float]]]
    hand_rotation_correction: np.ndarray
    teleop_upper_body_motion_scale: float
    t_pose_joint_q: Dict[str, Union[float, Dict[str, float]]] = field(
        default_factory=dict
    )
