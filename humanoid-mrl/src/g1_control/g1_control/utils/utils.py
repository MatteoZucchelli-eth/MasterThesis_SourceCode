import logging

import numpy as np
import yaml

from g1_definitions import definitions
from g1_definitions.definitions import VLA_CHUNK_SIZE

logger = logging.getLogger(__name__)


def resolve_eef_frames(robot_model) -> tuple[str, str]:
    """Return the (left, right) end-effector frames the IK drives.

    Right is the drill tip; without the drill URDF variant it falls back to the
    right palm with a warning.
    """
    right_frame = definitions.RIGHT_EEF_FRAME
    if not robot_model.has_frame(right_frame):
        right_frame = definitions.RIGHT_PALM_FRAME
        logger.warning(
            "'%s' not in the URDF — falling back to '%s' as the right end-effector. "
            "State/action will be palm poses, not drill-tip poses.",
            definitions.RIGHT_EEF_FRAME,
            right_frame,
        )
    return definitions.LEFT_EEF_FRAME, right_frame


def pr_slice(arr: np.ndarray, group: str) -> np.ndarray:
    """Return the slice of a PR-ordered joint array for a named joint group.

    ``group`` indexes :data:`definitions.JOINT_PR_SLICES` (e.g. ``"left_arm"``,
    ``"right_arm"``, ``"waist"``).
    """
    start, end = definitions.JOINT_PR_SLICES[group]
    return arr[start:end]


def project_on_sphere(
    center: np.ndarray, radius: float, point: np.ndarray
) -> np.ndarray:
    direction = point - center
    m = direction / np.linalg.norm(direction)
    return center + m * radius


def create_action_buffer(
    chunk_size: int = VLA_CHUNK_SIZE, num_upper_body_joints: int = 28
) -> dict:
    """Allocate a zeroed action-chunk buffer used by VLA inference."""
    return {
        definitions.RIGHT_EEF_KEY: np.zeros((chunk_size, 9), dtype=np.float32),
        "upper_body_joint_chunk": np.zeros(
            (chunk_size, num_upper_body_joints), dtype=np.float32
        ),
    }


def load_config(config_path: str) -> dict:
    """Load and process the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    array_keys = [
        "kps",
        "kds",
        "motor_kps",
        "motor_kds",
        "default_angles",
        "cmd_scale",
        "cmd_init",
    ]
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    return config


def get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Get gravity vector in body frame. quat = [w, x, y, z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    q_conj = np.array([w, -x, -y, -z])
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return np.array(
        [
            gravity_vec[0]
            * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + gravity_vec[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + gravity_vec[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            gravity_vec[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + gravity_vec[1]
            * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + gravity_vec[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            gravity_vec[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + gravity_vec[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + gravity_vec[2]
            * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ]
    )
