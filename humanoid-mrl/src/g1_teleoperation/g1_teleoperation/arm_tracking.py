# MIT License
"""XR → robot arm tracking: calibration and palm-pose mapping (pure math)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def xr_flat_to_matrix(flat_list) -> np.ndarray:
    """Reshape the XR payload's flat 16-float pose into a 4x4 matrix."""
    return np.array(flat_list, dtype=np.float64).reshape(4, 4)


class ArmTracker:
    """Maps VR (Quest) palm poses onto robot pelvis-frame palm targets.

    Lifecycle: :meth:`calibrate` captures the operator's reference pose, then
    :attr:`needs_robot_poses` asks the caller for the matching robot palm poses
    (:meth:`set_robot_references` computes the isotropic scale and rotation
    alignment), after which :meth:`palm_targets` maps every new payload.
    """

    def __init__(self) -> None:
        self.scale: float | None = None
        self.needs_robot_poses = False

        self._human_reference_left: np.ndarray | None = None
        self._human_reference_right: np.ndarray | None = None
        self._robot_reference_left: np.ndarray | None = None
        self._robot_reference_right: np.ndarray | None = None
        self._human_left_rot_local_calib: Rotation | None = None
        self._human_right_rot_local_calib: Rotation | None = None
        self._q_calib_global_to_local: Rotation | None = None
        self._m_align_left: Rotation | None = None
        self._m_align_right: Rotation | None = None

    def reset(self) -> None:
        self.scale = None
        self.needs_robot_poses = False

    def calibrate(self, payload: dict) -> None:
        """Capture the operator's head/palm reference from the current payload."""
        head_pose = xr_flat_to_matrix(payload["head_pose"])
        left_pose = xr_flat_to_matrix(payload["left_pose"])
        right_pose = xr_flat_to_matrix(payload["right_pose"])

        pos_head = head_pose[:3, 3]
        pos_left = left_pose[:3, 3]
        pos_right = right_pose[:3, 3]

        q_global_to_local = self._global_to_local(head_pose)
        self._q_calib_global_to_local = q_global_to_local

        # q_local_to_palm = q_global_to_local * q_global_to_palm
        self._human_left_rot_local_calib = q_global_to_local * Rotation.from_matrix(
            left_pose[:3, :3]
        )
        self._human_right_rot_local_calib = q_global_to_local * Rotation.from_matrix(
            right_pose[:3, :3]
        )

        self.needs_robot_poses = True

        self._human_reference_left = pos_left - pos_head
        self._human_reference_right = pos_right - pos_head
        np.set_printoptions(precision=3, suppress=True)
        print(f"Quest left ref:  {self._human_reference_left}")
        print(f"Quest right ref: {self._human_reference_right}")

    @staticmethod
    def _global_to_local(head_pose: np.ndarray) -> Rotation:
        """Static calibration frame: yaw-only head forward, world-up z."""
        head_rot_world = head_pose[:3, :3]
        head_forward = -head_rot_world[:, 2]
        # Flatten the vector to represent the yaw only, then re-normalize.
        head_forward[0] = 0.0
        x_local = head_forward / np.linalg.norm(head_forward)
        z_local = np.array([-1, 0, 0])
        y_local = np.cross(z_local, x_local)

        # Rows are the local axes.
        return Rotation.from_matrix(np.vstack((x_local, y_local, z_local)))

    def set_robot_references(self, left_4x4: np.ndarray, right_4x4: np.ndarray) -> None:
        """Compute scale and rotation alignment from the robot's palm poses.

        Scale = |robot_right - robot_left| / |quest_right - quest_left|
        """
        self._robot_reference_left = left_4x4[:3, 3]
        self._robot_reference_right = right_4x4[:3, 3]
        q_robot_left = Rotation.from_matrix(left_4x4[:3, :3])
        q_robot_right = Rotation.from_matrix(right_4x4[:3, :3])

        self._m_align_left = self._human_left_rot_local_calib.inv() * q_robot_left
        self._m_align_right = self._human_right_rot_local_calib.inv() * q_robot_right

        robot_span = np.linalg.norm(
            self._robot_reference_right - self._robot_reference_left
        )
        quest_span = np.linalg.norm(
            self._human_reference_right - self._human_reference_left
        )
        self.scale = robot_span / quest_span if quest_span > 1e-6 else 1.0
        self.needs_robot_poses = False
        print(
            f"Isotropic scale: {self.scale:.4f}  "
            f"(robot span {robot_span:.3f} m, quest span {quest_span:.3f} m)"
        )

    def palm_targets(self, payload: dict) -> dict | None:
        """Palm targets in the robot pelvis frame, or None until calibrated.

        Keys: ``left`` / ``right`` – 4×4 np.ndarray.
        Translation = scaled palm position in pelvis-local frame.
        Rotation    = VR palm orientation mapped to robot frame via M_align.
        """
        if self.scale is None:  # waiting for calibration / robot poses
            return None

        head_mat = xr_flat_to_matrix(payload["head_pose"])
        left_mat = xr_flat_to_matrix(payload["left_pose"])
        right_mat = xr_flat_to_matrix(payload["right_pose"])

        head_pos_world = head_mat[:3, 3]
        left_pos_world = left_mat[:3, 3]
        right_pos_world = right_mat[:3, 3]
        q_left_world = Rotation.from_matrix(left_mat[:3, :3])
        q_right_world = Rotation.from_matrix(right_mat[:3, :3])

        q_dynamic_global_to_local = self._global_to_local(head_mat)

        q_target_robot_left = (
            q_dynamic_global_to_local * q_left_world * self._m_align_left
        )
        q_target_robot_right = (
            q_dynamic_global_to_local * q_right_world * self._m_align_right
        )

        # Relative vector between the head and the palms in quest global frame.
        human_relative_left = left_pos_world - head_pos_world
        human_relative_right = right_pos_world - head_pos_world

        # Delta between the robot reference pose and the desired position, in the
        # robot local frame: express the human palm delta in the local frame and
        # apply the isotropic scale.
        left_delta = self.scale * (
            q_dynamic_global_to_local.apply(human_relative_left)
            - self._q_calib_global_to_local.apply(self._human_reference_left)
        )
        right_delta = self.scale * (
            q_dynamic_global_to_local.apply(human_relative_right)
            - self._q_calib_global_to_local.apply(self._human_reference_right)
        )

        left_target = np.eye(4)
        left_target[:3, :3] = q_target_robot_left.as_matrix()
        left_target[:3, 3] = self._robot_reference_left + left_delta

        right_target = np.eye(4)
        right_target[:3, :3] = q_target_robot_right.as_matrix()
        right_target[:3, 3] = self._robot_reference_right + right_delta

        return {"left": left_target, "right": right_target}
