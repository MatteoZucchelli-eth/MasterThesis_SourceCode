import logging
from typing import List, Optional, Set, Union

import numpy as np
import pinocchio as pin

from .robot_supplemental_info import RobotSupplementalInfo

logger = logging.getLogger(__name__)


class RobotModel:
    def __init__(
        self,
        urdf_path: str,
        asset_path: str,
        set_floating_base: bool = False,
        supplemental_info: Optional[RobotSupplementalInfo] = None,
    ):
        self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[asset_path],
            root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
        )
        # Recorded so consumers can identify the model variant.
        self.urdf_path = urdf_path
        self.is_floating_base_model = set_floating_base

        self.joint_to_dof_index = {}
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if set_floating_base
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q

        root_nq = 7 if set_floating_base else 0
        self.upper_joint_limits = self.pinocchio_wrapper.model.upperPositionLimit[
            root_nq:
        ].copy()
        self.lower_joint_limits = self.pinocchio_wrapper.model.lowerPositionLimit[
            root_nq:
        ].copy()

        self.supplemental_info = supplemental_info
        if self.supplemental_info is not None:
            self._body_actuated_joint_indices = [
                self.dof_index(name)
                for name in self.supplemental_info.body_actuated_joints
            ]
            self._left_hand_actuated_joint_indices = [
                self.dof_index(name)
                for name in self.supplemental_info.left_hand_actuated_joints
            ]
            self._right_hand_actuated_joint_indices = [
                self.dof_index(name)
                for name in self.supplemental_info.right_hand_actuated_joints
            ]
            self._hand_actuated_joint_indices = (
                self._left_hand_actuated_joint_indices
                + self._right_hand_actuated_joint_indices
            )

            self._joint_group_indices = {}
            for group_name, group_info in self.supplemental_info.joint_groups.items():
                indices = []
                indices.extend([self.dof_index(name) for name in group_info["joints"]])
                for subgroup_name in group_info["groups"]:
                    indices.extend(self.get_joint_group_indices(subgroup_name))
                self._joint_group_indices[group_name] = sorted(set(indices))

            # Absolute (full-q) and relative (within upper_body) arm indices,
            # cached for assemble_upper_body_target.
            self._abs_left_arm = self.get_joint_group_indices("left_arm")
            self._abs_right_arm = self.get_joint_group_indices("right_arm")
            self._rel_left_arm = self.get_relative_group_indices(
                "left_arm", "upper_body"
            )
            self._rel_right_arm = self.get_relative_group_indices(
                "right_arm", "upper_body"
            )

            if (
                hasattr(self.supplemental_info, "joint_limits")
                and self.supplemental_info.joint_limits
            ):
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

        self.default_body_pose = self.q_zero.copy()

        if self.supplemental_info is not None:
            default_joint_q = self.supplemental_info.default_joint_q
            for joint, joint_values in default_joint_q.items():
                joint_mapping = self.supplemental_info.joint_name_mapping[joint]
                if isinstance(joint_mapping, str):
                    if joint_mapping in self.joint_to_dof_index:
                        joint_idx = self.dof_index(joint_mapping)
                        self.default_body_pose[joint_idx] = joint_values
                else:
                    for side, value in joint_values.items():
                        if (
                            side in joint_mapping
                            and joint_mapping[side] in self.joint_to_dof_index
                        ):
                            joint_idx = self.dof_index(joint_mapping[side])
                            self.default_body_pose[joint_idx] = value

        self.initial_body_pose = self.default_body_pose.copy()

        # Build T-pose from q_zero, applying only t_pose_joint_q overrides.
        self.t_body_pose = self.q_zero.copy()
        if self.supplemental_info is not None and self.supplemental_info.t_pose_joint_q:
            for joint, joint_values in self.supplemental_info.t_pose_joint_q.items():
                joint_mapping = self.supplemental_info.joint_name_mapping.get(joint)
                if joint_mapping is None:
                    continue
                if isinstance(joint_mapping, str):
                    if joint_mapping in self.joint_to_dof_index:
                        self.t_body_pose[self.dof_index(joint_mapping)] = joint_values
                else:
                    for side, value in joint_values.items():
                        if (
                            side in joint_mapping
                            and joint_mapping[side] in self.joint_to_dof_index
                        ):
                            self.t_body_pose[self.dof_index(joint_mapping[side])] = (
                                value
                            )

        logger.debug(
            "T-pose upper body: %s",
            self.t_body_pose[self.get_joint_group_indices("upper_body")],
        )

    @property
    def num_dofs(self) -> int:
        return self.pinocchio_wrapper.model.nq

    @property
    def q_zero(self) -> np.ndarray:
        return self.pinocchio_wrapper.q0

    @property
    def joint_names(self) -> List[str]:
        return list(self.joint_to_dof_index.keys())

    @property
    def num_joints(self) -> int:
        return len(self.joint_to_dof_index)

    def dof_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_to_dof_index:
            raise ValueError(
                f"Unknown joint name: '{joint_name}'. "
                f"Available joints: {list(self.joint_to_dof_index.keys())}"
            )
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_joint_group_indices(self, group_names: Union[str, Set[str]]) -> List[int]:
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        if isinstance(group_names, str):
            group_names = {group_names}
        all_indices = set()
        for group_name in group_names:
            if group_name not in self._joint_group_indices:
                raise ValueError(f"Unknown joint group: {group_name}")
            all_indices.update(self._joint_group_indices[group_name])
        return sorted(all_indices)

    def get_relative_group_indices(self, group: str, within: str) -> List[int]:
        """Return the positions of ``group`` joints inside the ``within`` index list.

        Both lists are produced by :meth:`get_joint_group_indices` and are sorted,
        so the result is the rank of each group index inside the within-list.

        Example::

            rel = model.get_relative_group_indices("left_arm", "upper_body")
            upper_body_target[rel] = left_arm_values
        """
        abs_group = self.get_joint_group_indices(group)
        abs_within = self.get_joint_group_indices(within)
        within_map = {idx: pos for pos, idx in enumerate(abs_within)}
        return [within_map[idx] for idx in abs_group]

    def assemble_upper_body_target(
        self,
        default_upper_body_pose: np.ndarray,
        left_arm_q: np.ndarray,
        right_arm_q: np.ndarray,
    ) -> np.ndarray:
        """Map per-arm IK solutions into an upper-body target vector.

        ``left_arm_q`` / ``right_arm_q`` are full configuration vectors (e.g. the
        output of :meth:`solve_ik`). Their arm joints are written into a copy of
        ``default_upper_body_pose`` at the positions the arms occupy inside the
        ``upper_body`` group, leaving the remaining (e.g. waist, hand) entries at
        their default values.
        """
        target = np.asarray(default_upper_body_pose).copy()
        target[self._rel_left_arm] = left_arm_q[self._abs_left_arm]
        target[self._rel_right_arm] = right_arm_q[self._abs_right_arm]
        return target

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip: bool = True) -> None:
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]}")
        if auto_clip:
            q = self.clip_configuration(q)
        pin.framesForwardKinematics(
            self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q
        )

    def clip_configuration(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        q_clipped = q.copy()
        root_nq = 7 if self.is_floating_base_model else 0
        q_clipped[root_nq:] = np.clip(
            q[root_nq:],
            self.lower_joint_limits + margin,
            self.upper_joint_limits - margin,
        )
        return q_clipped

    def has_frame(self, frame_name: str) -> bool:
        """Whether ``frame_name`` exists in the model (e.g. drill_tip_link)."""
        return any(f.name == frame_name for f in self.pinocchio_wrapper.model.frames)

    def frame_placement(self, frame_name: str) -> pin.SE3:
        model = self.pinocchio_wrapper.model
        data = self.pinocchio_wrapper.data
        frame_id = model.getFrameId(frame_name)
        if frame_id < 0 or frame_id >= len(model.frames):
            valid_frames = [f.name for f in model.frames]
            raise ValueError(
                f"Unknown frame '{frame_name}'. Valid frames: {valid_frames}"
            )
        return data.oMf[frame_id].copy()

    def frame_pose_9d(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        """FK pose of ``frame_name`` at configuration ``q`` as 9D (xyz + rot6d).

        Re-caches FK at ``q`` unclipped, so it reflects exactly the configuration
        it was asked about (measured or commanded).
        """
        self.cache_forward_kinematics(q, auto_clip=False)
        return self.se3_to_9d_pose(self.frame_placement(frame_name))

    def get_body_actuated_joints(self, q: np.ndarray) -> np.ndarray:
        return q[self.get_body_actuated_joint_indices()]

    def get_hand_actuated_joints(self, q: np.ndarray, side: str = "both") -> np.ndarray:
        return q[self.get_hand_actuated_joint_indices(side)]

    def get_configuration_from_actuated_joints(
        self,
        body_actuated_joint_values: np.ndarray,
        hand_actuated_joint_values: Optional[np.ndarray] = None,
        left_hand_actuated_joint_values: Optional[np.ndarray] = None,
        right_hand_actuated_joint_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q = self.pinocchio_wrapper.q0.copy()
        q[self.get_body_actuated_joint_indices()] = body_actuated_joint_values
        if hand_actuated_joint_values is not None:
            q[self.get_hand_actuated_joint_indices("both")] = hand_actuated_joint_values
        else:
            if left_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("left")] = (
                    left_hand_actuated_joint_values
                )
            if right_hand_actuated_joint_values is not None:
                q[self.get_hand_actuated_joint_indices("right")] = (
                    right_hand_actuated_joint_values
                )
        return q

    def get_t_upper_body_pose(self) -> np.ndarray:
        return self.t_body_pose[self.get_joint_group_indices("upper_body")]

    def get_initial_upper_body_pose(self) -> np.ndarray:
        return self.initial_body_pose[self.get_joint_group_indices("upper_body")]

    def get_default_body_pose(self) -> np.ndarray:
        return self.default_body_pose

    def set_initial_body_pose(self, q: np.ndarray, q_idx=None) -> None:
        if q_idx is None:
            self.initial_body_pose = q
        else:
            self.initial_body_pose[q_idx] = q

    @staticmethod
    def se3_to_9d_pose(se3_pose) -> np.ndarray:
        xyz = se3_pose.translation

        rot_matrix = se3_pose.rotation

        # rot6d: first two rotation columns.
        rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]])

        return np.concatenate([xyz, rot6d]).astype(np.float32)

    @staticmethod
    def action_9d_to_se3(action_9d: np.ndarray) -> pin.SE3:
        """Converts the model's absolute 9D action into a Pinocchio SE3 target.

        Raises ``ValueError`` on a degenerate rot6d (e.g. an unfilled all-zero
        action-buffer slot) instead of producing a NaN pose.
        """
        target_translation = action_9d[:3]

        rot6d = action_9d[3:]
        x_raw = rot6d[0:3]
        y_raw = rot6d[3:6]

        # `not norm > eps` also rejects a NaN norm.
        x_norm = float(np.linalg.norm(x_raw))
        if not x_norm > 1e-9:
            raise ValueError(
                "degenerate rot6d in 9D action: the first column has norm "
                f"{x_norm:.3e}; action_9d={np.asarray(action_9d).tolist()}"
            )
        x = x_raw / x_norm

        # Gram-Schmidt; a y parallel to x leaves the roll undefined.
        y_ortho = y_raw - np.dot(x, y_raw) * x
        y_norm = float(np.linalg.norm(y_ortho))
        if not y_norm > 1e-9:
            raise ValueError(
                "degenerate rot6d in 9D action: the second column is parallel to "
                f"the first (residual norm {y_norm:.3e}); "
                f"action_9d={np.asarray(action_9d).tolist()}"
            )
        y = y_ortho / y_norm

        z = np.cross(x, y)

        target_rotation = np.column_stack((x, y, z))

        return pin.SE3(target_rotation, target_translation)

    def get_palm_poses(self, current_measured_q: np.ndarray):
        """Return the SE3 poses of the palm frames (used for teleoperation tracking)."""
        self.cache_forward_kinematics(current_measured_q, auto_clip=False)
        left_palm_se3 = self.frame_placement("left_hand_palm_joint")
        right_palm_se3 = self.frame_placement("right_hand_palm_joint")
        return left_palm_se3, right_palm_se3

    def get_eef_poses(self, q: np.ndarray, left_frame: str, right_frame: str):
        """Return the SE3 poses of the two recorded end-effector frames.

        Unlike :meth:`get_palm_poses`, the right end-effector is the drill tip.
        Teleoperation tracks the palms; the dataset records these.
        """
        self.cache_forward_kinematics(q, auto_clip=False)
        return self.frame_placement(left_frame), self.frame_placement(right_frame)

    def solve_ik(
        self,
        frame_name: str,
        target_se3: pin.SE3,
        current_q: np.ndarray,
        max_iters=100,
        alpha=1.0,  # Step size multiplier (1.0 is standard Newton-Raphson)
        eps_lin=1e-4,  # Linear tolerance (meters)
        eps_ang=1e-4,  # Angular tolerance (radians)
        max_step=0.2,  # Max per-iteration joint step (rad), prevents divergence
        retry_from_neutral=True,  # Second solve from the rest pose if the first misses
    ) -> np.ndarray:

        frame_id = self.pinocchio_wrapper.model.getFrameId(frame_name)
        q = current_q.copy()

        side = "left" if "left" in frame_name else "right"

        # Only this side's arm moves; the waist belongs to the locomotion policy.
        # Fixed-base, all-revolute model: q-index == v-index.
        arm_dofs = self.get_joint_group_indices(f"{side}_arm")
        solve_dofs = list(arm_dofs)

        pin.framesForwardKinematics(
            self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q
        )
        # Local copy: teleop records the caller's raw target after this returns.
        target_se3 = pin.SE3(
            target_se3.rotation.copy(),
            target_se3.translation.copy(),
        )

        # LM damping; 1e-3 keeps arm-only-unreachable targets stable.
        damp = 1e-3
        I_6 = np.eye(6)
        model = self.pinocchio_wrapper.model
        data = self.pinocchio_wrapper.data
        # Same (supplemental) limits as clip_configuration, so the final clip does
        # not move the converged pose.
        margin = 1e-6
        lo = self.lower_joint_limits + margin
        hi = self.upper_joint_limits - margin

        def _iterate(q_seed: np.ndarray):
            """Damped-LS IK over ``solve_dofs``, keeping every iterate within joint
            limits. Returns (q, err_lin, err_ang)."""
            q = q_seed.copy()
            q[solve_dofs] = np.clip(q[solve_dofs], lo[solve_dofs], hi[solve_dofs])
            err_lin = err_ang = np.inf
            for _ in range(max_iters):
                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                oMcur = data.oMf[frame_id]

                err_local = pin.log6(oMcur.inverse() * target_se3).vector
                error = np.concatenate(
                    [oMcur.rotation @ err_local[:3], oMcur.rotation @ err_local[3:]]
                )
                if np.isnan(error).any():
                    logger.warning(
                        "Poisoned matrix detected in solve_ik for %s.", frame_name
                    )
                    return q, np.inf, np.inf

                err_lin = np.linalg.norm(error[:3])
                err_ang = np.linalg.norm(error[3:])
                if err_lin < eps_lin and err_ang < eps_ang:
                    break

                J_dofs = pin.getFrameJacobian(
                    model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )[:, solve_dofs]
                x = np.linalg.solve(J_dofs @ J_dofs.T + damp * I_6, error)
                delta = J_dofs.T @ x

                step_norm = np.linalg.norm(delta)
                if step_norm > max_step:
                    delta *= max_step / step_norm

                delta_q = np.zeros(model.nv)
                delta_q[solve_dofs] = delta
                q = pin.integrate(model, q, delta_q * alpha)
            # Project into the feasible box every step.
                q[solve_dofs] = np.clip(q[solve_dofs], lo[solve_dofs], hi[solve_dofs])
            return q, err_lin, err_ang

        q_best, lin_best, _ = _iterate(q)

        # Retry once from the neutral rest pose if the warm start ends far from the
        # target (limit-pinned basin). Disable for per-tick solvers: the retry can
        # jump between basins on consecutive ticks and make the command buzz.
        if retry_from_neutral and lin_best > eps_lin:
            neutral = current_q.copy()
            rel_arm = self.get_relative_group_indices(f"{side}_arm", "upper_body")
            neutral[arm_dofs] = self.get_initial_upper_body_pose()[rel_arm]
            q_alt, lin_alt, _ = _iterate(neutral)
            if lin_alt < lin_best:
                q_best, lin_best = q_alt, lin_alt

        return self.clip_configuration(q_best)
