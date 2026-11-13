# MIT License
"""Post-processing of VLA action chunks: IK, smoothing and OOD correction."""

from __future__ import annotations

import logging

import numpy as np
import pinocchio as pin

from g1_definitions import definitions

from g1_control.utils import resolve_eef_frames
from g1_control.utils.obs_ood_scorer import ObsOODScorer
from g1_control.utils.wbc_robot_model import RobotModel

from .observation import assemble_flat_state

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """Turns raw VLA drill-tip chunks into feasible upper-body joint chunks.

    Post-processing stages run on every freshly inferred chunk:

    * :meth:`process_ik` - batched IK from predicted eef poses to joint targets,
    * :meth:`correct_ood` - per-step energy correction of out-of-distribution
      waypoints (optional, only when an OOD scorer is supplied),
    * :meth:`smooth` - zero-lag moving-average de-jitter along the chunk.

    Each method takes the latest measured ``joint_pos`` (PR order).
    """

    def __init__(
        self,
        robot_model: RobotModel,
        default_upper_body_pose: np.ndarray,
        ood_scorer: ObsOODScorer | None = None,
        smooth_radius: int = 2,
    ) -> None:
        self.robot_model = robot_model
        self.default_upper_body_pose = default_upper_body_pose
        self._ood_scorer = ood_scorer

        # Right end-effector is the drill tip; the left palm is unused.
        _, self.right_eef_frame = resolve_eef_frames(robot_model)

        # Joint-index caches for the upper body / right arm.
        self.abs_right_arm = robot_model.get_joint_group_indices("right_arm")
        self.abs_upper_body = robot_model.get_joint_group_indices("upper_body")
        self.rel_right_arm = robot_model.get_relative_group_indices(
            "right_arm", "upper_body"
        )
        self.actuated_indices = robot_model.get_body_actuated_joint_indices()

        self._safe_threshold = 200  # need to be tuned (energy is whitened)
        self._max_correction_iters = 30

        # Per-joint correction rate over the upper body; only the right arm has
        # non-zero J_state columns. (Needs tuning.)
        n_upper = len(self.abs_upper_body)
        self._correction_rate = np.full(n_upper, 0.001)

        # Half-width of the zero-lag moving average along the chunk (window =
        # 2*radius+1); 0 disables. Smooths within a chunk only; between-chunk
        # variation is handled by `chunk_within_deadband`.
        self._smooth_radius = int(smooth_radius)

    # ── IK ──────────────────────────────────────────────────────────────────

    def process_ik(self, action_dict: dict, joint_pos: np.ndarray | None) -> dict:
        """Run batched IK on the predicted drill-tip poses into upper-body targets.

        The policy predicts the right drill tip only, so the left arm is held at
        its default pose for every step of the chunk.
        """
        right_9d = action_dict.get(definitions.RIGHT_EEF_KEY)
        if right_9d is None:
            return action_dict

        if right_9d.ndim > 2:
            right_9d = right_9d.squeeze(0)

        chunk_len = right_9d.shape[0]
        upper_body_chunk = []

        # Full config holding the default upper-body pose; its left-arm slice
        # keeps the (unpredicted) left arm put in assemble_upper_body_target.
        q_default = self.robot_model.q_zero.copy()
        q_default[self.abs_upper_body] = self.default_upper_body_pose

        q_seed_right = self.robot_model.q_zero.copy()
        if joint_pos is not None:
            q_seed_right[self.actuated_indices] = joint_pos

        for i in range(chunk_len):
            target_right = self.robot_model.action_9d_to_se3(right_9d[i])

            # Solve IK using the previous step as the starting guess.
            right_arm_q = self.robot_model.solve_ik(
                self.right_eef_frame, target_right, q_seed_right
            )
            q_seed_right = right_arm_q.copy()

            upper_body_chunk.append(
                self.robot_model.assemble_upper_body_target(
                    self.default_upper_body_pose, q_default, right_arm_q
                )
            )

        action_dict["upper_body_joint_chunk"] = np.array(upper_body_chunk)
        return action_dict

    # ── OOD correction ──────────────────────────────────────────────────────

    def correct_ood(
        self, action_dict: dict, joint_pos: np.ndarray
    ) -> tuple[dict, np.ndarray | None]:
        """Per-step energy correction of the upper-body action chunk.

        A step whose energy exceeds ``_safe_threshold`` is iterated as

            a <- clip( a - rate * J_state(a)^T @ dE/ds(s(a)) )

        re-scoring each iteration until in distribution or
        ``_max_correction_iters``; clipping is to joint limits.

        Returns the (possibly mutated) action dict and the uncorrected drill-tip
        trajectory for RViz, or ``None`` when no step was corrected.
        """
        upper_body_chunk = action_dict.get("upper_body_joint_chunk")
        if upper_body_chunk is None:
            return action_dict, None

        # Uncorrected tip trajectory, captured before any edits, for RViz.
        prev_right_9d = self.eef_9d_chunk(joint_pos, upper_body_chunk)

        corrected_chunk = upper_body_chunk.copy()
        num_corrected = 0
        for i, step in enumerate(upper_body_chunk):
            corrected = step
            iters = 0
            for _ in range(self._max_correction_iters):
                state = self._predict_state_for_step(corrected, joint_pos)
                energy, grad_s = self._ood_scorer.energy_and_grad(state)
                if energy <= self._safe_threshold:
                    break

                J_state = self._state_jacobian_for_step(corrected, joint_pos)
                delta = self._correction_rate * (J_state.T @ grad_s)

                trial = self._clip_upper_body(corrected - delta, joint_pos)

                if np.max(np.abs(trial - corrected)) < 1e-9:
                    break  # converged / pinned at a joint limit
                corrected = trial
                iters += 1

            if iters == 0:
                continue  # step was already in-distribution

            corrected_chunk[i] = corrected
            num_corrected += 1

        if num_corrected == 0:
            # Nothing was OOD; None clears the RViz "previous" trajectory.
            return action_dict, None

        logger.info(
            "[OOD] corrected %d/%d chunk steps (energy > %s)",
            num_corrected,
            len(upper_body_chunk),
            self._safe_threshold,
        )
        action_dict["upper_body_joint_chunk"] = corrected_chunk

        # Recompute the eef chunk via FK so recorder / TF / RViz match the joints.
        action_dict[definitions.RIGHT_EEF_KEY] = self.eef_9d_chunk(
            joint_pos, corrected_chunk
        )

        return action_dict, prev_right_9d

    # ── smoothing ───────────────────────────────────────────────────────────

    def smooth(self, action_dict: dict, joint_pos: np.ndarray) -> dict:
        """Low-pass the upper-body joint chunk along time to de-jitter waypoints.

        Symmetric moving average (zero lag); the tip-pose chunk is recomputed
        via FK to stay consistent.
        """
        chunk = action_dict.get("upper_body_joint_chunk")
        if chunk is None or self._smooth_radius < 1 or len(chunk) <= 1:
            return action_dict

        radius = min(self._smooth_radius, (len(chunk) - 1) // 2)
        if radius < 1:
            return action_dict

        window = 2 * radius + 1
        kernel = np.ones(window) / window
        padded = np.pad(chunk, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.stack(
            [
                np.convolve(padded[:, j], kernel, mode="valid")
                for j in range(chunk.shape[1])
            ],
            axis=1,
        )

        action_dict["upper_body_joint_chunk"] = smoothed
        action_dict[definitions.RIGHT_EEF_KEY] = self.eef_9d_chunk(joint_pos, smoothed)
        return action_dict

    # ── kinematics helpers ──────────────────────────────────────────────────

    def eef_9d_chunk(
        self, joint_pos: np.ndarray, upper_body_chunk: np.ndarray
    ) -> np.ndarray:
        """FK the drill-tip 9D pose for every chunk step."""
        q = self.robot_model.q_zero.copy()
        q[self.actuated_indices] = joint_pos

        right_list = []
        for step in upper_body_chunk:
            q[self.abs_upper_body] = step
            self.robot_model.cache_forward_kinematics(q)
            right_list.append(
                RobotModel.se3_to_9d_pose(
                    self.robot_model.frame_placement(self.right_eef_frame)
                )
            )
        return np.array(right_list)

    def _predict_state_for_step(
        self, step: np.ndarray, joint_pos: np.ndarray
    ) -> np.ndarray:
        """FK a single upper-body action ``step`` into a flat obs state.

        Layout matches ``ObservationBuilder.flat_state``.
        """
        q = self.robot_model.q_zero.copy()
        q[self.actuated_indices] = joint_pos
        q[self.abs_upper_body] = step
        self.robot_model.cache_forward_kinematics(q)

        right_9d = RobotModel.se3_to_9d_pose(
            self.robot_model.frame_placement(self.right_eef_frame)
        )

        return assemble_flat_state(right_9d, step[self.rel_right_arm])

    def _state_jacobian_for_step(
        self, step: np.ndarray, joint_pos: np.ndarray
    ) -> np.ndarray:
        """Jacobian ds/da of the flat obs state wrt the upper-body action ``step``."""
        q = self.robot_model.q_zero.copy()
        q[self.actuated_indices] = joint_pos
        q[self.abs_upper_body] = step
        q = self.robot_model.clip_configuration(q)

        model = self.robot_model.pinocchio_wrapper.model
        data = self.robot_model.pinocchio_wrapper.data

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        right_id = model.getFrameId(self.right_eef_frame)

        J_right_6 = pin.getFrameJacobian(
            model, data, right_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[
            :, self.abs_upper_body
        ]  # (6, n_upper)

        def _jac_6d_to_9d(J6, R):
            J_lin = J6[:3]  # (3, n_upper)
            J_ang = J6[3:]  # (3, n_upper)
            J_x0 = np.cross(J_ang.T, R[:, 0]).T  # (3, n_upper)
            J_x1 = np.cross(J_ang.T, R[:, 1]).T  # (3, n_upper)
            return np.vstack([J_lin, J_x0, J_x1])  # (9, n_upper)

        J_9d_right = _jac_6d_to_9d(J_right_6, data.oMf[right_id].rotation)

        # Joint part: the right arm joints are a direct linear slice of upper_body
        n_upper = len(self.abs_upper_body)
        J_joints_right = np.eye(n_upper)[self.rel_right_arm]  # (n_right_arm, n_upper)

        # Full state Jacobian ds/da  (state_dim, n_upper)
        return np.vstack([J_9d_right, J_joints_right])

    def _clip_upper_body(self, step: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """Clip an upper-body action vector to the robot's joint limits."""
        q = self.robot_model.q_zero.copy()
        q[self.actuated_indices] = joint_pos
        q[self.abs_upper_body] = step
        q = self.robot_model.clip_configuration(q)
        return q[self.abs_upper_body]
