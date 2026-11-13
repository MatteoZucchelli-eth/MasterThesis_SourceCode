# MIT License
"""Per-tick IK commanding of the drill tip.

* IK is warm-started from the last *commanded* configuration, so measured
  droop and joint noise don't feed back into the solution.
* Left arm and waist are held at their default pose.
* The recorded action is the FK pose of the commanded configuration.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from g1_control.states.base import ControlContext
from g1_control.utils.wbc_robot_model import RobotModel

# Per-tick IK budget: warm-started solves converge in a few iterations; the cap
# bounds tick time on unreachable targets.
_IK_MAX_ITERS = 20


class TipPoseCommander:
    """Caches the commanded upper-body target and the eef pose it implies."""

    def __init__(
        self, robot_model: RobotModel, ctx: ControlContext, tip_frame: str
    ) -> None:
        self._model = robot_model
        self._ctx = ctx
        self._tip_frame = tip_frame
        self.reset()

    def reset(self) -> None:
        self._commanded_upper_body: np.ndarray | None = None
        self._commanded_eef_9d: np.ndarray | None = None

    @property
    def commanded_upper_body(self) -> np.ndarray | None:
        """Upper-body joint vector currently commanded (None before the first)."""
        return self._commanded_upper_body

    @property
    def commanded_eef_9d(self) -> np.ndarray | None:
        """FK eef pose (9D) of the commanded configuration, for the recorder."""
        return self._commanded_eef_9d

    def command_tip_pose(self, tip_se3: pin.SE3) -> None:
        """Solve right-arm IK for a tip pose and cache the joint target."""
        arm_q = self._model.solve_ik(
            self._tip_frame,
            tip_se3,
            self._seed_q(),
            max_iters=_IK_MAX_ITERS,
            # No neutral-pose retry: it can jump IK basins between ticks.
            retry_from_neutral=False,
        )
        self.command_upper_body(
            self._model.assemble_upper_body_target(
                self._ctx.default_upper_body_pose,
                self._model.default_body_pose,
                arm_q,
            )
        )

    def command_upper_body(self, upper_body_target: np.ndarray) -> None:
        """Cache an upper-body joint target and its corresponding eef pose."""
        self._commanded_upper_body = np.asarray(upper_body_target, dtype=np.float64)
        q = self._ctx.full_q.copy()
        q[self._ctx.upper_body_indices] = self._commanded_upper_body
        self._commanded_eef_9d = self._model.frame_pose_9d(q, self._ctx.right_eef_frame)

    def _seed_q(self) -> np.ndarray:
        """Full q to warm-start IK from: the last command, else the measured pose."""
        q = self._ctx.full_q.copy()
        if self._commanded_upper_body is not None:
            q[self._ctx.upper_body_indices] = self._commanded_upper_body
        return q
