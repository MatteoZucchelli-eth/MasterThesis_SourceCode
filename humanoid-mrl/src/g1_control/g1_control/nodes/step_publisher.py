# MIT License
"""Perfect-sync ControlStep snapshots for the dataset recorder."""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from g1_definitions import definitions
from g1_interfaces.msg import ControlStep

from g1_control.states.base import ControlContext
from g1_control.utils.ros_helpers import lookup_se3


def _commanded_or_measured(
    target_9d: np.ndarray | None, measured_9d: np.ndarray
) -> np.ndarray:
    """Return ``target_9d`` if it is a usable command, else the measured pose.

    ``None``, non-finite values and a zero rot6d column (which would normalise
    to NaN) all mean "nothing commanded".
    """
    if target_9d is None:
        return measured_9d
    target_9d = np.asarray(target_9d)
    if not np.all(np.isfinite(target_9d)):
        return measured_9d
    # Degenerate rot6d: either column vanishing leaves the rotation undefined.
    if not np.any(target_9d[3:6]) or not np.any(target_9d[6:9]):
        return measured_9d
    return target_9d


class ControlStepPublisher:
    """Assembles and publishes the per-tick state+action snapshot.

    One ControlStep per control tick: the raw LowState, the measured and
    commanded end-effector poses, the base twist and the tip→target task metric,
    all sampled at the same instant so the recorder never has to re-synchronize.
    """

    def __init__(self, node, ctx: ControlContext) -> None:
        self._ctx = ctx
        self._pub = node.create_publisher(
            ControlStep, definitions.TOPIC_RECORDED_STEP, 10
        )

    def publish(self, low_state_msg) -> None:
        ctx = self._ctx
        msg = ControlStep()
        msg.low_state = low_state_msg

        msg.left_eef_pose = ctx.current_left_eef_9d.tolist()
        msg.right_eef_pose = ctx.current_right_eef_9d.tolist()

        # With no commanded target, report the measured pose.
        msg.left_eef_target = _commanded_or_measured(
            ctx.targets.left_9d, ctx.current_left_eef_9d
        ).tolist()
        msg.right_eef_target = _commanded_or_measured(
            ctx.targets.right_9d, ctx.current_right_eef_9d
        ).tolist()

        if ctx.robot_state.twist_command is not None:
            msg.vel_cmd = ctx.robot_state.twist_command.tolist()
        else:
            msg.vel_cmd = [0.0, 0.0, 0.0]

        tip_target = self._tip_target_pose()
        if tip_target is None:
            msg.tip_target_valid = False
            msg.tip_target_9d = [0.0] * 9
            msg.tip_target_dist = 0.0
        else:
            msg.tip_target_valid = True
            msg.tip_target_9d = ctx.robot_model.se3_to_9d_pose(tip_target).tolist()
            msg.tip_target_dist = float(np.linalg.norm(tip_target.translation))

        self._pub.publish(msg)

    def _tip_target_pose(self) -> pin.SE3 | None:
        """Pose of the wall target in the drill-tip frame (T_tip⁻¹·T_target), or None.

        Per-frame task label, egocentric (tip frame). The tip comes from FK at
        the measured q, re-cached here since handlers leave it at commanded q.
        """
        ctx = self._ctx
        if not ctx.has_drill_tip:
            return None
        target_se3 = lookup_se3(ctx.tf_buffer, ctx.drill_target_frame, "pelvis")
        if target_se3 is None:
            return None

        ctx.robot_model.cache_forward_kinematics(ctx.full_q, auto_clip=False)
        tip_se3 = ctx.robot_model.frame_placement(definitions.DRILL_TIP_FRAME)
        return tip_se3.inverse() * target_se3
