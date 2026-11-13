# MIT License
"""Tests for drill-tip-relative GOAL_POSE commands."""

from types import SimpleNamespace

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
pytest.importorskip("geometry_msgs.msg")
from geometry_msgs.msg import Vector3

from g1_control.nodes.goal_pose_publisher_node import (
    _build_translation,
    _parse_args,
)
from g1_control.states.goal_pose import GoalPoseHandler, relative_tip_goal
from g1_definitions import definitions


class _Logger:
    def __init__(self):
        self.info_lines = []
        self.error_lines = []

    def info(self, message, **_kwargs):
        self.info_lines.append(str(message))

    def error(self, message, **_kwargs):
        self.error_lines.append(str(message))


class _Model:
    def __init__(self, tip_pose):
        self.tip_pose = tip_pose
        self.cached_q = None
        self.solve_call = None
        self.pose_frame = None

    def cache_forward_kinematics(self, q, auto_clip=True):
        self.cached_q = np.asarray(q).copy()

    def frame_placement(self, frame):
        assert frame == definitions.DRILL_TIP_FRAME
        return self.tip_pose.copy()

    def solve_ik(self, frame, goal, current_q):
        self.solve_call = (frame, goal.copy(), np.asarray(current_q).copy())
        return np.arange(10, dtype=float)

    def assemble_upper_body_target(self, default, current, right_arm_q):
        return np.asarray(right_arm_q).copy()

    def frame_pose_9d(self, q, frame):
        self.pose_frame = frame
        return np.arange(9, dtype=float)


def _handler(model, *, has_drill_tip=True):
    handler = object.__new__(GoalPoseHandler)
    handler._ctx = SimpleNamespace(
        has_drill_tip=has_drill_tip,
        robot_model=model,
        full_q=np.zeros(10),
    )
    handler._logger = _Logger()
    handler._goal_se3 = None
    return handler


def test_relative_tip_goal_rotates_local_xyz_into_pelvis():
    # Tip +x points along pelvis +y and tip +y points along pelvis -x.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tip = pin.SE3(rotation, np.array([0.4, -0.2, 0.7]))

    goal = relative_tip_goal(tip, [0.10, 0.02, -0.03])

    np.testing.assert_allclose(goal.translation, [0.38, -0.10, 0.67])
    np.testing.assert_allclose(goal.rotation, rotation)


@pytest.mark.parametrize("xyz", [[1.0, 2.0], [0.0, np.nan, 0.0]])
def test_relative_tip_goal_rejects_invalid_xyz(xyz):
    with pytest.raises(ValueError):
        relative_tip_goal(pin.SE3.Identity(), xyz)


def test_callback_snapshots_current_tip_pose_once():
    model = _Model(pin.SE3(np.eye(3), np.array([0.5, -0.1, 0.2])))
    handler = _handler(model)

    handler._goal_callback(Vector3(x=0.04, y=-0.02, z=0.01))
    snapshotted_goal = handler._goal_se3.copy()

    model.tip_pose = pin.SE3(np.eye(3), np.array([0.9, 0.8, 0.7]))

    np.testing.assert_allclose(snapshotted_goal.translation, [0.54, -0.12, 0.21])
    np.testing.assert_allclose(handler._goal_se3.translation, [0.54, -0.12, 0.21])
    np.testing.assert_allclose(model.cached_q, handler._ctx.full_q)


def test_callback_rejects_command_without_drill_tip_frame():
    model = _Model(pin.SE3.Identity())
    handler = _handler(model, has_drill_tip=False)

    handler._goal_callback(Vector3(x=0.01, y=0.0, z=0.0))

    assert handler._goal_se3 is None
    assert "not in the URDF" in handler._logger.error_lines[-1]


def test_update_solves_and_records_in_the_drill_tip_frame():
    model = _Model(pin.SE3.Identity())
    handler = _handler(model)
    goal = pin.SE3(np.eye(3), np.array([0.1, 0.2, 0.3]))
    handler._goal_se3 = goal
    handler._ctx.robot_state = SimpleNamespace(
        twist_command=None,
        upper_body_target=None,
        upper_body_target_chunk=np.ones((1, 1)),
    )
    handler._ctx.default_upper_body_pose = np.zeros(10)
    handler._ctx.targets = SimpleNamespace(right_9d=None)
    handler._log_residual = lambda *_args: None
    handler._log_joint_error = lambda *_args: None
    handler._publish_markers = lambda *_args: None

    handler.update()

    assert model.solve_call[0] == definitions.DRILL_TIP_FRAME
    np.testing.assert_allclose(model.solve_call[1].translation, goal.translation)
    assert model.pose_frame == definitions.DRILL_TIP_FRAME
    np.testing.assert_allclose(handler._ctx.robot_state.twist_command, np.zeros(3))
    assert handler._ctx.robot_state.upper_body_target_chunk is None
    np.testing.assert_allclose(handler._ctx.targets.right_9d, np.arange(9))


def test_publisher_builds_vector3_with_zero_defaults():
    defaults = _parse_args([])
    assert (defaults.x, defaults.y, defaults.z) == (0.0, 0.0, 0.0)

    args = _parse_args(["--x", "0.05", "--y", "-0.02", "--z", "0.01"])
    message = _build_translation(args)
    assert (message.x, message.y, message.z) == pytest.approx((0.05, -0.02, 0.01))
