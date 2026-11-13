# MIT License
"""Unit tests for RobotModel kinematics helpers (require the G1 URDF)."""

import numpy as np
import pytest

from g1_control.utils.wbc_robot_model import RobotModel


def test_se3_9d_round_trip():
    pin = pytest.importorskip("pinocchio")
    rng = np.random.default_rng(0)
    # Random valid rotation via a quaternion, random translation.
    quat = rng.standard_normal(4)
    quat /= np.linalg.norm(quat)
    rot = pin.Quaternion(*quat).matrix()
    trans = rng.standard_normal(3)
    se3 = pin.SE3(rot, trans)

    nine_d = RobotModel.se3_to_9d_pose(se3)
    assert nine_d.shape == (9,)

    back = RobotModel.action_9d_to_se3(nine_d)
    assert np.allclose(back.translation, trans, atol=1e-6)
    assert np.allclose(back.rotation, rot, atol=1e-5)
    # Reconstructed rotation is orthonormal.
    assert np.allclose(back.rotation @ back.rotation.T, np.eye(3), atol=1e-6)


def test_action_9d_to_se3_orthonormalizes_non_orthogonal_input():
    pin = pytest.importorskip("pinocchio")  # noqa: F841
    # x and y not orthogonal / not unit; result must still be a rotation.
    nine_d = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    se3 = RobotModel.action_9d_to_se3(nine_d)
    R = se3.rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "nine_d, why",
    [
        # Unfilled action-buffer slot (zeros).
        (np.zeros(9), "all-zero pose"),
        # Position predicted, rotation absent.
        (np.array([0.4, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "zero rot6d"),
        # x is fine but y carries no independent direction, so roll is undefined.
        (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]), "y parallel to x"),
        # NaNs must not sneak past the norm comparison.
        (np.full(9, np.nan), "NaN pose"),
    ],
)
def test_action_9d_to_se3_rejects_degenerate_rot6d(nine_d, why):
    pytest.importorskip("pinocchio")
    with pytest.raises(ValueError, match="degenerate rot6d"):
        RobotModel.action_9d_to_se3(nine_d)


def test_relative_group_indices(robot_model):
    abs_left = robot_model.get_joint_group_indices("left_arm")
    abs_upper = robot_model.get_joint_group_indices("upper_body")
    rel_left = robot_model.get_relative_group_indices("left_arm", "upper_body")
    # Each relative index points back to the matching absolute index.
    assert [abs_upper[r] for r in rel_left] == abs_left


def test_assemble_upper_body_target(robot_model):
    abs_left = robot_model.get_joint_group_indices("left_arm")
    abs_right = robot_model.get_joint_group_indices("right_arm")
    rel_left = robot_model.get_relative_group_indices("left_arm", "upper_body")
    rel_right = robot_model.get_relative_group_indices("right_arm", "upper_body")

    default = robot_model.get_initial_upper_body_pose()
    rng = np.random.default_rng(1)
    left_q = rng.standard_normal(robot_model.num_dofs)
    right_q = rng.standard_normal(robot_model.num_dofs)

    out = robot_model.assemble_upper_body_target(default, left_q, right_q)

    assert np.allclose(out[rel_left], left_q[abs_left])
    assert np.allclose(out[rel_right], right_q[abs_right])
    # The input default vector is not mutated.
    assert not np.shares_memory(out, default)


def test_clip_configuration_respects_limits(robot_model):
    q = robot_model.q_zero.copy() + 100.0  # way out of range
    clipped = robot_model.clip_configuration(q)
    assert np.all(clipped <= robot_model.upper_joint_limits)
    assert np.all(clipped >= robot_model.lower_joint_limits)


def test_dof_index_unknown_joint_raises(robot_model):
    with pytest.raises(ValueError):
        robot_model.dof_index("not_a_real_joint")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
