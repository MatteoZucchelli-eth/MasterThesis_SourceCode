# MIT License
"""The policy's first stacked observation repeats the live frame, not zeros.

An all-zero frame (including a zero gravity vector) is out of distribution
at the moment the policy takes over the lower body.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from g1_control.policy.gear_wbc_policy import G1GearWbcPolicy  # noqa: E402

_N_JOINTS = 29  # 13 + 2 * 29 + 15 = 86, the single-frame observation width
_SINGLE_OBS = 86
_HISTORY_LEN = 6


def _policy():
    """A policy with its ONNX sessions left out: only the obs stacking is used."""
    policy = G1GearWbcPolicy.__new__(G1GearWbcPolicy)
    policy.config = {
        "obs_history_len": _HISTORY_LEN,
        "num_obs": _SINGLE_OBS * _HISTORY_LEN,
        "num_actions": 15,
        "default_angles": np.zeros(_N_JOINTS, dtype=np.float32),
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 1.0,
        "ang_vel_scale": 1.0,
        "cmd_scale": np.ones(3, dtype=np.float32),
    }
    policy.robot_model = SimpleNamespace(
        get_joint_group_indices=lambda group: list(range(_N_JOINTS))
    )
    import collections

    policy.obs_history = collections.deque(maxlen=_HISTORY_LEN)
    policy.obs_buffer = np.zeros(policy.config["num_obs"], dtype=np.float32)
    policy.action = np.zeros(15, dtype=np.float32)
    policy.cmd = np.zeros(3, dtype=np.float32)
    policy.height_cmd = 0.0
    policy.roll_cmd = 0.0
    policy.pitch_cmd = 0.0
    policy.yaw_cmd = 0.0
    policy.freq_cmd = 0.75
    policy.gait_indices = np.zeros(1, dtype=np.float32)
    return policy


def _observation(joint_angle: float):
    """A stationary robot, upright, with every body joint at ``joint_angle``."""
    return {
        "q": np.full(_N_JOINTS, joint_angle),
        "dq": np.zeros(_N_JOINTS),
        # Identity quaternion at [3:7], zero base velocity.
        "floating_base_pose": np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
        "floating_base_vel": np.zeros(6),
    }


def _frames(policy):
    return policy.obs_buffer.reshape(_HISTORY_LEN, _SINGLE_OBS)


def test_the_first_observation_fills_the_history_with_itself():
    """No frame of the first policy input may be a fabricated zero frame."""
    policy = _policy()
    policy.set_observation(_observation(0.3))

    frames = _frames(policy)
    live = frames[-1]
    assert np.any(live), "the live frame is all zeros; the test proves nothing"
    for i, frame in enumerate(frames):
        np.testing.assert_allclose(frame, live, err_msg=f"history frame {i}")


def test_the_warm_start_frames_are_copies():
    """Padding with the same array object would rewrite history in place."""
    policy = _policy()
    policy.set_observation(_observation(0.3))
    first_pad = policy.obs_history[0]

    policy.set_observation(_observation(-0.4))

    frames = _frames(policy)
    np.testing.assert_allclose(frames[0], first_pad)
    assert not np.allclose(frames[-1], frames[0]), "the new frame overwrote the old"


def test_a_full_history_is_left_alone():
    """Once six real frames exist the padding must not run again."""
    policy = _policy()
    for i in range(_HISTORY_LEN + 2):
        policy.set_observation(_observation(0.01 * i))

    frames = _frames(policy)
    # Six *different* observations, in order, oldest first.
    angles = [frame[13] for frame in frames]
    assert angles == sorted(angles)
    assert len(set(angles)) == _HISTORY_LEN
