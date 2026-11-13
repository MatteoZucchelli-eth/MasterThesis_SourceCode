# MIT License
"""Unit tests for the VLA ObservationBuilder."""

import numpy as np
import pytest
from g1_control.vla.observation import assemble_flat_state, ObservationBuilder
from g1_definitions import definitions


def test_assemble_flat_state_layout():
    right_tip = np.arange(9.0)
    right_arm = np.array([3.0, 4.0])
    out = assemble_flat_state(right_tip, right_arm)
    assert out.dtype == np.float32
    assert np.allclose(out, np.concatenate([right_tip, right_arm]))


def test_flat_state_uses_pr_slices():
    rng = np.random.default_rng(0)
    joint_pos = rng.standard_normal(definitions.NUM_MOTOR_G1_BASE)
    right_tip = rng.standard_normal(9)

    out = ObservationBuilder.flat_state(joint_pos, right_tip)

    ra = definitions.JOINT_PR_SLICES["right_arm"]
    expected = np.concatenate([right_tip, joint_pos[ra[0] : ra[1]]])
    assert np.allclose(out, expected)
    assert out.shape == (definitions.STATE_DIM,)


def test_flat_state_matches_dataset_slices():
    """The deployed state must land in the same columns the recorder writes."""
    rng = np.random.default_rng(1)
    joint_pos = rng.standard_normal(definitions.NUM_MOTOR_G1_BASE)
    right_tip = rng.standard_normal(9)

    out = ObservationBuilder.flat_state(joint_pos, right_tip)
    ra = definitions.JOINT_PR_SLICES["right_arm"]

    lo, hi = definitions.STATE_SLICES[definitions.RIGHT_EEF_KEY]
    assert np.allclose(out[lo:hi], right_tip)
    lo, hi = definitions.STATE_SLICES["right_arm"]
    assert np.allclose(out[lo:hi], joint_pos[ra[0] : ra[1]])


def test_build_observation_structure():
    builder = ObservationBuilder()
    # Pretend a modality config was applied.
    builder.state_keys = [definitions.RIGHT_EEF_KEY, "right_arm"]
    builder.video_keys = ["ego_view", "wrist_view"]
    builder.language_keys = ["task"]
    builder.state_horizon = 2

    joint_pos = np.zeros(definitions.NUM_MOTOR_G1_BASE)
    image_seqs = {
        "ego_view": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "wrist_view": np.ones((2, 8, 8, 3), dtype=np.uint8),
    }
    right_tip = np.zeros(9, dtype=np.float32)

    obs = builder.build(joint_pos, image_seqs, right_tip, "do it")

    assert obs["language"]["task"] == [["do it"]]
    # Video gets a leading batch dim, and each view carries its own frames.
    assert obs["video"]["ego_view"].shape == (1, 2, 8, 8, 3)
    assert obs["video"]["wrist_view"].shape == (1, 2, 8, 8, 3)
    assert not np.array_equal(obs["video"]["ego_view"], obs["video"]["wrist_view"])
    # State stacked over the horizon then batched.
    assert obs["state"][definitions.RIGHT_EEF_KEY].shape == (1, 2, 9)
    n_right_arm = (
        definitions.JOINT_PR_SLICES["right_arm"][1]
        - definitions.JOINT_PR_SLICES["right_arm"][0]
    )
    assert obs["state"]["right_arm"].shape == (1, 2, n_right_arm)


def test_build_observation_skips_unknown_state_key(caplog):
    builder = ObservationBuilder()
    builder.state_keys = ["bogus_key"]
    builder.video_keys = []
    builder.language_keys = []
    builder.state_horizon = 1

    obs = builder.build(
        np.zeros(definitions.NUM_MOTOR_G1_BASE),
        {},
        np.zeros(9, dtype=np.float32),
        "t",
    )
    assert "bogus_key" not in obs["state"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
