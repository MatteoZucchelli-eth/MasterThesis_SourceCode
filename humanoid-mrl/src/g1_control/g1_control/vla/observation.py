# MIT License
"""GR00T observation assembly for the VLA client."""

from __future__ import annotations

import logging

import numpy as np
from g1_definitions import definitions

from g1_control.utils import pr_slice

logger = logging.getLogger(__name__)


def assemble_flat_state(
    right_eef_9d: np.ndarray,
    right_arm: np.ndarray,
) -> np.ndarray:
    """Concatenate the components of the flat ``observation.state`` vector.

    The layout (right drill tip, right arm positions) must match the one the
    recorder writes for the training dataset so the OOD scorer compares like
    with like — see ``definitions.STATE_KEY_SIZES``.
    """
    return np.concatenate([right_eef_9d, right_arm]).astype(np.float32)


class ObservationBuilder:
    """Builds GR00T observation dicts and flat OOD state vectors.

    The modality keys and history horizons are populated from the server's
    modality config via :meth:`update_modality`; until then the builder uses
    safe defaults and :meth:`build` should not be called.
    """

    def __init__(self) -> None:
        self.state_keys: list[str] | None = None
        self.video_keys: list[str] | None = None
        self.language_keys: list[str] | None = None
        self.state_horizon = 1

    def update_modality(self, modality_config: dict) -> int:
        """Adopt server modality keys/horizons; return the video horizon."""
        self.state_keys = modality_config["state"].modality_keys
        self.video_keys = modality_config["video"].modality_keys
        self.language_keys = modality_config["language"].modality_keys
        self.state_horizon = len(modality_config["state"].delta_indices)
        return len(modality_config["video"].delta_indices)

    def build(
        self,
        joint_pos: np.ndarray,
        image_seqs: dict[str, np.ndarray],
        right_eef_9d: np.ndarray,
        task: str,
    ) -> dict:
        """Assemble a GR00T-format observation dict from current sensor data."""
        joint_pos = joint_pos.astype(np.float32)

        obs: dict = {"video": {}, "state": {}, "language": {}}

        for key in self.language_keys:
            obs["language"][key] = [[task]]

        for key in self.video_keys:
            obs["video"][key] = image_seqs[key][np.newaxis, ...]

        # Only the drilling arm is observed: tip pose and right-arm joints.
        state_map = {
            definitions.RIGHT_EEF_KEY: right_eef_9d,
            "right_arm": pr_slice(joint_pos, "right_arm"),
        }

        for key in self.state_keys:
            if key in state_map:
                state_seq = np.stack([state_map[key]] * self.state_horizon, axis=0)
                obs["state"][key] = state_seq[np.newaxis, ...]
            else:
                logger.warning("Unexpected state key requested by server: %s", key)

        return obs

    @staticmethod
    def flat_state(joint_pos: np.ndarray, right_eef_9d: np.ndarray) -> np.ndarray:
        """Assemble the flat state vector matching ``observation.state``."""
        return assemble_flat_state(right_eef_9d, pr_slice(joint_pos, "right_arm"))
