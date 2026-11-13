"""GR00T modality config for the WRIST-ONLY perception model (fine approach).

Used by gr00t/experiment/launch_finetune_perception.py to finetune the
perception-only model (VLM backbone + target heatmap head) on every labelled
frame of the drilling dataset.

Differences vs the full-VLA config (register_new_modality.py):
  - video:  ONLY the wrist camera, single CURRENT frame (delta_indices=[0]).
            The heatmap label (target_uv) is defined in this camera's pixel
            frame, and the perception processor transforms it through the same
            augmentation as this one image.
  - action: single step (delta_indices=[0]), ABSOLUTE. The perception model
            ignores actions entirely; a minimal action config is kept only
            because the data pipeline (sharding / statistics) requires one.
            Horizon 1 also means the effective episode length is the FULL
            episode, so the final close-range frames — the most valuable for
            this model — are not trimmed away by an action horizon.
  - state:  kept minimal (drill-tip pose); also ignored by the model.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


perception_unitree_g1 = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_drill_tip_eef_9d"],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_drill_tip_eef_9d"],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(perception_unitree_g1, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
