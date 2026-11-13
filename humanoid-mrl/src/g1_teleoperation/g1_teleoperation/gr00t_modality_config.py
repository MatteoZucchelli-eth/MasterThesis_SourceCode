"""GR00T modality config for the G1 drill-tip embodiment.

Usage (fine-tuning):
    uv run python gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path /path/to/g1_dataset \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path /path/to/this/file/gr00t_modality_config.py \\
        --num-gpus 1 \\
        --output-dir /tmp/g1_drill_tip \\
        --max-steps 5000 --save-steps 1000 \\
        --global-batch-size 32

The recorded dataset has:
  state  (16-dim): right_drill_tip_eef_9d(9) + right_arm(7)
  action  (9-dim): right_drill_tip_eef_9d(9)
  video:           observation.images.ego_view    →  "ego_view" key
                   observation.images.wrist_view  →  "wrist_view" key
  language:        annotation.human.action.task_description

Only the drilling arm is described; the left arm and waist are constant.

The right end-effector is the drill tip, not the palm; state and action share
the frame so the RELATIVE action is a true tip displacement.

Auxiliary per-frame parquet columns (not GR00T modalities):
  next.tip_target_9d:    target pose in the drill-tip frame (T_tip^-1 * T_target),
                         xyz(3) + rot6d(6). Training label for the auxiliary head.
  next.tip_target_dist:  ‖xyz‖ [m], for analysis and filtering.
  Unresolved labels are NaN, never 0.0.

After fine-tuning, deploy with chunk_size=G1_DRILL_TIP_CHUNK_SIZE (see below).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Number of future steps the model predicts. Must match --action-horizon at
# eval time and the chunk_size passed to VLAActionManager at deployment.
G1_DRILL_TIP_CHUNK_SIZE: int = 30

mrl_unitree_g1 = {
    # Head and wrist cameras, previous and current frame.
    "video": ModalityConfig(
        delta_indices=[-1, 0],
        modality_keys=["ego_view", "wrist_view"],
    ),
    # Drill-tip pose (FK) and right arm joints.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_drill_tip_eef_9d",
            "right_arm",
        ],
    ),
    # Chunk of drill-tip targets, RELATIVE to the matching state key.
    "action": ModalityConfig(
        delta_indices=list(range(G1_DRILL_TIP_CHUNK_SIZE)),
        modality_keys=[
            "right_drill_tip_eef_9d",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
        ],
    ),
    # Language: task instruction stored as task index in the parquet.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Always register custom embodiments under NEW_EMBODIMENT.
register_modality_config(mrl_unitree_g1, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
