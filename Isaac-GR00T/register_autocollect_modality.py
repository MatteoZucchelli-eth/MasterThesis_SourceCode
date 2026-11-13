"""GR00T modality config for g1_dataset_autocollect_v2 (RIGHT-ARM-ONLY embodiment).

Why this file exists instead of reusing register_new_modality.py
----------------------------------------------------------------
The autocollect_v2 capture dropped every left-arm and waist channel. Its
meta/modality.json declares a 16-dim state as

    right_drill_tip_eef_9d   0:9    (xyz + rot6d, FK from Pinocchio)
    right_arm                9:16   (7 joint positions)

and a 9-dim action (right_drill_tip_eef_9d only). register_new_modality.py asks
for `left_arm`, which no longer exists. LeRobotEpisodeLoader._extract_joint_groups
only PRINTS a warning for a missing group and returns without that column — it does
not raise — so reusing the old file would silently train on a state vector that is
7 dims narrower than the config claims. That failure mode is quiet and expensive,
which is why the two datasets get two files.

Everything else is deliberately identical to register_new_modality.py (both
cameras at [-1, 0], 30-step RELATIVE drill-tip EEF chunk, task-description
language), so a run on this capture stays comparable to the g1_dataset_red_2k /
g1_dataset_negatives runs modulo the state width.

Usage:
    torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path ./g1_dataset_autocollect_v2/ \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path register_autocollect_modality.py \\
        ...

    (or:  MODALITY_CONFIG=register_autocollect_modality.py \\
          DATASET_HOST_DIR=/cluster/work/mrl/$USER/datasets/g1_dataset_autocollect_v2 \\
          RUN_NAME=autocollect_vanilla_e1 sbatch submit_euler.sh)

Only ONE modality config may hold NEW_EMBODIMENT in a process, so never import
this alongside register_new_modality.py, and make
sure eval loads the SAME file the checkpoint was trained with.
"""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Number of future steps the model predicts. Must match --action-horizon at eval
# time and the chunk_size passed to VLAActionManager at deployment. Kept at 30 to
# match every previous run on this project.
G1_AUTOCOLLECT_CHUNK_SIZE: int = 30

# Which frames each camera contributes. DEFAULT IS SINGLE-FRAME [0], changed from
# the [-1, 0] every earlier run used. Set GR00T_VIDEO_DELTA="-1,0" to reproduce the
# old behaviour for a step-matched comparison against the red_2k / negatives runs.
#
# Measured on this dataset (scratchpad/ablate_temporal_frame.py, 266,686 fine-phase
# frame pairs over 102 episodes), the t-1 frame buys nothing:
#
#   lag  perp baseline   disparity   sigma_Z/Z @ 0.5px matching
#     1       0.074 mm     0.337 px            148%      <- [-1, 0]
#    15       1.155 mm     5.291 px              9%
#    30       2.245 mm    10.287 px              5%
#
# At 30 fps the camera moves 0.074 mm perpendicular to the line of sight between
# consecutive frames and rotates 0.005 deg. That is 0.34 px of disparity — below any
# feature-matching precision — so there is NO recoverable parallax, and the ego-motion
# the frames do encode is already in observation.state. Confirmed independently in
# pixels (ablate_temporal_pixels.py): consecutive wrist frames differ by 4.6% of the
# "unrelated image" scale, roughly half of which is mpeg4 noise
# (ablate_temporal_noisefloor.py measures a 0.54 RMS floor on motionless pairs).
#
# It also agrees with the model-level ablation already run on g1_dataset_negatives
# (scratchpad/ablate_vla_views.py): replacing wrist t-1 with a copy of wrist t moved
# tip-target error 2.93 -> 3.06 mm (noise), while swapping the EGO view doubled it
# (2.93 -> 5.81 mm). The second VIEW carries the range signal; the second FRAME does not.
#
# And with upstream: 5 of GR00T's 6 built-in embodiment configs use video=[0],
# including both Unitree G1 ones. The sole exception, oxe_droid_*, uses [-15, 0] on a
# 15 Hz dataset — a ONE SECOND lag, which at 30 fps would be [-30, 0], not [-1, 0].
#
# Dropping the frame halves the visual tokens (256 -> 128 at image_target_size 256),
# which is budget better spent on resolution: the drill target is ~25 native px, i.e.
# 0.30 of a single token at 256 px.
#
# NOTE the `or "0"`, not os.environ.get(..., "0"): submit_euler.sh forwards this as
# `--env GR00T_VIDEO_DELTA="${GR00T_VIDEO_DELTA:-}"`, so when the user does not set it
# apptainer still puts an EMPTY STRING in the environment. The key is then present and
# os.environ.get's default never fires, which parsed to [] -- an empty delta_indices,
# which makes extract_step_data load zero frames and blows up far downstream in
# processing_gr00t_n1d7.py::_get_vlm_inputs with "stack expects a non-empty TensorList".
# Bash's ${VAR:-default} treats empty as unset; Python's os.environ.get does not.
_delta = os.environ.get("GR00T_VIDEO_DELTA") or "0"
VIDEO_DELTA_INDICES: list[int] = [int(x) for x in _delta.split(",") if x.strip()]
if not VIDEO_DELTA_INDICES:
    raise ValueError(
        f"GR00T_VIDEO_DELTA={_delta!r} parsed to an empty delta_indices list. "
        "Use e.g. '0' (single frame) or '-1,0' (two frames)."
    )
if VIDEO_DELTA_INDICES[-1] != 0:
    raise ValueError(
        f"GR00T_VIDEO_DELTA={_delta!r} must end at 0 (the current frame), "
        f"got {VIDEO_DELTA_INDICES}."
    )
print(f"[modality] video delta_indices = {VIDEO_DELTA_INDICES} (GR00T_VIDEO_DELTA={_delta!r})")

mrl_unitree_g1_autocollect = {
    # Video: head (ego_view, 480px) + wrist (wrist_view, 720px). BOTH views are
    # load-bearing — the ego camera is what carries range (swapping it doubles the
    # tip-target error). The number of FRAMES per view is the measured question
    # settled above. Note the ego camera is 480 here where g1_dataset_red_2k was
    # 720 — the processor resizes both to image_target_size, so this costs ego
    # detail, not tokens.
    "video": ModalityConfig(
        delta_indices=VIDEO_DELTA_INDICES,
        modality_keys=["ego_view", "wrist_view"],
    ),
    # State: drill-tip pose + right-arm joints. `right_drill_tip_eef_9d` is
    # load-bearing — it is the reference frame the RELATIVE action below is
    # computed against (matched BY NAME against the action key), so it cannot be
    # dropped or renamed.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_drill_tip_eef_9d",
            "right_arm",
        ],
    ),
    # Action: 30-step chunk of drill-tip EEF targets.
    #   RELATIVE   -> deltas w.r.t. the current pose (key-matched to `state`)
    #   EEF        -> composed as SE(3), not vector-subtracted
    #   XYZ_ROT6D  -> the 9D layout the G1 FK/IK pipeline uses
    "action": ModalityConfig(
        delta_indices=list(range(G1_AUTOCOLLECT_CHUNK_SIZE)),
        modality_keys=["right_drill_tip_eef_9d"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            )
        ],
    ),
    # Language: task instruction stored as a task index in the parquet.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(mrl_unitree_g1_autocollect, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
