"""GR00T modality config for the G1 PALM-EEF embodiment.

Usage (fine-tuning):
    uv run python gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path /path/to/g1_dataset \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path /path/to/this/file/gr00t_modality_config.py \\
        --num-gpus 1 \\
        --output-dir /tmp/g1_PALM_eef \\
        --max-steps 5000 --save-steps 1000 \\
        --global-batch-size 32

The recorded dataset has:
  state  (35-dim): left_palm_eef_9d(9) + right_drill_tip_eef_9d(9)
                   + left_arm(7) + right_arm(7) + waist(3)
  action (18-dim): left_palm_eef_9d(9) + right_drill_tip_eef_9d(9)
  video:           observation.images.ego_view    →  "ego_view" key
                   observation.images.wrist_view  →  "wrist_view" key
  language:        annotation.human.action.task_description

The RIGHT end-effector is the DRILL TIP, not the palm. The tip sits ~0.20 m out along
the palm's +x axis, so a 1 deg palm orientation error becomes ~3.5 mm of tip error
(delta_p_tip = delta_p_palm + delta_theta x R*d). Regressing palm poses spends loss
budget on a rotation channel the task scores through that lever arm; the tip's position
error maps 1:1 onto the distance from the target square. State and action name the SAME
frames, which is what lets the RELATIVE action representation below derive a true tip
displacement by key-matching state to action. Teleoperation still *inputs* palm poses
(the operator holds controllers in their hands) — that is the input device, not the label.

Auxiliary per-frame columns (in the parquet, NOT GR00T modalities):
  next.tip_target_9d:    pose of the red target IN THE DRILL-TIP FRAME,
                   T_tip^-1 * T_target, as xyz(3) + rot6d(6) — the same 9D layout
                   as the state/action palm poses. Tip-relative, so it is
                   egocentric and invariant to the (randomised) base yaw.
                     xyz[0:3]   tip → target displacement, tip frame.
                                Direction = where to move; norm = the distance.
                     rot6d[3:9] a genuine 3-DOF rotation: how the drill must
                                rotate to sit square-on to the wall.
  next.tip_target_dist:  ‖xyz‖ in metres. Reaches ~0 during the hold-on-target
                   phase at the end of an episode.

  Predict rot6d and the distance with auxiliary heads: rot6d is well-posed here
  because it encodes a rotation. Do NOT try to encode the tip → target *direction*
  as a rot6d — a direction is only 2 DOF and has no canonical rotation (the roll
  about it is arbitrary, and any Gram-Schmidt gauge fixing is singular where the
  direction meets the reference axis). Take the direction from xyz instead.

  Both are TRAINING LABELS, not policy inputs. The tip pose is observable at
  deployment (FK from proprioception) but the target position is not — it has to
  be inferred from the camera, which is exactly what the model is being trained
  to do. Putting these in observation.state would hand the model the answer and
  teach it to ignore the image (causal confusion): it would train beautifully and
  fail on the robot. Predict them with an auxiliary head and condition action
  generation on the *prediction*.

  GR00T's loader only consumes video/state/action/language/mask
  (ALLOWED_MODALITIES), so these are not registered as modalities below; read
  them straight from the parquet for a distance/value head, reward shaping, or
  episode filtering.

  Frames whose label could not be resolved (no red_target TF) are written as NaN
  rather than 0.0, so an unlabelled frame is never mistaken for "on target".

After fine-tuning, deploy with chunk_size=G1_PALM_EEF_CHUNK_SIZE (see below).
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
G1_PALM_EEF_CHUNK_SIZE: int = 30

mrl_unitree_g1 = {
    # Video: head (ego_view) + wrist (wrist_view) cameras, each providing the
    # previous and current frame (delta_indices=[-1, 0]).
    "video": ModalityConfig(
        delta_indices=[-1, 0],
        modality_keys=["ego_view", "wrist_view"],
    ),
    # State: current EEF poses (FK from Pinocchio) + arm + waist joints.
    # Left = palm, right = drill tip.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            # "left_palm_eef_9d",
            "right_drill_tip_eef_9d",
            "left_arm",
            "right_arm",
            # "waist",
        ],
    ),
    # Action: predict a chunk of EEF targets only (no base velocity).
    # RELATIVE means the pipeline computes deltas w.r.t. the current pose (matched by
    # key against `state` above), which improves generalisation across different start
    # configurations. XYZ_ROT6D matches the 9D layout used by the G1 FK / IK pipeline.
    "action": ModalityConfig(
        delta_indices=list(range(G1_PALM_EEF_CHUNK_SIZE)),
        modality_keys=[
            # "left_palm_eef_9d",
            "right_drill_tip_eef_9d",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            )
            # ,
            # ActionConfig(
            #     rep=ActionRepresentation.RELATIVE,
            #     type=ActionType.EEF,
            #     format=ActionFormat.XYZ_ROT6D,
            # ),
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
