# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 30
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # Tip-target head parameters. All default off so existing embodiments /
    # checkpoints are unaffected; enable via the finetune config / CLI.
    # When enabled, an auxiliary head attention-pools the VLM latent (learned
    # queries cross-attending over ALL vision-language tokens, so language can
    # select which target to localize) and regresses the target pose in the
    # end-effector (drill-tip) frame as xyz(3) + rot6d(6) (the
    # `next.tip_target_9d` column in the parquet). xyz is divided by
    # `tip_target_xyz_scale` inside the head; the distance to target is derived
    # as ||xyz|| rather than regressed separately. rot6d is orthonormalized
    # (Gram-Schmidt) only at inference (Zhou et al., CVPR 2019).
    #
    # The position error is split into a LATERAL and a DEPTH term, and the two
    # are weighted separately. The split happens in the CAMERA frame, not the
    # label's: the camera->tip rotation recovered by Kabsch over 44.7k
    # (sidecar u/v/depth_z, next.tip_target_9d) pairs -- rigid, rms residual
    # 1.18 mm, since the wrist camera and the drill tip share a link -- puts the
    # optical axis at [0.9037, -0.0012, -0.4281] in the tip frame. Tip-frame `z`
    # is therefore NOT depth: depth is 0.9037*x - 0.4281*z, and splitting the
    # label's own x,y vs z would leave 90% of the depth direction in the
    # "lateral" term.
    #
    # Why bother: lateral error is directly observable (1 processed px = 0.381 mm
    # at the median 148 mm working depth) while depth has to be inferred from
    # apparent size, so the two deserve different loss budgets. Camera-frame
    # target statistics over 44.7k visible sidecar frames:
    #     cam X  +3.7 +/- 38.9 mm     cam Y  -2.0 +/- 25.4 mm
    #     depth  148.0 +/- 30.0 mm    (range 53-225 mm)
    #
    # `tip_target_cam_axes` holds that rotation; if the camera is remounted,
    # re-derive it by Kabsch-fitting the sidecar's (u, v, depth_z) against
    # `next.tip_target_9d` — both describe the same point, in the camera and the
    # tip frame respectively, and the transform is rigid (they share a link).
    # A rotation preserves the L2 norm, so lat=2/3 + depth=1/3 reproduces the
    # previous single xyz MSE exactly -- the split is inert until reweighted.
    #
    # Conditioning is taught by teacher forcing: training conditions on the
    # ground-truth 9d (plus small noise), inference on the head's own
    # prediction. Frames with NaN labels are excluded from
    # the loss and conditioned on the (detached) prediction instead. Setting
    # `tip_target_tf_gate_mm` hands conditioning over to the head's own
    # prediction once its error is small enough (see the finetune config).
    use_tip_target_head: bool = False
    tip_target_loss_weight: float = 0.5  # overall multiplier vs the action loss
    # Position split, in CAMERA-frame components. 2/3 + 1/3 == the old single xyz MSE.
    tip_target_lat_loss_weight: float = 2.0 / 3.0  # perpendicular to the optical axis
    tip_target_depth_loss_weight: float = 1.0 / 3.0  # along the optical axis (the hard one)
    tip_target_rot_loss_weight: float = 1.0  # weight of the rot6d MSE term
    # Camera axes in the TIP frame, row-major 3x3: row k is camera axis k
    # (X, Y right/down in the image; Z = optical axis = depth). Kabsch over the
    # heatmap sidecar, rms 1.18 mm. Stored in the config so inference and any
    # re-derivation stay in sync with the checkpoint that was trained on it.
    tip_target_cam_axes: list = field(
        default_factory=lambda: [
            [0.4281, -0.0108, 0.9037],
            [-0.0057, -0.9999, -0.0093],
            [0.9037, -0.0012, -0.4281],
        ]
    )
    # Probability of hiding the wrist image tokens from the head's pooling during
    # training, so it has to use the ego view too. 0 disables. The view split is read
    # off the image-token run structure, so these two must match the modality config's
    # `video.modality_keys` order (ego_view, wrist_view -> 2 views, wrist is index 1).
    tip_target_wrist_mask_prob: float = 0.0
    tip_target_num_views: int = 2
    tip_target_wrist_view_index: int = 1
    use_tip_target_conditioning: bool = True
    tip_target_xyz_scale: float = 0.1  # metres; xyz is divided by this inside the head
    tip_target_conditioning_noise_xyz: float = 0.005  # metres; teacher-forcing noise std
    tip_target_conditioning_noise_rot: float = 0.02  # rot6d units; teacher-forcing noise std
    tip_target_hidden_dim: int = 512
    tip_target_num_queries: int = 4  # learned queries for the attention pooling
    tip_target_attn_heads: int = 8  # heads of the pooling cross-attention
    # "attn_residual" (default) = learned-query cross-attention pooling started at the
    # mean-pool solution (zero-init output projection + mean residual), so the attention
    # only learns a correction and the mean path carries gradient from step 0;
    # "mean" = plain masked-mean pool. The bare no-residual "attn" pool was removed.
    tip_target_pool: str = "attn_residual"
    # Scheduled sampling for the conditioning token: hand over from the label to the
    # head's own prediction once its error is small enough. None = always teacher-force.
    tip_target_tf_gate_mm: float | None = None
    tip_target_tf_ramp_steps: int = 2000
    tip_target_tf_max_pred_prob: float = 1.0
    tip_target_tf_gate_patience: int = 200
    tip_target_tf_gate_ema: float = 0.99

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
