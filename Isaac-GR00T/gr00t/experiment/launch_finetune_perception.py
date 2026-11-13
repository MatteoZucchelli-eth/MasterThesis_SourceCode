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

# Launch finetuning of the PERCEPTION-ONLY model (VLM backbone + target heatmap
# head) for the short-range fine approach. Trains on every wrist-camera frame
# represented in the heatmap_labels/ sidecar parquets.

import os

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import PerceptionFinetuneConfig
from gr00t.configs.model.gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig
from gr00t.experiment.experiment import run
from gr00t.experiment.launch_finetune import load_modality_config, resolve_max_steps


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    ft_config = tyro.cli(PerceptionFinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag).value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Swap the model config for the perception one.
    model = Gr00tN1d7PerceptionConfig()
    config.model = model

    model.tune_llm = ft_config.tune_llm
    model.tune_visual = ft_config.tune_visual
    model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    # No projector/diffusion in this model; flags kept for pipeline compatibility.
    model.tune_projector = False
    model.tune_diffusion_model = False

    model.load_bf16 = False
    model.reproject_vision = False
    model.model_name = "nvidia/Cosmos-Reason2-2B"
    model.backbone_trainable_params_fp32 = True
    # Perception ignores state/action; keep the state-action pipeline trivial.
    model.use_relative_action = False
    model.state_dropout_prob = 0.0
    model.action_horizon = 1

    # Image resolution fed to the VLM.
    assert ft_config.image_crop_size < ft_config.image_target_size, (
        "image_crop_size must be smaller than image_target_size"
    )
    model.image_target_size = (ft_config.image_target_size, ft_config.image_target_size)
    model.image_crop_size = (ft_config.image_crop_size, ft_config.image_crop_size)
    model.shortest_image_edge = None
    model.crop_fraction = None
    print(
        f"[resolution] image_target_size={model.image_target_size} "
        f"image_crop_size={model.image_crop_size} "
        f"(crop_fraction {ft_config.image_crop_size / ft_config.image_target_size:.3f}); "
        f"the processor OUTPUT is {ft_config.image_target_size}px -- the crop is a zoom "
        f"augmentation, it is resized back up"
    )
    model.color_jitter_params = ft_config.color_jitter_params
    model.random_rotation_angle = None  # would desync the wall-normal label
    model.extra_augmentation_config = None

    # Perception head / loss.
    model.heatmap_hidden_dim = ft_config.heatmap_hidden_dim
    model.heatmap_target_sigma_px = ft_config.heatmap_target_sigma_px
    model.heatmap_loss_weight = ft_config.heatmap_loss_weight
    model.offset_loss_weight = ft_config.offset_loss_weight
    model.visibility_loss_weight = ft_config.visibility_loss_weight
    model.visibility_neg_weight = ft_config.visibility_neg_weight
    model.depth_loss_weight = ft_config.depth_loss_weight
    model.wall_normal_loss_weight = ft_config.wall_normal_loss_weight
    model.use_text_conditioning = ft_config.use_text_conditioning

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    # Count all sidecar frames, matching perception_labels_only below.
    config.training.max_steps = resolve_max_steps(
        ft_config,
        embodiment_tag,
        perception_labels_only=True,
    )
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project
    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    # Train on every sidecar frame. Out-of-view/occluded rows supervise visibility;
    # localization, depth, and normal losses are masked to visible rows.
    config.data.perception_labels_only = True

    run(config)
