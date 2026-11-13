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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import math
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def count_trainable_steps(dataset_path: str, action_horizon: int) -> int:
    """Number of timesteps eligible to start an action chunk, across the dataset.

    Mirrors ShardedSingleStepDataset.get_effective_episode_length: the last
    `action_horizon - 1` frames of an episode cannot start a full chunk. Reads only
    meta/episodes.jsonl (no parquet), so this is cheap enough to run at launch.
    """
    episodes_file = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(
            f"Cannot derive max_steps from --num-epochs: {episodes_file} not found. "
            "Pass --max-steps instead."
        )
    with open(episodes_file, "r") as f:
        lengths = [json.loads(line)["length"] for line in f if line.strip()]
    return sum(max(0, length - action_horizon + 1) for length in lengths)


def count_perception_label_steps(dataset_path: str, action_horizon: int) -> int:
    """Trainable steps represented by perception-label sidecars.

    Mirrors ``ShardedSingleStepDataset.get_valid_step_indices`` with
    ``perception_labels_only=True``: every sidecar row that can still start a full
    action chunk counts. Episodes without a sidecar contribute nothing.

    Reads one column from every heatmap_labels/ parquet. The submit scripts stage the
    dataset on local NVMe first (they already pay this scan at shard time), where it
    costs seconds; on Lustre it would not.
    """
    import pandas as pd

    root = Path(dataset_path)
    episodes_file = root / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(
            f"Cannot derive max_steps from --num-epochs: {episodes_file} not found. "
            "Pass --max-steps instead."
        )
    # Keyed by episode index rather than rebuilt from chunk_size, so a dataset whose
    # chunking differs from meta/info.json still resolves.
    label_files = {
        int(p.stem.split("_")[1]): p
        for p in (root / "heatmap_labels").glob("chunk-*/episode_*.parquet")
    }
    if not label_files:
        raise FileNotFoundError(
            f"Cannot derive max_steps from --num-epochs: no heatmap_labels/ parquets "
            f"under {root}. Pass --max-steps instead."
        )

    total = 0
    labelled = 0
    with open(episodes_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            episode = json.loads(line)
            path = label_files.get(int(episode["episode_index"]))
            if path is None:
                continue
            effective = max(0, int(episode["length"]) - action_horizon + 1)
            if effective == 0:
                continue
            frame_index = pd.read_parquet(path, columns=["frame_index"])["frame_index"].to_numpy()
            total += int((frame_index < effective).sum())
            labelled += 1
    print(
        f"[epochs] perception-label scan: {labelled:,}/{len(label_files):,} labelled episodes "
        f"contribute {total:,} trainable steps"
    )
    return total


def resolve_action_horizon(embodiment_tag: str) -> int:
    """Chunk length the dataset uses, from the registered modality config.

    Must match ShardedSingleStepDataset, which derives it from the action modality's
    delta indices (NOT from model.action_horizon, which can differ -- the processor
    pads to the model's value).
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    delta_indices = MODALITY_CONFIGS[embodiment_tag]["action"].delta_indices
    return max(delta_indices) - min(delta_indices) + 1


def resolve_max_steps(
    ft_config,
    embodiment_tag: str,
    perception_labels_only: bool = False,
) -> int:
    """max_steps from --num-epochs when given, else --max-steps verbatim.

    One "epoch" is a target expressed in samples: num_epochs * trainable_steps
    consumed in total. It is NOT a pass over the data -- the training iterator draws
    batches from a shard buffer with replacement (see trainer._PrefetchIterator), so
    some frames are visited several times before others are visited once. Read it as
    "each frame seen num_epochs times in expectation".

    ``perception_labels_only`` must match ``config.data.perception_labels_only``:
    perception finetunes require sidecars, so their epoch is counted over sidecar
    frame indices rather than blindly counting every episode.
    """
    if ft_config.num_epochs is None:
        return ft_config.max_steps
    if ft_config.num_epochs <= 0:
        raise ValueError(f"--num-epochs must be > 0, got {ft_config.num_epochs}")

    action_horizon = resolve_action_horizon(embodiment_tag)
    if perception_labels_only:
        trainable = count_perception_label_steps(ft_config.dataset_path, action_horizon)
    else:
        trainable = count_trainable_steps(ft_config.dataset_path, action_horizon)
    per_step = ft_config.global_batch_size * ft_config.gradient_accumulation_steps
    steps = math.ceil(ft_config.num_epochs * trainable / per_step)
    print(
        f"[epochs] {ft_config.num_epochs} epoch(s) over {trainable:,} trainable steps "
        f"(action_horizon {action_horizon}) "
        f"at {per_step} samples/optimizer step (global_batch_size "
        f"{ft_config.global_batch_size} x grad_accum {ft_config.gradient_accumulation_steps}) "
        f"-> --max-steps {steps:,} (overriding {ft_config.max_steps:,})"
    )
    if ft_config.save_steps > steps:
        print(
            f"[epochs] WARNING: save_steps ({ft_config.save_steps:,}) > max_steps ({steps:,}); "
            "only the final model would be written."
        )
    return steps


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
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

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Image resolution fed to the VLM. Set the (target, crop) pair together and leave
    # shortest_image_edge / crop_fraction as None (experiment.py validates this).
    assert ft_config.image_crop_size < ft_config.image_target_size, (
        "image_crop_size must be smaller than image_target_size"
    )
    config.model.image_target_size = (ft_config.image_target_size, ft_config.image_target_size)
    config.model.image_crop_size = (ft_config.image_crop_size, ft_config.image_crop_size)
    config.model.shortest_image_edge = None
    config.model.crop_fraction = None

    # Optional tip-target head.
    config.model.use_tip_target_head = ft_config.use_tip_target_head
    config.model.tip_target_loss_weight = ft_config.tip_target_loss_weight
    config.model.tip_target_pool = ft_config.tip_target_pool
    config.model.tip_target_num_queries = ft_config.tip_target_num_queries
    config.model.tip_target_attn_heads = ft_config.tip_target_attn_heads
    config.model.tip_target_lat_loss_weight = ft_config.tip_target_lat_loss_weight
    config.model.tip_target_depth_loss_weight = ft_config.tip_target_depth_loss_weight
    config.model.tip_target_rot_loss_weight = ft_config.tip_target_rot_loss_weight
    config.model.tip_target_wrist_mask_prob = ft_config.tip_target_wrist_mask_prob
    config.model.use_tip_target_conditioning = ft_config.use_tip_target_conditioning
    config.model.tip_target_xyz_scale = ft_config.tip_target_xyz_scale
    config.model.tip_target_tf_gate_mm = ft_config.tip_target_tf_gate_mm
    config.model.tip_target_tf_ramp_steps = ft_config.tip_target_tf_ramp_steps
    config.model.tip_target_tf_max_pred_prob = ft_config.tip_target_tf_max_pred_prob
    config.model.tip_target_tf_gate_patience = ft_config.tip_target_tf_gate_patience
    config.model.tip_target_tf_gate_ema = ft_config.tip_target_tf_gate_ema

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
    config.training.max_steps = resolve_max_steps(ft_config, embodiment_tag)
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
