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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                # Without this the model keeps the BASE checkpoint's action_horizon
                # (40 for GR00T-N1.7-3B) while the processor and dataset use
                # config.model.action_horizon. Training takes its trajectory length from
                # the batch, so it trains on the processor's horizon, but
                # get_action samples noise of config.action_horizon -- so the DiT would
                # roll out 40 steps having only ever seen length-30 sequences. No
                # parameter is sized by action_horizon (verified: identical state-dict
                # shapes at 30 and 40), so this only aligns the rollout length.
                action_horizon=self.config.model.action_horizon,
                use_tip_target_head=self.config.model.use_tip_target_head,
                tip_target_loss_weight=self.config.model.tip_target_loss_weight,
                tip_target_pool=self.config.model.tip_target_pool,
                tip_target_num_queries=self.config.model.tip_target_num_queries,
                tip_target_attn_heads=self.config.model.tip_target_attn_heads,
                tip_target_lat_loss_weight=self.config.model.tip_target_lat_loss_weight,
                tip_target_depth_loss_weight=self.config.model.tip_target_depth_loss_weight,
                tip_target_rot_loss_weight=self.config.model.tip_target_rot_loss_weight,
                tip_target_cam_axes=self.config.model.tip_target_cam_axes,
                tip_target_wrist_mask_prob=self.config.model.tip_target_wrist_mask_prob,
                tip_target_num_views=self.config.model.tip_target_num_views,
                tip_target_wrist_view_index=self.config.model.tip_target_wrist_view_index,
                use_tip_target_conditioning=self.config.model.use_tip_target_conditioning,
                tip_target_xyz_scale=self.config.model.tip_target_xyz_scale,
                tip_target_tf_gate_mm=self.config.model.tip_target_tf_gate_mm,
                tip_target_tf_ramp_steps=self.config.model.tip_target_tf_ramp_steps,
                tip_target_tf_max_pred_prob=self.config.model.tip_target_tf_max_pred_prob,
                tip_target_tf_gate_patience=self.config.model.tip_target_tf_gate_patience,
                tip_target_tf_gate_ema=self.config.model.tip_target_tf_gate_ema,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            # The optional tip-target head is newly added on top of a base
            # checkpoint that has no such weights, so its parameters are
            # expected to be missing. They must be explicitly re-initialized:
            # from_pretrained materializes the model on the meta device, and
            # transformers' missing-key fallback leaves bare nn.Parameters
            # (pool_queries) as uninitialized memory, which NaNs the head's
            # attention on the first forward.
            if self.config.model.use_tip_target_head and any(
                "tip_target_head" in key for key in missing_keys
            ):
                model.action_head.tip_target_head.reset_parameters()
                logging.info("tip_target_head not in checkpoint - re-initialized explicitly")

            # Same story for the persistent scheduled-sampling buffers
            # (action_head.tip_target_tf_*): they live on the action head rather than
            # inside tip_target_head, so they need their own re-initialization.
            if self.config.model.use_tip_target_head and any(
                "tip_target_tf_" in key for key in missing_keys
            ):
                model.action_head.reset_tip_target_tf_state()
                logging.info("tip_target teacher-forcing state not in checkpoint - schedule reset")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            # Mirror the missing-keys tolerance above: a checkpoint trained WITH
            # an auxiliary head (or its conditioning embed) can be loaded with
            # that option disabled — the now-unused head weights are dropped
            # instead of hard-failing.
            dropped_aux = [k for k in unexpected_keys if "progress_head" in k or "tip_target" in k]
            if dropped_aux:
                logging.info(
                    f"auxiliary head weights in checkpoint ignored (head disabled): {dropped_aux}"
                )
                unexpected_keys = [k for k in unexpected_keys if k not in dropped_aux]
            mismatched_keys = loading_info.get("mismatched_keys", [])
            other_missing = [
                k
                for k in missing_keys
                if "mask_token" not in k and "progress_head" not in k and "tip_target" not in k
            ]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        # Surface the horizon that actually reached the model: it silently inherited the
        # base checkpoint's value until 2026-07-30, and a train/inference mismatch here
        # is invisible in the loss.
        logging.info(
            f"[action-horizon] model action_horizon={model.config.action_horizon} "
            f"(requested {self.config.model.action_horizon}); the dataset chunk length "
            "comes from the modality config's action delta_indices and must match"
        )
        logging.debug(f"Model Config: {model.config}")
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        # Log the EFFECTIVE image geometry at INFO. These four interact by precedence
        # (shortest_image_edge beats image_target_size, crop_fraction beats
        # image_crop_size) and used to be silently dropped on the way here, so every
        # run trained at the base checkpoint's resolution whatever was requested. Print
        # what the processor actually ended up with, not what was asked for.
        _target = getattr(processor, "image_target_size", None)
        _crop = getattr(processor, "image_crop_size", None)
        _edge = getattr(processor, "shortest_image_edge", None)
        _frac = getattr(processor, "crop_fraction", None)
        _resize_to = _edge if _edge is not None else (_target[0] if _target else None)
        _crop_frac = _frac if _frac is not None else (_crop[0] / _target[0] if _crop else None)
        logging.info(
            "[resolution] EFFECTIVE image geometry: "
            f"image_target_size={_target} image_crop_size={_crop} "
            f"shortest_image_edge={_edge} crop_fraction={_frac}"
        )
        if _resize_to is not None:
            # Backbone patches are 16 px with spatial_merge_size 2, so one visual token
            # covers a 32 px block of the resized image.
            _tokens = (_resize_to // 32) ** 2
            logging.info(
                f"[resolution] every input image becomes {_resize_to}x{_resize_to} "
                f"(crop fraction {_crop_frac:.3f} of the frame, then resized back up) "
                f"=> ~{_tokens} visual tokens per image"
            )
        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Rank-guarded for the same reason as final_processor_config.json above.
        if get_rank() == 0:
            stats = train_dataset.get_dataset_statistics()
            stats_dict = convert_tensors_to_lists(stats)
            with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                json.dump(stats_dict, f, indent=2)
            logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
