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

from gr00t.configs.model.gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig
from gr00t.model.gr00t_n1d7.perception import Gr00tN1d7Perception
from gr00t.model.gr00t_n1d7.processing_perception import Gr00tN1d7PerceptionProcessor
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.registry import register_model


PARTIAL_LOAD_PREFIXES = ("backbone.", "perception_head.")


def _resolve_checkpoint_dir(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(checkpoint, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
    )


def load_matching_weights(model, checkpoint: str) -> None:
    """Selectively load backbone (and, if present, perception_head) weights.

    Works with either a full VLA checkpoint (Gr00tN1d7: backbone + action head —
    the action-head weights are ignored) or a previous perception checkpoint.
    Keys not present in the checkpoint (e.g. a fresh perception head) keep their
    constructor initialization.
    """
    from safetensors import safe_open

    ckpt_dir = _resolve_checkpoint_dir(checkpoint)
    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted({ckpt_dir / fname for fname in weight_map.values()})
    elif (ckpt_dir / "model.safetensors").exists():
        files = [ckpt_dir / "model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors weights found in {ckpt_dir}")

    own_state = model.state_dict()
    to_load = {}
    mismatched = []
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if not key.startswith(PARTIAL_LOAD_PREFIXES):
                    continue
                if key not in own_state:
                    mismatched.append(key)
                    continue
                tensor = sf.get_tensor(key)
                if own_state[key].shape != tensor.shape:
                    mismatched.append(key)
                    continue
                to_load[key] = tensor

    backbone_keys = {k for k in own_state if k.startswith("backbone.")}
    missing_backbone = backbone_keys - set(to_load)
    if missing_backbone:
        raise RuntimeError(
            f"Checkpoint {checkpoint} is missing {len(missing_backbone)} backbone keys "
            f"(e.g. {sorted(missing_backbone)[:5]}). Wrong checkpoint?"
        )
    head_loaded = sum(1 for k in to_load if k.startswith("perception_head."))
    model.load_state_dict(to_load, strict=False)
    logging.info(
        f"Loaded {len(to_load)} tensors from {checkpoint} "
        f"({len(to_load) - head_loaded} backbone, {head_loaded} perception_head; "
        f"{len(mismatched)} skipped)"
    )
    if head_loaded == 0:
        logging.info("perception_head not in checkpoint - keeping fresh initialization")


class Gr00tN1d7PerceptionPipeline(Gr00tN1d7Pipeline):
    model_class = Gr00tN1d7Perception
    processor_class = Gr00tN1d7PerceptionProcessor

    def _create_model(self):
        """Build the perception model and overlay backbone weights from the
        starting checkpoint (full VLA or previous perception checkpoint).

        Unlike the base pipeline this never calls AutoModel.from_pretrained on
        the checkpoint: the checkpoint's config describes a different model
        (the full VLA), so the perception model is constructed from the
        perception config and only matching weights are transferred.
        """
        # Checked BEFORE constructing anything: the backbone is a 2B model that gets
        # downloaded and materialized, and there is no point paying for that to then
        # fail. No silent fall-back to the bare Cosmos-Reason2-2B foundation weights
        # either -- the backbone is ALWAYS built from that architecture, so a missing
        # checkpoint would not fail loudly, it would just train the un-adapted
        # foundation and look like a perfectly normal run.
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        checkpoint = self.config.training.start_from_checkpoint
        # "scratch"/"none"/"fresh" were the sentinels that selected the removed
        # bare-Cosmos path; treat them as the mistake they now are rather than letting
        # them reach the loader as a literal directory name.
        if isinstance(checkpoint, str) and checkpoint.strip().lower() in {
            "scratch",
            "none",
            "fresh",
        }:
            checkpoint = None
        if not skip_weight_loading and not checkpoint:
            raise ValueError(
                "The perception finetune requires a GR00T-lineage backbone checkpoint: "
                "pass --base-model-path (nvidia/GR00T-N1.7-3B, one of your VLA runs, or "
                "a previous perception checkpoint). Starting from the bare "
                "Cosmos-Reason2-2B foundation is no longer supported -- it threw away "
                "GR00T's robot adaptation for no gain. Use --skip-weight-loading only "
                "for architecture-only smoke tests."
            )

        model = self.model_class(
            self.config.model,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
        )

        if not skip_weight_loading:
            load_matching_weights(model, checkpoint)

        # Re-apply trainable flags after weight loading.
        model.backbone.set_trainable_parameters(
            tune_llm=self.config.model.tune_llm,
            tune_visual=self.config.model.tune_visual,
            tune_top_llm_layers=self.config.model.tune_top_llm_layers,
        )

        from gr00t.experiment.dist_utils import get_rank

        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        return model

    def _create_dataset(self, save_cfg_dir: Path):
        """Same as the base pipeline, but ALWAYS build a fresh perception processor.

        AutoProcessor.from_pretrained on a VLA checkpoint would (a) resolve to the
        base Gr00tN1d7Processor with no keypoint handling and (b) silently reuse
        the checkpoint's stored image sizing (e.g. 256px / shortest_image_edge),
        overriding the perception config's resolution — which is precisely what
        this model must control. Statistics are set from the actual dataset by
        ShardedMixtureDataset.merge_statistics, so nothing is lost."""
        from gr00t.data.dataset.factory import DatasetFactory
        from gr00t.experiment.dist_utils import get_rank
        from gr00t.model.gr00t_n1d7.setup import convert_tensors_to_lists

        processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=self._get_statistics(),
            embodiment_id_mapping=self._get_embodiment_id_mapping(),
            max_state_dim=self.model_config.max_state_dim,
            max_action_dim=self.model_config.max_action_dim,
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
            use_relative_action=self.model_config.use_relative_action,
            exclude_state=self.model_config.exclude_state,
            state_dropout_prob=self.model_config.state_dropout_prob,
            use_mean_std=self.model_config.use_mean_std,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
        )

        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        if get_rank() == 0:
            stats = train_dataset.get_dataset_statistics()
            with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                json.dump(convert_tensors_to_lists(stats), f, indent=2)
            logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset


register_model(Gr00tN1d7PerceptionConfig, Gr00tN1d7PerceptionPipeline)
