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

"""Inference policy for the perception-only model (Gr00tN1d7Perception).

Unlike Gr00tPolicy (which calls model.get_action and decodes a robot action),
this policy calls model.predict and returns the target-perception outputs:
where the wall target is in the wrist camera (pixel coords), whether it is
visible, and optionally its depth and the wall normal.

It implements the standard BasePolicy.get_action interface, so it is served over
the SAME `get_action` endpoint as the VLA policy and queried with the SAME
PolicyClient — the "action" dict just carries perception outputs instead of a
motor command. See gr00t/eval/run_perception_server.py.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy


class PerceptionPolicy(BasePolicy):
    """Serve a Gr00tN1d7Perception checkpoint: image + prompt -> target location.

    Observation (same nested layout as Gr00tPolicy, but only video + language are
    required; state is optional and ignored by the model):
        video:    {"wrist_view": np.uint8 (B, T, H, W, C)}  (T = 1)
        language: {"<lang_key>": list[list[str]] of shape (B, 1)}

    Returned `action` dict (all np.float32 unless noted, batch-first):
        target_uv_px            (B, 2)  target center in RAW wrist-camera px
        target_uv_processed_px  (B, 2)  same, in the model's processed-image px
        visibility              (B,)    sigmoid prob. the target is visible
        visible                 (B,) bool  visibility > threshold
        target_depth_m          (B,)    camera-frame depth (if the depth head is on)
        wall_normal_cam         (B, 3)  unit wall normal in camera frame (if on)
    `info` carries the full softmax heatmap (B, gh, gw) when options={"return_heatmap": True}.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        visibility_threshold: float = 0.5,
    ):
        # Registers Gr00tN1d7Perception + its processor with AutoModel/AutoProcessor.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        self.embodiment_tag = embodiment_tag
        self.device = device
        self.visibility_threshold = visibility_threshold

        model_dir = Path(model_path)
        model = AutoModel.from_pretrained(model_dir)
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()  # deterministic center-crop transform (matches uv un-mapping)

        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' (value='{self.embodiment_tag.value}') "
                f"is not in this checkpoint. Available: {sorted(all_modality_configs.keys())}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        self._video_keys = self.modality_configs["video"].modality_keys
        self.language_key = self.modality_configs["language"].modality_keys[0]

        # Zero-state template: the model ignores state, but the shared processor
        # still normalizes it, so we feed zeros of the right per-key dim rather
        # than forcing the caller to supply a state they don't need.
        self._state_keys = self.modality_configs["state"].modality_keys
        self._state_T = len(self.modality_configs["state"].delta_indices)
        stats = getattr(self.processor, "statistics", {}) or {}
        self._state_dims: dict[str, int] = {}
        for k in self._state_keys:
            try:
                self._state_dims[k] = len(stats[self.embodiment_tag.value]["state"][k]["mean"])
            except Exception:
                self._state_dims[k] = 9  # right_drill_tip_eef_9d fallback

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def check_observation(self, observation: dict[str, Any]) -> None:
        for modality in ["video", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dict, got {type(observation[modality])}"
            )
        for video_key in self._video_keys:
            assert video_key in observation["video"], f"Video key '{video_key}' missing"
            v = observation["video"][video_key]
            assert isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 5, (
                f"Video '{video_key}' must be uint8 ndarray (B, T, H, W, C); got "
                f"{type(v)} {getattr(v, 'dtype', None)} {getattr(v, 'shape', None)}"
            )
            assert v.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video '{video_key}' horizon must be "
                f"{len(self.modality_configs['video'].delta_indices)}; got {v.shape[1]}"
            )
            assert v.shape[-1] == 3, f"Video '{video_key}' must have 3 channels; got {v.shape[-1]}"
        assert self.language_key in observation["language"], (
            f"Language key '{self.language_key}' missing"
        )

    def check_action(self, action: dict[str, Any]) -> None:
        # Perception outputs, not a robot action — nothing to validate against an
        # action modality. Kept as a no-op to satisfy the BasePolicy interface.
        return None

    def _unbatch(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        bs = observation["video"][self._video_keys[0]].shape[0]
        out = []
        for i in range(bs):
            item = {
                "video": {k: observation["video"][k][i] for k in self._video_keys},
                "language": {self.language_key: observation["language"][self.language_key][i]},
            }
            if "state" in observation:
                item["state"] = {
                    k: observation["state"][k][i]
                    for k in self._state_keys
                    if k in observation["state"]
                }
            out.append(item)
        return out

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        return_heatmap = bool(options.get("return_heatmap", False))

        unbatched = self._unbatch(observation)
        processed = []
        cam_sizes: list[tuple[int, int]] = []  # (W, H) per sample
        for obs in unbatched:
            frame = np.asarray(obs["video"][self._video_keys[0]])  # (T, H, W, C)
            cam_sizes.append((int(frame.shape[2]), int(frame.shape[1])))
            states = obs.get("state") or {
                k: np.zeros((self._state_T, self._state_dims[k]), dtype=np.float32)
                for k in self._state_keys
            }
            step = VLAStepData(
                images=obs["video"],
                states=states,
                actions={},
                text=obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]
            processed.append(self.processor(messages))

        collated = self.collate_fn(processed)  # BatchFeature({"inputs": {...}})

        with torch.inference_mode():
            pred = self.model.predict(**collated)  # model handles device/dtype cast

        uv_proc = pred["target_uv"].float().cpu()  # (B, 2)
        cam_uv = torch.stack(
            [
                self.model.uv_processed_to_camera(uv_proc[i : i + 1], camera_size=cam_sizes[i])[0]
                for i in range(uv_proc.shape[0])
            ]
        )
        vis = pred["visibility"].float().cpu().numpy().astype(np.float32)
        action: dict[str, Any] = {
            "target_uv_px": cam_uv.numpy().astype(np.float32),
            "target_uv_processed_px": uv_proc.numpy().astype(np.float32),
            "visibility": vis,
            "visible": (vis > self.visibility_threshold),
        }
        if "depth" in pred:
            action["target_depth_m"] = pred["depth"].float().cpu().numpy().astype(np.float32)
        if "wall_normal" in pred:
            action["wall_normal_cam"] = pred["wall_normal"].float().cpu().numpy().astype(np.float32)

        info: dict[str, Any] = {}
        if return_heatmap:
            info["heatmap"] = pred["heatmap"].float().cpu().numpy().astype(np.float32)
        return action, info
