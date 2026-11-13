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

"""Processor for the perception-only GR00T model.

Extends Gr00tN1d7Processor with keypoint-aware image augmentation: the target
pixel label (target_uv, raw wrist-camera px) rides through the SAME albumentations
transform as the image (crop / resize), so the supervised heatmap stays aligned
with what the model sees. If the random crop pushes the target out of the frame,
the sample's visibility label is flipped to 0 — the augmented image genuinely no
longer shows the target.
"""

from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from .image_augmentations import build_image_transformations_albumentations
from .processing_gr00t_n1d7 import Gr00tN1d7Processor


class Gr00tN1d7PerceptionProcessor(Gr00tN1d7Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_albumentations, (
            "Gr00tN1d7PerceptionProcessor requires albumentations transforms "
            "(use_albumentations_transforms=True) for keypoint-aware augmentation."
        )
        # Rebuild the transforms with keypoint support (same parameters).
        self.train_image_transform, self.eval_image_transform = (
            build_image_transformations_albumentations(
                self.image_target_size,
                self.image_crop_size,
                self.random_rotation_angle,
                self.color_jitter_params,
                self.shortest_image_edge,
                self.crop_fraction,
                extra_augmentation_config=self.extra_augmentation_config,
                with_keypoints=True,
            )
        )
        # Per-sample keypoint handoff between __call__ and _get_vlm_inputs
        # (processor instances are per-worker; calls are sequential).
        self._kp_uv_in: np.ndarray | None = None
        self._kp_uv_out: np.ndarray | None = None

    def __call__(self, messages: list[dict[str, Any]]):
        assert len(messages) == 1
        content = messages[0]["content"]
        meta = content.metadata or {}

        uv = meta.get("target_uv")
        self._kp_uv_in = None
        self._kp_uv_out = None
        if uv is not None and np.all(np.isfinite(uv)):
            self._kp_uv_in = np.asarray(uv, dtype=np.float32)

        transformed_inputs = super().__call__(messages)

        # Final image size the label lives in: (H, W) of the processed image.
        proc_h, proc_w = self.image_target_size

        visible = bool(meta.get("target_visible", False))
        uv_out = self._kp_uv_out
        if visible and uv_out is not None:
            u, v = float(uv_out[0]), float(uv_out[1])
            if 0.0 <= u < proc_w and 0.0 <= v < proc_h:
                uv_final = np.array([u, v], dtype=np.float32)
            else:
                # Augmentation cropped the target out of the frame.
                visible = False
                uv_final = np.array([np.nan, np.nan], dtype=np.float32)
        else:
            visible = False
            uv_final = np.array([np.nan, np.nan], dtype=np.float32)

        dtype = torch.get_default_dtype()
        transformed_inputs["target_uv"] = torch.as_tensor(uv_final, dtype=dtype)
        transformed_inputs["target_visible"] = torch.tensor(float(visible), dtype=dtype)
        transformed_inputs["target_depth"] = torch.tensor(
            float(meta.get("target_depth", np.nan)), dtype=dtype
        )
        wall_normal = meta.get("target_wall_normal")
        if wall_normal is None:
            wall_normal = np.full(3, np.nan, dtype=np.float32)
        transformed_inputs["target_wall_normal"] = torch.as_tensor(
            np.asarray(wall_normal, dtype=np.float32), dtype=dtype
        )
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: dict[str, list[Image.Image]],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform,
        language: str,
    ):
        """Single-view, single-frame transform that carries the target keypoint."""
        assert len(image_keys) == 1, (
            f"Perception processor expects exactly one camera view (the wrist), "
            f"got {image_keys}. Use a wrist-only modality config."
        )
        view = image_keys[0]
        frames = images[view]
        assert len(frames) == 1, (
            f"Perception processor expects a single current frame (video "
            f"delta_indices=[0]), got {len(frames)} frames."
        )
        if masks:
            raise ValueError("Mask transforms are not supported by the perception processor.")

        img_array = np.array(frames[0])
        keypoints = (
            self._kp_uv_in.reshape(1, 2)
            if self._kp_uv_in is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        result = image_transform(image=img_array, keypoints=keypoints)

        out_img = result["image"]
        if out_img.dtype == np.float32:
            out_img = (out_img * 255).astype(np.uint8)
        elif out_img.dtype != np.uint8:
            raise ValueError(f"Unexpected image dtype after transform: {out_img.dtype}")

        out_kps = np.asarray(result["keypoints"], dtype=np.float32).reshape(-1, 2)
        self._kp_uv_out = out_kps[0] if len(out_kps) > 0 else None

        img_tensor = torch.from_numpy(out_img).permute(2, 0, 1)  # (C, H, W) uint8
        stacked_images = img_tensor.unsqueeze(0).numpy()  # (T*V=1, C, H, W)
        return self._apply_vlm_processing(stacked_images, language)


AutoProcessor.register("Gr00tN1d7Perception", Gr00tN1d7PerceptionProcessor)
