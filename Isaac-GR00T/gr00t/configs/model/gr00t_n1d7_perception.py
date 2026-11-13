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

from dataclasses import dataclass

from . import register_model_config
from .gr00t_n1d7 import Gr00tN1d7Config


@dataclass
class Gr00tN1d7PerceptionConfig(Gr00tN1d7Config):
    """Config for the perception-only GR00T model (VLM backbone + target heatmap head).

    Used by the short-range fine-approach stage: the model sees ONLY the wrist
    camera (single current frame) + the task instruction, and predicts
      - a spatial heatmap over the VLM image-token grid locating the wall
        target in wrist-camera pixel coordinates,
      - a subpixel (du, dv) offset at each cell (CenterNet-style refinement),
      - a visibility logit ("is the target in this image at all?"),
      - optionally the target depth (camera-frame z) and the wall normal.

    No action head / DiT is instantiated; state and action inputs are ignored.
    Labels come from the heatmap_labels/ sidecar parquets and training samples
    every sidecar row (data.perception_labels_only=True).
    """

    # Model identification (overrides Gr00tN1d7)
    model_type: str = "Gr00tN1d7Perception"

    # Heatmap head
    heatmap_hidden_dim: int = 512
    """Hidden width of the per-token MLP producing heatmap/offset logits."""

    heatmap_target_sigma_px: float = 8.0
    """Std (in FINAL processed-image pixels) of the Gaussian rendered around the
    labelled target center to form the soft heatmap target."""

    heatmap_loss_weight: float = 1.0
    """Weight of the heatmap cross-entropy (visible frames only)."""

    offset_loss_weight: float = 1.0
    """Weight of the subpixel offset L1 loss at the ground-truth cell (visible frames only)."""

    visibility_loss_weight: float = 0.5
    """Weight of the visibility BCE (all frames)."""

    visibility_neg_weight: float = 1.0
    """Per-sample weight on target_visible=0 rows inside the visibility BCE.

    1.0 gives an unweighted loss. A class-balanced choice is n_pos/n_neg, measured
    from the generated sidecar for the dataset being trained. The loss is normalized
    by the weight sum, so its magnitude does not move with this value."""

    depth_loss_weight: float = 0.1
    """Weight of the target-depth SmoothL1 loss (visible frames only). 0 disables the head."""

    depth_scale: float = 0.3
    """Metres; depth is divided by this inside the head so its loss is O(1) at close range."""

    wall_normal_loss_weight: float = 0.1
    """Weight of the wall-normal MSE loss (visible frames only). 0 disables the head."""

    perception_pool_heads: int = 8
    """Attention heads of the learned-query pooling used by the visibility/depth/normal heads."""

    use_text_conditioning: bool = True
    """Let the head cross-attend over the instruction tokens (TargetPerceptionHead).

    Required for any dataset where the instruction selects between several candidate
    targets in the same frame — e.g. red/green/blue squares all in view. The backbone
    cannot supply this on its own: its language model is causal and the prompt is
    `<image><instruction>`, so the image block wholly precedes the instruction and no
    image token can attend to it. With this off, the head is provably language-blind
    (identical logits for "drill in the green target" and "drill in the blue target").

    Set to False only to reproduce a pre-existing checkpoint like perceptionv1, or as
    the ablation that shows the conditioning is what helps. Turning it on changes the
    head's parameter set, so a checkpoint trained one way cannot be loaded the other."""

    text_attn_heads: int = 8
    """Attention heads of the head's instruction cross-attention."""

    text_gate_init: float = 0.5
    """Initial value of the learned scalar gate on the instruction cross-attention.

    0 reproduces the unconditioned head exactly at init but leaves the branch inert
    (its gradient is scaled by tanh(0)=0), costing roughly 3000 steps at lr 1e-4
    before the gate opens far enough for the branch to learn.

    0.5 (tanh -> 0.46) is the default because the intended init is GR00T-N1.7-3B, which
    builds the WHOLE perception head fresh — there is no trained head stream to protect,
    so the only thing the gate buys is not swamping the backbone's pretrained layer-12
    features with a random projection. Lower it toward 0 only when warm-starting from an
    already-trained perception checkpoint, where the head's own weights are worth
    protecting from a randomly-initialised branch."""

    def __init__(self, **kwargs):
        # Defining __init__ here stops @dataclass from generating one; the parent
        # __init__ handles PretrainedConfig setup and fills dataclass-field
        # defaults (including this subclass's fields) via __dataclass_fields__.
        super().__init__(**kwargs)


register_model_config("Gr00tN1d7Perception", Gr00tN1d7PerceptionConfig)
