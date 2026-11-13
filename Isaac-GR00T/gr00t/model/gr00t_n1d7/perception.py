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

"""Perception-only GR00T model for the short-range fine approach.

VLM backbone (Cosmos-Reason2 / Qwen3-VL, same as Gr00tN1d7) + a target heatmap
head. Input: the CURRENT wrist-camera frame + task instruction. Output: where
the wall target is in wrist-camera pixel coordinates (heatmap over the VLM
image-token grid + subpixel offset), whether it is visible at all, and
optionally its depth and the wall normal in the camera frame.

There is no action head: this model does perception only. The coarse approach
keeps using the full VLA; this model takes over close to the wall, where the
heatmap_labels/ sidecar provides supervision on every recorded frame.
"""

import logging
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig

from .gr00t_n1d7 import get_backbone_cls


logger = logging.getLogger(__name__)


class TargetPerceptionHead(nn.Module):
    """Heatmap + visibility (+ depth / wall-normal) head over the VLM image tokens.

    Per image token (spatial grid cell after the ViT merge): a small MLP produces
    a heatmap logit and a subpixel (du, dv) offset in cell units. Globally: one
    learned query cross-attends over the image tokens and predicts a visibility
    logit, the target depth (scaled), and the wall normal.

    All coordinates inside the head are NORMALIZED to [0, 1] over the processed
    image, so the head is independent of the VLM's internal smart-resize.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        use_depth: bool,
        use_normal: bool,
        num_attn_heads: int = 8,
        use_text_conditioning: bool = False,
        text_attn_heads: int = 8,
        text_gate_init: float = 0.1,
    ):
        super().__init__()
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.use_text_conditioning = use_text_conditioning

        self.token_norm = nn.LayerNorm(latent_dim)

        # Language conditioning. Each image token cross-attends over the instruction
        # tokens before ANY output is produced, so both the per-token heatmap and the
        # pooled visibility/depth/normal branch become instruction-dependent.
        #
        # Needed because the backbone cannot supply this: its language model is causal
        # and the prompt is `<image><instruction>`, so the whole image block precedes
        # the instruction and no image token can attend to it. Without this the head is
        # provably language-blind, which is only harmless on a single-task dataset.
        #
        # The residual is scaled by a learned scalar gate (tanh(g)), so the model can
        # regulate how much the branch perturbs the feature stream (cf. ReZero /
        # Flamingo). text_gate_init trades warm-up cost against init safety:
        #
        #   g=0    exactly reproduces the unconditioned head at init, but the branch is
        #          then INERT — its gradient is scaled by tanh(g)=0, so only the gate
        #          learns at first. Measured: grad|gate|=4.6e-1 vs grad|cross_attn|=1.6e-30.
        #          Under Adam at lr 1e-4 the gate moves ~1e-4/step, so reaching a
        #          meaningful tanh(g)~0.3 burns ~3000 steps before the branch trains.
        #   g>0    both learn from step 1, at the cost of perturbing the features at init.
        #
        # The safety argument for g=0 only applies when warm-starting from a checkpoint
        # whose head is already trained: initialising from nvidia/GR00T-N1.7-3B builds the
        # WHOLE head fresh, so there is no pretrained head stream to protect and a larger
        # gate is strictly better. Hence Gr00tN1d7PerceptionConfig.text_gate_init = 0.5,
        # which is what actually reaches this argument in a normal run.
        self.text_gate_init = text_gate_init
        if use_text_conditioning:
            self.text_norm = nn.LayerNorm(latent_dim)
            self.text_cross_attn = nn.MultiheadAttention(
                latent_dim, text_attn_heads, batch_first=True
            )
            self.text_gate = nn.Parameter(torch.full((1,), float(text_gate_init)))
        # 3 outputs per token: heatmap logit, du, dv (cell units, centered).
        self.token_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.pool_query = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(latent_dim, num_attn_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(latent_dim)
        global_out = 1 + (1 if use_depth else 0) + (3 if use_normal else 0)
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, global_out),
        )

    def reset_parameters(self) -> None:
        """Re-initialize all parameters (needed after meta-device materialization,
        same caveat as TipTargetHead.reset_parameters)."""
        with torch.no_grad():
            self.pool_query.normal_(0.0, 0.02)
        self.pool_attn._reset_parameters()
        self.token_norm.reset_parameters()
        self.pool_norm.reset_parameters()
        for module in list(self.token_mlp) + list(self.global_mlp):
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        if self.use_text_conditioning:
            self.text_norm.reset_parameters()
            self.text_cross_attn._reset_parameters()
            with torch.no_grad():
                self.text_gate.fill_(float(self.text_gate_init))

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """image_tokens: (B, N, D) in raster order of the (gh, gw) token grid.

        text_tokens: (B, L, D) instruction tokens from text_token_sequence(), required
            when use_text_conditioning is set. text_padding_mask: (B, L), True = padding.
        """
        tokens = self.token_norm(image_tokens)

        if self.use_text_conditioning:
            assert text_tokens is not None, (
                "TargetPerceptionHead was built with use_text_conditioning=True but "
                "forward() got text_tokens=None — the head would silently fall back to "
                "being language-blind. Pass the output of text_token_sequence()."
            )
            text = self.text_norm(text_tokens.to(tokens.dtype))
            context, _ = self.text_cross_attn(
                tokens, text, text, key_padding_mask=text_padding_mask, need_weights=False
            )
            tokens = tokens + torch.tanh(self.text_gate).to(tokens.dtype) * context

        per_token = self.token_mlp(tokens)  # (B, N, 3)
        heat_logits = per_token[..., 0]
        offsets = per_token[..., 1:3]

        query = self.pool_query.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, tokens, tokens, need_weights=False)
        pooled = self.pool_norm(pooled).squeeze(1)
        global_out = self.global_mlp(pooled)

        out = {"heat_logits": heat_logits, "offsets": offsets, "vis_logit": global_out[..., 0]}
        idx = 1
        if self.use_depth:
            out["depth_scaled"] = global_out[..., idx]
            idx += 1
        if self.use_normal:
            out["wall_normal"] = global_out[..., idx : idx + 3]
        return out


def processed_image_size(image_target_size) -> Tuple[float, float]:
    """(W, H) of the processor's output image — the frame target_uv lives in."""
    if isinstance(image_target_size, (tuple, list)):
        return float(image_target_size[1]), float(image_target_size[0])  # stored as (H, W)
    return float(image_target_size), float(image_target_size)


def image_token_grid(
    vl_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int,
) -> Tuple[torch.Tensor, int, int]:
    """Gather image-token features into (B, gh*gw, D) plus the grid shape.

    Requires exactly one image per sample with identical grid shape across the
    batch (the perception setup: single current wrist frame at a fixed
    resolution).
    """
    B = vl_embeds.shape[0]
    assert image_grid_thw.shape[0] == B, (
        f"Perception head expects exactly 1 image per sample, got "
        f"{image_grid_thw.shape[0]} images for batch of {B}. Use a wrist-only "
        f"modality config with video delta_indices=[0]."
    )
    t, h, w = (int(x) for x in image_grid_thw[0])
    gh, gw = (h // merge_size), (w // merge_size)
    n_tokens = t * gh * gw
    counts = image_mask.sum(dim=1)
    assert bool((counts == n_tokens).all()), (
        f"Image token count mismatch: expected {n_tokens}, got {counts.tolist()}"
    )
    image_tokens = vl_embeds[image_mask].view(B, n_tokens, -1)
    return image_tokens, gh, gw


def text_token_sequence(
    vl_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather the NON-image tokens (instruction + chat-template scaffold) into a
    padded batch, for the head's text cross-attention.

    The backbone's language model is causal and the prompt is `<image><instruction>`,
    so image tokens never attend to the instruction — the instruction only reaches
    the head through this path. Lengths vary per sample (different instructions
    tokenize to different lengths), hence the padding mask.

    Returns:
        text_tokens: (B, L, D), left-aligned and zero-padded
        padding_mask: (B, L), True where the slot is padding — nn.MultiheadAttention's
            `key_padding_mask` convention (True = ignore this key)
    """
    B, T, D = vl_embeds.shape
    text_mask = attention_mask.bool() & ~image_mask.bool()
    lengths = text_mask.sum(dim=1)
    L = int(lengths.max().item())
    if L == 0:
        # No language at all in the batch: hand back a single zero key so the
        # attention is well-defined and contributes nothing.
        zeros = vl_embeds.new_zeros(B, 1, D)
        return zeros, torch.zeros(B, 1, dtype=torch.bool, device=vl_embeds.device)

    ranks = text_mask.cumsum(dim=1) - 1  # each selected token's index within its row
    rows = torch.arange(B, device=vl_embeds.device).unsqueeze(1).expand(B, T)
    text_tokens = vl_embeds.new_zeros(B, L, D)
    padding_mask = torch.ones(B, L, dtype=torch.bool, device=vl_embeds.device)
    text_tokens[rows[text_mask], ranks[text_mask]] = vl_embeds[text_mask]
    padding_mask[rows[text_mask], ranks[text_mask]] = False

    # A row with zero text tokens would be fully masked, and a softmax over an empty
    # key set yields NaN. Unmask its (zero-valued) first slot instead: attention then
    # returns a zero context, making the gated residual a no-op for that sample.
    empty = lengths == 0
    if bool(empty.any()):
        padding_mask[empty, 0] = False
    return text_tokens, padding_mask


def decode_uv(
    heat_logits: torch.Tensor,
    offsets: torch.Tensor,
    gh: int,
    gw: int,
    proc_w: float,
    proc_h: float,
) -> torch.Tensor:
    """Argmax cell + its subpixel offset -> uv in processed-image px, (B, 2)."""
    B = heat_logits.shape[0]
    best = heat_logits.argmax(dim=-1)  # (B,)
    off = offsets[torch.arange(B, device=heat_logits.device), best].clamp(-0.5, 0.5)
    cx = (best % gw).float() + 0.5 + off[:, 0]
    cy = (best // gw).float() + 0.5 + off[:, 1]
    return torch.stack((cx / gw * proc_w, cy / gh * proc_h), dim=-1)


def perception_losses(
    config: Gr00tN1d7PerceptionConfig,
    head_out: dict[str, torch.Tensor],
    inputs: dict,
    gh: int,
    gw: int,
    use_depth: bool,
    use_normal: bool,
    pred_uv: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """All perception losses + monitoring metrics for one batch.

    Shared by the perception-only model (where the weighted sum IS the training
    loss) and the perception-action model (where it is added to the action loss).
    Returns the individual terms, the weighted sum as ``perception_loss``, and
    the no-grad quality metrics ``uv_err_px_metric`` / ``visibility_acc_metric``.

    All masked terms are computed batch-wide and mask-weighted rather than by
    indexing valid rows, so every head parameter receives a gradient on every
    step and DDP never sees an unused parameter.
    """
    heat_logits = head_out["heat_logits"].float()  # (B, N)
    offsets = head_out["offsets"].float()  # (B, N, 2)
    vis_logit = head_out["vis_logit"].float()  # (B,)
    B, N = heat_logits.shape
    device = heat_logits.device

    proc_w, proc_h = processed_image_size(config.image_target_size)

    # Labels (from the perception processor): uv in processed-image px.
    target_uv = inputs["target_uv"].float().view(B, 2)
    target_visible = inputs["target_visible"].float().view(B)
    heat_valid = target_visible * (~torch.isnan(target_uv).any(dim=-1)).float()

    # Normalized target coords in [0,1] (NaN-safe; invalid rows are masked out).
    uv = torch.nan_to_num(target_uv)
    u_norm = (uv[:, 0] / proc_w).clamp(0.0, 1.0 - 1e-6)
    v_norm = (uv[:, 1] / proc_h).clamp(0.0, 1.0 - 1e-6)
    gx = u_norm * gw  # continuous grid coords, cell units
    gy = v_norm * gh

    # --- Heatmap CE against a normalized Gaussian rendered on the cell grid ---
    sigma_x = config.heatmap_target_sigma_px / proc_w * gw
    sigma_y = config.heatmap_target_sigma_px / proc_h * gh
    xs = torch.arange(gw, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(gh, device=device, dtype=torch.float32) + 0.5
    dx2 = (xs.view(1, 1, gw) - gx.view(B, 1, 1)) ** 2 / (2 * sigma_x**2)
    dy2 = (ys.view(1, gh, 1) - gy.view(B, 1, 1)) ** 2 / (2 * sigma_y**2)
    target_heat = torch.exp(-(dx2 + dy2)).view(B, N)
    target_heat = target_heat / target_heat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probs = F.log_softmax(heat_logits, dim=-1)
    heat_ce = -(target_heat * log_probs).sum(dim=-1)  # (B,)
    heatmap_loss = (heat_ce * heat_valid).sum() / heat_valid.sum().clamp_min(1.0)

    # --- Subpixel offset L1 at the ground-truth cell ---
    cell_x = gx.long().clamp(max=gw - 1)
    cell_y = gy.long().clamp(max=gh - 1)
    cell_idx = cell_y * gw + cell_x  # (B,)
    pred_off = offsets[torch.arange(B, device=device), cell_idx]  # (B, 2)
    target_off = torch.stack((gx - cell_x.float() - 0.5, gy - cell_y.float() - 0.5), dim=-1)
    offset_l1 = F.smooth_l1_loss(pred_off, target_off, reduction="none").sum(dim=-1)
    offset_loss = (offset_l1 * heat_valid).sum() / heat_valid.sum().clamp_min(1.0)

    # --- Visibility BCE over all rows, optionally class-weighted ---
    # Weight-normalized rather than mean-reduced so visibility_loss_weight keeps its
    # meaning when visibility_neg_weight changes; at neg_weight=1 this is the mean.
    vis_w = 1.0 + (config.visibility_neg_weight - 1.0) * (1.0 - target_visible)
    visibility_loss = (
        F.binary_cross_entropy_with_logits(vis_logit, target_visible, reduction="none") * vis_w
    ).sum() / vis_w.sum().clamp_min(1e-6)

    loss = (
        config.heatmap_loss_weight * heatmap_loss
        + config.offset_loss_weight * offset_loss
        + config.visibility_loss_weight * visibility_loss
    )

    outputs = {
        "heatmap_loss": heatmap_loss,
        "offset_loss": offset_loss,
        "visibility_loss": visibility_loss,
    }

    # --- Optional depth ---
    if use_depth and "target_depth" in inputs:
        depth_target = torch.nan_to_num(inputs["target_depth"].float().view(B) / config.depth_scale)
        depth_valid = heat_valid * (~torch.isnan(inputs["target_depth"].view(B))).float()
        depth_err = F.smooth_l1_loss(
            head_out["depth_scaled"].float(), depth_target, reduction="none"
        )
        depth_loss = (depth_err * depth_valid).sum() / depth_valid.sum().clamp_min(1.0)
        loss = loss + config.depth_loss_weight * depth_loss
        outputs["depth_loss"] = depth_loss

    # --- Optional wall normal ---
    if use_normal and "target_wall_normal" in inputs:
        normal_target = inputs["target_wall_normal"].float().view(B, 3)
        normal_valid = heat_valid * (~torch.isnan(normal_target).any(dim=-1)).float()
        normal_err = F.mse_loss(
            head_out["wall_normal"].float(), torch.nan_to_num(normal_target), reduction="none"
        ).sum(dim=-1)
        normal_loss = (normal_err * normal_valid).sum() / normal_valid.sum().clamp_min(1.0)
        loss = loss + config.wall_normal_loss_weight * normal_loss
        outputs["normal_loss"] = normal_loss

    outputs["perception_loss"] = loss

    # --- Monitoring metrics (no grad) ---
    with torch.no_grad():
        if pred_uv is None:
            pred_uv = decode_uv(heat_logits, offsets, gh, gw, proc_w, proc_h)
        err_px = torch.linalg.vector_norm(pred_uv.float() - uv, dim=-1)
        n_vis = heat_valid.sum().clamp_min(1.0)
        outputs["uv_err_px_metric"] = (err_px * heat_valid).sum() / n_vis
        vis_pred = (torch.sigmoid(vis_logit) > 0.5).float()
        correct = (vis_pred == target_visible).float()

        # Raw accuracy is inflated to uselessness by class imbalance: only ~3.5% of
        # post-augmentation frames have target_visible=0, so a batch of 16 carries 0.55
        # negatives on average and 57% of batches carry NONE. In such a batch every label
        # is "visible", so a model that always answers "visible" scores exactly 1.0 while
        # never being tested on the only question that matters. Kept for continuity, but
        # read visibility_tnr_metric instead.
        outputs["visibility_acc_metric"] = correct.mean()

        n_pos = target_visible.sum()
        n_neg = (1.0 - target_visible).sum()
        nan = torch.tensor(float("nan"), device=correct.device, dtype=correct.dtype)
        # NaN (not 0 or 1) when the batch has no example of a class — the metric is
        # genuinely undefined there, and a fake value is what made the old one
        # unreadable. wandb renders NaN as a gap.
        tpr = torch.where(n_pos > 0, (correct * target_visible).sum() / n_pos.clamp_min(1.0), nan)
        tnr = torch.where(
            n_neg > 0, (correct * (1.0 - target_visible)).sum() / n_neg.clamp_min(1.0), nan
        )
        # Recall on the negatives: the one number that exposes a collapse to
        # always-visible, since it is 0 exactly when the model never says "not visible".
        outputs["visibility_tnr_metric"] = tnr
        outputs["visibility_tpr_metric"] = tpr
        outputs["visibility_balanced_acc_metric"] = 0.5 * (tpr + tnr)
        # How often the batch could test the negative class at all — if this sits near 0,
        # the visibility metrics are noise and the batch size is too small to judge them.
        outputs["visibility_neg_frac_metric"] = n_neg / correct.numel()

    return outputs


class Gr00tN1d7Perception(PreTrainedModel):
    """VLM-only perception model: Qwen3-VL backbone + TargetPerceptionHead."""

    config_class = Gr00tN1d7PerceptionConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7PerceptionConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        self.perception_head = TargetPerceptionHead(
            latent_dim=config.backbone_embedding_dim,
            hidden_dim=config.heatmap_hidden_dim,
            use_depth=config.depth_loss_weight > 0,
            use_normal=config.wall_normal_loss_weight > 0,
            num_attn_heads=config.perception_pool_heads,
            use_text_conditioning=config.use_text_conditioning,
            text_attn_heads=config.text_attn_heads,
            text_gate_init=config.text_gate_init,
        )

        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        vision_cfg = self.backbone.model.config.vision_config
        self._merge_size = getattr(vision_cfg, "spatial_merge_size", 2)

    # ------------------------------------------------------------------
    # Input preparation (mirrors Gr00tN1d7.prepare_input)
    # ------------------------------------------------------------------
    def prepare_input(self, inputs: dict) -> BatchFeature:
        if "vlm_content" in inputs:
            vlm_content_list = inputs["vlm_content"]
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        def to_device_with_dtype(x):
            if not torch.is_tensor(x):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        return tree.map_structure(to_device_with_dtype, dict(inputs))

    # ------------------------------------------------------------------
    # Token grid helpers
    # ------------------------------------------------------------------
    def _image_token_grid(
        self, backbone_output: BatchFeature, image_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Gather image-token features into (B, gh*gw, D) plus the grid shape."""
        return image_token_grid(
            backbone_output["backbone_features"],
            backbone_output["image_mask"],
            image_grid_thw,
            self._merge_size,
        )

    def _head_inputs(self, backbone_output: BatchFeature, image_grid_thw: torch.Tensor):
        """(image_tokens, gh, gw, text_tokens, text_padding_mask) for the head.

        text_* are None unless the head was built with text conditioning, so the
        unconditioned path stays allocation-free.
        """
        image_tokens, gh, gw = self._image_token_grid(backbone_output, image_grid_thw)
        if not self.perception_head.use_text_conditioning:
            return image_tokens, gh, gw, None, None
        text_tokens, text_padding_mask = text_token_sequence(
            backbone_output["backbone_features"],
            backbone_output["image_mask"],
            backbone_output["backbone_attention_mask"],
        )
        return image_tokens, gh, gw, text_tokens, text_padding_mask

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(self, inputs: dict) -> BatchFeature:
        inputs = self.prepare_input(inputs)
        backbone_output = self.backbone(BatchFeature(data=inputs))
        image_tokens, gh, gw, text_tokens, text_pad = self._head_inputs(
            backbone_output, inputs["image_grid_thw"]
        )
        head_out = self.perception_head(image_tokens, text_tokens, text_pad)

        outputs = perception_losses(
            self.config,
            head_out,
            inputs,
            gh,
            gw,
            use_depth=self.perception_head.use_depth,
            use_normal=self.perception_head.use_normal,
        )
        # For this model the perception loss IS the training loss; drop the alias
        # so wandb does not log the same number under two names.
        outputs["loss"] = outputs.pop("perception_loss")

        if self.perception_head.use_text_conditioning:
            # tanh(gate), i.e. the ACTUAL multiplier on the language branch — not the
            # raw parameter. Flat near its init means the head is not using the
            # instruction, which is visible within a few hundred steps instead of
            # having to be inferred from a bad final uv_err_px_metric.
            # The trainer auto-logs any 0-dim output whose name ends in _metric.
            with torch.no_grad():
                outputs["text_gate_metric"] = torch.tanh(
                    self.perception_head.text_gate.detach()
                ).squeeze()

        # Plain dict (not BatchFeature): HF Trainer's compute_loss requires a dict
        # subclass to index outputs["loss"].
        return outputs

    def _processed_image_size(self) -> Tuple[float, float]:
        """(W, H) of the processor's output image — the frame target_uv lives in."""
        return processed_image_size(self.config.image_target_size)

    def _decode_uv(
        self, heat_logits: torch.Tensor, offsets: torch.Tensor, gh: int, gw: int
    ) -> torch.Tensor:
        """Argmax cell + its subpixel offset -> uv in processed-image px, (B, 2)."""
        proc_w, proc_h = self._processed_image_size()
        return decode_uv(heat_logits, offsets, gh, gw, proc_w, proc_h)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, inputs: dict) -> BatchFeature:
        """Run perception on processed inputs (from the perception processor).

        Returns:
            target_uv: (B, 2) predicted target center, processed-image px
                       (use uv_processed_to_camera to map back to raw wrist px)
            visibility: (B,) sigmoid probability the target is visible
            heatmap: (B, gh, gw) softmax heatmap over the token grid
            depth: (B,) target depth in metres (if the depth head is enabled)
            wall_normal: (B, 3) wall normal in camera frame (if enabled)
        """
        inputs = self.prepare_input(inputs)
        backbone_output = self.backbone(BatchFeature(data=inputs))
        image_tokens, gh, gw, text_tokens, text_pad = self._head_inputs(
            backbone_output, inputs["image_grid_thw"]
        )
        head_out = self.perception_head(image_tokens, text_tokens, text_pad)

        heat_logits = head_out["heat_logits"].float()
        offsets = head_out["offsets"].float()
        result = {
            "target_uv": self._decode_uv(heat_logits, offsets, gh, gw),
            "visibility": torch.sigmoid(head_out["vis_logit"].float()),
            "heatmap": F.softmax(heat_logits, dim=-1).view(-1, gh, gw),
        }
        if self.perception_head.use_depth:
            result["depth"] = head_out["depth_scaled"].float() * self.config.depth_scale
        if self.perception_head.use_normal:
            result["wall_normal"] = F.normalize(head_out["wall_normal"].float(), dim=-1)
        return BatchFeature(data=result)

    def uv_processed_to_camera(
        self, uv: torch.Tensor, camera_size: Tuple[int, int] = (720, 720)
    ) -> torch.Tensor:
        """Map uv from processed-image px back to RAW wrist-camera px.

        Inverts the deterministic eval transform (letterbox -> resize shortest
        edge to image_target_size -> center crop crop_fraction -> resize back).
        Assumes a square camera image (no letterbox padding).
        """
        cam_w, cam_h = float(camera_size[0]), float(camera_size[1])
        proc_w, proc_h = self._processed_image_size()
        crop = self.config.image_crop_size
        crop_h, crop_w = crop if isinstance(crop, (tuple, list)) else (crop, crop)
        # Forward: resize cam -> (proc_h, proc_w); center-crop to (crop_h, crop_w)
        # using integer offsets; resize back to (proc_h, proc_w).
        x_min = (proc_w - crop_w) // 2
        y_min = (proc_h - crop_h) // 2
        u = (uv[..., 0] * crop_w / proc_w + x_min) * cam_w / proc_w
        v = (uv[..., 1] * crop_h / proc_h + y_min) * cam_h / proc_h
        return torch.stack((u, v), dim=-1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


AutoConfig.register("Gr00tN1d7Perception", Gr00tN1d7PerceptionConfig)
AutoModel.register(Gr00tN1d7PerceptionConfig, Gr00tN1d7Perception)
