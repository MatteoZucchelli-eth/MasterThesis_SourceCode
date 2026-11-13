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

# Finetune config used for single node post-training.
from dataclasses import dataclass


GR00T_VLM_CHECKPOINT = "nvidia/GR00T-N1.7-3B"
"""The GR00T N1.7 release whose VLM backbone the perception finetune starts from.

Its backbone is a robot-finetuned nvidia/Cosmos-Reason2-2B, which is also the
architecture every model here is constructed from; only the weights differ. The
~1B action-head tensors in this checkpoint are simply not loaded by the perception
pipeline."""


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    use_tip_target_head: bool = False
    """If True, add an auxiliary head that predicts the target pose in the end-effector (tip) frame
    as xyz(3) + rot6d(6) from the VLM latent (target = the `next.tip_target_9d` column in the
    parquet; NaN rows are masked out). The distance to target is derived as ||xyz||. Useful for
    precise end-effector placement: the action head is conditioned on where the target is relative
    to the tip."""

    tip_target_loss_weight: float = 0.5
    """Overall multiplier on the tip-target loss (pos + rot terms) when added to the action loss
    (only used if use_tip_target_head)."""

    tip_target_lat_loss_weight: float = 2.0 / 3.0
    """Weight of the LATERAL position term -- the two components of the position error
    perpendicular to the wrist camera's optical axis, i.e. the directions that show up
    directly as pixel displacement.

    The position error is rotated out of the tip frame into the CAMERA frame before it is
    split, because the label's own axes do not line up with the observable ones. Measured
    by Kabsch on 44.7k (sidecar u/v/depth_z, next.tip_target_9d) pairs, rigid fit rms
    1.18 mm (the wrist camera and the drill tip share a link, so the transform is a
    constant): the optical axis in the tip frame is [0.9037, -0.0012, -0.4281]. Tip-frame
    `z` is therefore NOT depth -- depth is 0.9037*x - 0.4281*z, and a naive x,y-vs-z split
    leaves 90% of the depth direction inside the "lateral" term.

    Lateral error is cheap to observe: 1 processed px is 0.381 mm at the median working
    depth of 148 mm. See tip_target_depth_loss_weight for the other half.

    NOTE the defaults 2/3 and 1/3 reproduce the previous single xyz MSE EXACTLY -- a
    rotation preserves the L2 norm, so an equally-weighted split is the same number. Change
    them only as a deliberate experiment, one at a time."""

    tip_target_depth_loss_weight: float = 1.0 / 3.0
    """Weight of the DEPTH position term -- the component of the position error along the
    wrist camera's optical axis. This is the hard direction: it cannot be read off pixel
    positions and has to be inferred from apparent size and perspective. Measured over the
    heatmap sidecar (44.7k visible frames), the target sits at 148.0 +/- 30.0 mm of camera
    depth, spanning 53-225 mm.

    Raise it to spend more capacity on depth; lower it to stop an under-determined depth
    signal from polluting the lateral estimate. See tip_target_lat_loss_weight for the
    default that reproduces the old loss."""

    tip_target_rot_loss_weight: float = 1.0
    """Weight of the rotation term (plain MSE on the raw rot6d part) relative to the
    combined position term. 1.0 weights position and rotation equally, which is what the
    previous "mse" objective did."""

    tip_target_wrist_mask_prob: float = 0.0
    """Probability of hiding the WRIST image tokens from the tip-target head's pooling on a
    given training sample, so the head has to recover the target from the ego (head) camera
    alone and can learn to fuse the two views rather than leaning entirely on the wrist.

    This masks the head's key/value set only -- the action head still sees every token, so
    it cannot destabilize policy training. The masking is meaningful rather than cosmetic
    because the backbone is causal and the views are concatenated ego-first (both
    register_new_modality.py and register_autocollect_modality.py list
    ["ego_view", "wrist_view"]), so ego tokens can never attend to wrist tokens: their
    representations are genuinely wrist-free. The split is read off the image-token run
    structure, so it is correct for single-frame [0] and for [-1, 0] alike.

    0.0 (default) disables it. Applied in training only; inference always sees both views.

    CAVEAT before trusting a result: nothing supervises the ego view directly (the
    heatmap_labels/ sidecar is wrist-only), the ego camera is natively 480px against the
    wrist's 720, and image_target_size is a single global value, so above 480 the ego view
    is being upsampled. Check that the target is resolvable in ego at all first."""

    tip_target_pool: str = "attn_residual"
    """How the vision-language token sequence is collapsed into the vector fed to the head's
    predictor MLP.

      "attn_residual" (default)  `tip_target_num_queries` learned query tokens cross-attend
                                 over the whole sequence, so the head can focus on the patches
                                 containing the target and on the text tokens that name it --
                                 PLUS the masked-mean vector added back to every pooled query,
                                 with the attention's output projection zero-initialized.
      "mean"                     plain masked mean over all valid tokens.

    The residual matters for exactly the reason you would expect. Without it the pool is a
    single cross-attention layer with no skip connection and no FFN whose queries start as
    random vectors, so the softmax begins near-uniform over the whole sequence: training
    starts at "a mean pool seen through an untrained 16.8M-parameter random projection" (at
    latent_dim 2048, against ~1.3M total for "mean") and has to climb out, while the gradient
    reaching the backbone is routed by whatever those random queries happened to attend to.
    With the residual the head starts AT the mean-pool solution, the mean path carries
    gradient to every token from step 0, and the attention only learns a correction. Same
    function class, better conditioned -- which is why the bare "attn" variant was removed
    rather than kept as an option.

    A checkpoint trained with the old bare "attn" still LOADS (identical parameter shapes) but
    will NOT reproduce, because the mean vector is now added in the forward pass.

    "mean" is not weight-compatible with "attn_residual" (predictor input width
    `num_queries * latent_dim` vs `latent_dim`), so give it its own run name."""

    tip_target_num_queries: int = 4
    """Number of learned query tokens for tip_target_pool="attn_residual" (inert under
    "mean")."""

    tip_target_attn_heads: int = 8
    """Attention heads of the pooling cross-attention for tip_target_pool="attn_residual"
    (inert under "mean")."""

    use_tip_target_conditioning: bool = True
    """If True, feed the tip-target 9d back as a conditioning token to the action head.
    Teacher-forced: training conditions on the ground truth (+ small noise), inference on the
    head's prediction. See tip_target_tf_gate_mm to hand that over to the head's own
    prediction once it is good enough."""

    tip_target_tf_gate_mm: float | None = None
    """Scheduled sampling: once the head is accurate enough, start conditioning the action
    head on its OWN prediction instead of the label, so the action head stops training
    against a conditioning token it will never see at deployment.

    Set it to the mean position error (MILLIMETRES) the head must reach before the handover
    begins. None (default) keeps teacher forcing on for the whole run, the previous
    behaviour.

    The gate watches an EMA of the head's position error, requires it to hold under the
    threshold for `tip_target_tf_gate_patience` steps, and is STICKY -- once open it never
    closes, so a noisy batch cannot flip the conditioning distribution back and forth.
    Crossing it does not switch anything abruptly: the per-sample probability of using the
    prediction ramps 0 -> tip_target_tf_max_pred_prob over tip_target_tf_ramp_steps.

    The decision is all-reduced across ranks (it must be: if two GPUs disagreed about whether
    the gate is open they would feed different conditioning tokens and their gradients would
    diverge) and is saved in the checkpoint, so resubmitting on Euler resumes the schedule
    instead of restarting it."""

    tip_target_tf_ramp_steps: int = 2000
    """Steps over which the probability of conditioning on the prediction ramps from 0 to
    tip_target_tf_max_pred_prob after the gate opens. Inert when tip_target_tf_gate_mm is
    None."""

    tip_target_tf_max_pred_prob: float = 1.0
    """Probability of conditioning on the head's own prediction once the ramp completes. 1.0
    hands over entirely; values below 1.0 keep a fraction of teacher-forced samples as a
    stabilizer. Inert when tip_target_tf_gate_mm is None."""

    tip_target_tf_gate_patience: int = 200
    """Consecutive steps the EMA'd position error must stay under tip_target_tf_gate_mm
    before the gate opens. Inert when tip_target_tf_gate_mm is None."""

    tip_target_tf_gate_ema: float = 0.99
    """Decay of the EMA over the head's position error that the gate watches. Inert when
    tip_target_tf_gate_mm is None."""

    tip_target_xyz_scale: float = 0.1
    """Scale (metres) dividing the xyz part of the tip-target label inside the head so position and
    rot6d errors contribute comparable gradients. Stored in the model config so inference matches."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Image Resolution ---
    image_target_size: int = 256
    """Square edge length (px) the image is resized to before it is fed to the VLM
    (model config `image_target_size`). Larger = more detail preserved for small
    targets, at the cost of more visual tokens / GPU memory. The Cosmos-Reason2 (Qwen3-VL)
    backbone uses native dynamic resolution, so raising this is safe. Must be > image_crop_size."""

    image_crop_size: int = 230
    """Square edge length (px) of the crop taken during the resize pipeline (model config
    `image_crop_size`). Kept slightly below image_target_size (default ratio ~0.9) to allow
    the random/center crop augmentation. Set together with image_target_size."""

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of optimizer steps to run before stopping. Ignored when num_epochs is set."""

    num_epochs: float | None = None
    """If set, derive max_steps from the dataset size instead of specifying it directly:
    max_steps = ceil(num_epochs * trainable_steps / (global_batch_size * gradient_accumulation_steps)),
    where trainable_steps counts the timesteps that can start a full action chunk
    (sum over episodes of length - action_horizon + 1, read from meta/episodes.jsonl).
    Fractional values are allowed (0.5 = half a pass).

    CAVEAT: this is a sample-count target, not a true pass over the data. The training
    iterator draws batches from a shard buffer with replacement, so after "1 epoch" some
    frames have been seen several times and others not at all -- read it as "each frame
    seen num_epochs times in expectation". Note also that warmup_ratio is a fraction of
    max_steps and so scales automatically, but save_steps is absolute and does not."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""


@dataclass
class PerceptionFinetuneConfig:
    """Fine-tune the perception-only model (GR00T VLM backbone + target heatmap head).

    The model sees ONLY the current wrist-camera frame + instruction and predicts the
    wall-target location as a heatmap in wrist-camera pixels, a visibility flag, and
    optionally target depth / wall normal. Training samples every frame represented in
    the heatmap_labels/ sidecar next to data/.
    """

    base_model_path: str = GR00T_VLM_CHECKPOINT
    """Checkpoint supplying the VLM backbone weights. Must be GR00T-lineage: either
    `nvidia/GR00T-N1.7-3B` itself (the default), one of your finetuned VLA runs
    (action-head weights are ignored, only the backbone is taken), or a previous
    perception checkpoint.

    There is deliberately NO "start from the bare Cosmos-Reason2-2B foundation" option.
    The backbone ARCHITECTURE is always Cosmos-Reason2-2B -- GR00T N1.7's VLM is that
    same 2B model, robot-finetuned -- so the only thing this choice ever controlled was
    whether GR00T's weights got overlaid on top or not, and not overlaying them throws
    away the robot adaptation for nothing. Passing an empty value is an error rather
    than a silent fall-back to the raw foundation weights."""

    dataset_path: str = "./g1_dataset"
    """Path to the dataset root (must contain the heatmap_labels/ sidecar directory)."""

    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag (name or value, case-insensitive)."""

    modality_config_path: str | None = None
    """Path to a Python file registering the WRIST-ONLY modality config
    (see register_perception_modality.py)."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the (truncated) language model of the backbone."""

    tune_visual: bool = True
    """If True, fine-tune the vision encoder. Recommended: the heatmap precision
    comes almost entirely from the visual features."""

    tune_top_llm_layers: int = 0
    """Number of top LLM layers to tune (0 = none). Cheaper alternative to tune_llm
    for adapting the fused vision-language features."""

    # --- Perception Head / Loss ---
    heatmap_hidden_dim: int = 512
    """Hidden width of the perception head MLPs."""

    heatmap_target_sigma_px: float = 8.0
    """Std (processed-image px) of the Gaussian target rendered around the label."""

    heatmap_loss_weight: float = 1.0
    """Weight of the heatmap cross-entropy (visible frames only)."""

    offset_loss_weight: float = 1.0
    """Weight of the subpixel offset loss at the ground-truth cell."""

    visibility_loss_weight: float = 0.5
    """Weight of the visibility BCE (all frames)."""

    visibility_neg_weight: float = 1.0
    """Per-sample weight on target_visible=0 rows in the visibility BCE. 1.0 is the
    unweighted loss. If class balancing is desired, derive n_pos/n_neg from the
    generated sidecar; the ratio is dataset-dependent now that every frame is used."""

    depth_loss_weight: float = 0.1
    """Weight of the target-depth loss (0 disables the depth output)."""

    wall_normal_loss_weight: float = 0.1
    """Weight of the wall-normal loss (0 disables the wall-normal output)."""

    # --- Language conditioning ---
    use_text_conditioning: bool = True
    """Let the perception head cross-attend over the instruction tokens.

    Mandatory on multi-target datasets (several candidate squares in frame, the
    instruction picks one): the causal backbone puts the whole image block before the
    instruction, so without this the head cannot see the instruction at all. Pass
    --no-use-text-conditioning to reproduce the old language-blind behaviour."""

    # --- Image Resolution ---
    image_target_size: int = 720
    """Square edge (px) the wrist image is resized to before the VLM. Native wrist
    resolution is 720; keep it for maximum px precision at short range."""

    image_crop_size: int = 648
    """Square edge (px) of the random/center crop (must be < image_target_size)."""

    # --- Data Augmentation ---
    color_jitter_params: dict[str, float] | None = None
    """Color jitter parameters ({"brightness","contrast","saturation","hue"}).
    Keep hue/saturation LOW: the target is red-semantic. NOTE: never enable
    rotation augmentation here without care — the keypoint follows the transform,
    but the wall-normal label does not."""

    # --- Training Configuration ---
    global_batch_size: int = 32
    dataloader_num_workers: int = 2
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    output_dir: str = "./outputs-perception"
    experiment_name: str | None = None
    wandb_project: str = "finetune-gr00t-n1d7-perception"
    save_steps: int = 1000
    save_total_limit: int = 5
    num_gpus: int = 1
    use_wandb: bool = False
    max_steps: int = 10000
    """Total optimizer steps to run. Ignored when num_epochs is set."""

    num_epochs: float | None = None
    """If set, derive max_steps from the dataset size instead of specifying it directly.
    Same meaning and same caveat as FinetuneConfig.num_epochs, with one difference that
    matters: these finetunes run with perception_labels_only=True, so an "epoch" is
    counted over every row in the heatmap_labels/ sidecar, not over episodes without
    labels. Datasets without that sidecar cannot use num_epochs at all."""

    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)
    save_only_model: bool = False
    skip_weight_loading: bool = False
    """If True, skip loading weights from base_model_path (smoke tests)."""
