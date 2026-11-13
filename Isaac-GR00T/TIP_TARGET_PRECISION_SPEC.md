 # Tip-Target Precision — Implementation Spec

Handoff spec for improving target-prediction precision of the custom tip-target head in this
repo (Isaac GR00T N1.7, Unitree G1 drilling task). Written to be implemented by someone with
no prior context. Read section 0–2 before touching code.

---

## 0. Context

- The right end-effector is a **drill tip** that must be placed on a small target square.
- A custom auxiliary **tip-target head** predicts the target pose in the drill-tip frame
  (`xyz(3) + rot6d(6)`, the `next.tip_target_9d` label) from the pooled vision-language latent,
  and feeds that prediction back as **one conditioning token** to the action diffusion model (DiT).
  - Head + forward + loss: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
    (class `TipTargetHead` ~L107; `Gr00tN1d7ActionHead.forward` head/loss ~L429–521).
  - Config: `gr00t/configs/model/gr00t_n1d7.py` (~L154–160), `gr00t/configs/finetune_config.py` (~L73–92).
  - Enabled from `start_finetune.sh` via `--use-tip-target-head`.
  - Label pass-through: `gr00t/data/dataset/lerobot_episode_loader.py` (~L405),
    `gr00t/data/dataset/sharded_single_step_dataset.py` (~L86),
    `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` (~L617).
  - Prediction surfaced for deployment: `gr00t/policy/gr00t_policy.py` (~L481).
- Training data: `g1_dataset_small_synthetic_1500` (synthetic). Eval runs in Isaac Sim.

## 1. Problem & root cause

- Goal: the drill tip should land on the target more accurately. The dominant error is the
  **head's target prediction**, not the low-level controller (tracking is already ~2–12 mm).
- Observed: when the target is **visible**, the head is ~1–2 cm off; when the target is **not
  visible** in the camera, the head outputs garbage (0.3–0.5 m). The garbage cases dominate the
  average and are the failures worth fixing.
- **Root cause:** the training set labels the target on ~100% of frames (0% occlusion) — the
  simulator knows the target pose even when it is off-camera — so the head learned to *always*
  emit a confident target, including on frames whose image shows nothing. A single-frame
  feed-forward head **cannot** localize an off-screen target. The fix must (a) let the model know
  when it can't see the target, and (b) remember where the target was.
- Secondary issues: eval `finalDist` is polluted by **contact-force spikes in Isaac** (use
  min / pre-contact distance instead); **action-horizon mismatch** (checkpoints trained at 40,
  `register_new_modality.py` now uses 30).

## 2. Prerequisites that are NOT code in this repo (owner: user / sim / deploy harness)

These gate how much the code changes below help. Note them; do not attempt them here.

- **P1 — sim data (enables Task A):** regenerate `g1_dataset_small_synthetic_1500` with occlusion
  handling. For each frame, project the target into each camera; on frames where it is outside
  **every** camera FOV, either write `next.tip_target_9d = NaN` or add a boolean column
  `next.tip_target_visible = False`. Without occluded frames, the visibility head has nothing to learn.
- **P2 — deploy/eval harness (external):** base-frame target memory. Transform each *visible*-frame
  head prediction into the world/base frame using the current tip pose (FK from proprioception),
  fuse over time (running median / Kalman), and hold it through occlusion, gated by the visibility
  signal the model exposes. This is likely what the existing "VLM planner" does (it reaches ~7 mm).
- **P3 — eval metric:** report min / pre-contact distance, not `finalDist`.

## 3. In-repo tasks (do these)

### Task A — Visibility head + conditioning gate  [highest priority]
Stop the policy from acting on a hallucinated target when the target is not visible.
- In `TipTargetHead`, add a **visibility logit** output (small linear head off the same pooled
  feature); return `(pred_scaled_9d, vis_logit)`.
- **Visibility target:** `visible = ~isnan(tip_target_9d).any(-1)` (this reuses the NaN mask already
  computed at `forward` ~L444; once P1 adds occluded/NaN frames it trains correctly). Optionally also
  consume an explicit `next.tip_target_visible` column if present (wire it through the loader +
  processor exactly like `tip_target_9d`).
- **Loss:** add `BCEWithLogits(vis_logit, visible)` over **all** rows (weight ≈ 0.2). Keep the 9d
  regression loss over **visible rows only** (already masked).
- **Conditioning gate:** where the conditioning token is built (~L451), when the frame is predicted
  invisible (`sigmoid(vis_logit) < τ` at inference; `visible == False` in training) feed a **learned
  `nn.Parameter` "unknown" token** instead of the pose token, so the DiT learns a distinct
  "target unknown" mode rather than trusting a garbage pose.
- Surface `vis_logit`/`sigmoid` next to `tip_target_pred` in `gr00t/policy/gr00t_policy.py` for P2.
- Add config flags in `gr00t/configs/model/gr00t_n1d7.py` + `finetune_config.py`
  (`use_tip_target_visibility`, `tip_target_visibility_loss_weight`, threshold).
- **Accept:** post-P1, visibility AUC > 0.9 on held-out episodes; on invisible frames the
  conditioning token is the "unknown" embedding, not a pose.

### Task B — Attention pooling over image tokens only  [high impact when visible]
Sharpen the visible-case prediction; the current pooling throws away spatial location.
- Replace `_masked_mean_pool(vl_embeds, attention_mask)` in `TipTargetHead.predict_scaled` (~L150)
  with a **learned-query cross-attention pool** (`nn.MultiheadAttention`, 1–4 learned query tokens)
  over the VLM tokens.
- Pool over **image tokens only**: pass `backbone_output.image_mask` (already available; used by the
  DiT at `forward` ~L459) into the head and exclude text/padding tokens. Currently the head pools
  *all* tokens including the language prompt.
- If feasible, pool **per camera** (split image tokens by view) so the head can use whichever view
  frames the target.
- Keep the 9d output shape and loss unchanged.
- **Accept:** visible-frame `‖xyz_pred − xyz_gt‖` drops vs the mean-pool baseline on a fixed val set.
- **Status:** the attention pool exists (`tip_target_pool=attn`) but keeps **losing to `mean`** in
  practice. Two measured reasons it can, which point at different fixes:
  - It is ~21.3M fresh randomly-initialized parameters against `mean`'s ~1.3M — 16.8M of that is
    the `nn.MultiheadAttention` alone at `latent_dim=2048`. That is a lot of weight to learn from
    scratch on a finetune budget, and the pool has no residual path, so training *starts* at
    "a mean pool seen through an untrained 16.8M-parameter random projection".
  - At `image_target_size=256` the target square spans **0.47 visual tokens** (42.2 native px,
    measured over 49k fine-phase frames). No single token contains the target, so the
    "attend to the right patch" advantage does not exist at all. It should appear above ~512px.
  - `tip_target_pool=attn_residual` addresses the first: the attention's output projection is
    zero-initialized and the masked-mean vector is added back, so the head **starts exactly at the
    mean-pool solution** and the attention only ever learns a correction. It cannot start worse
    than `mean`.
  - The clean experiment is therefore 2x2: `{mean, attn_residual}` x `{256, 512}`. If
    `attn_residual` beats `mean` at 256, the problem was optimization; if only at 512, it was
    resolution. `attn` and `attn_residual` are mutually resumable; `mean` is not weight-compatible
    with either, so give it its own `RUN_NAME`.

### Task C — Decoupled position/rotation loss  [DONE, and taken further]
- Position and rot6d are separate weighted terms, and the **position term is itself split into a
  lateral and a depth part** — the two are not equally learnable, so one weight for "position"
  was hiding the real asymmetry.
- The split is taken in the **camera frame**, not the tip frame. Kabsch over 44.7k
  (heatmap-sidecar `u`/`v`/`depth_z`, `next.tip_target_9d`) pairs — rigid, rms 1.18 mm, since the
  wrist camera and the drill tip share a link — puts the optical axis at
  `[0.9037, -0.0012, -0.4281]` in the tip frame. **Tip-frame `z` is not depth**: depth is
  `0.9037*x - 0.4281*z`, so an x,y-vs-z split would leave 90% of the depth direction in the
  "lateral" term. Camera-frame target stats over 44.7k visible frames: lateral X `+3.7 ± 38.9 mm`,
  lateral Y `-2.0 ± 25.4 mm`, depth `148.0 ± 30.0 mm` (58–220 mm). 1 processed px = 0.381 mm of
  lateral error at the median depth; depth has no such direct readout.
- Config: `tip_target_lat_loss_weight` / `tip_target_depth_loss_weight` /
  `tip_target_rot_loss_weight`, with the rotation stored in `tip_target_cam_axes`. The defaults
  `2/3 + 1/3` reproduce the previous single xyz MSE **exactly** (a rotation preserves the L2
  norm), so the split is inert until deliberately reweighted.
- Logged in mm as `tip_target_lat_err_mm` / `tip_target_depth_err_mm` / `tip_target_pos_err_mm`.
- Not done: Huber/SmoothL1 on the position term. Still worth trying on top.
- The `rel_abs` objective has been **removed** (it was measured 5.7x worse).

### Task D — Config alignment & hygiene  [trivial]
- Make the action horizon consistent across `register_new_modality.py` (`G1_PALM_EEF_CHUNK_SIZE`),
  model config `action_horizon`, and the eval `--action-horizon` flag (checkpoints=40, config=30 now).
- Reduce color jitter in `start_finetune.sh` (task is red/green-semantic): `hue 0.08 → ~0.02`,
  `saturation 0.5 → ~0.25`; keep brightness/contrast.

### Task E — Exposure-bias mitigation  [DONE]
- Scheduled sampling is implemented and **off by default**. Set `tip_target_tf_gate_mm` (or
  `TIP_TARGET_TF_GATE_MM` in `start_finetune.sh`) to the mean position error in millimetres the
  head must reach before conditioning starts being handed over to its own prediction.
- It is a gated ramp, not a switch: an EMA of `tip_target_pos_err_mm` must hold under the
  threshold for `tip_target_tf_gate_patience` steps, the gate is sticky once open, and the
  per-sample handover probability then ramps `0 -> tip_target_tf_max_pred_prob` over
  `tip_target_tf_ramp_steps`.
- Two things that would silently break it and are handled: the gate metric is **all-reduced**
  across ranks (two GPUs disagreeing would feed different conditioning tokens for the same
  optimizer step), and the schedule lives in **persistent buffers** so a time-limited Euler job
  resumes it rather than restarting it.
- Pick the threshold off a previous run's `tip_target_pos_err_mm` curve — a value the head
  demonstrably reaches, not an aspiration. Too low and the gate never opens (harmless); too high
  and the action head is handed a conditioning token that is still mostly noise.

### Task F — Force the ego view into the head  [NEW, off by default]
- `tip_target_wrist_mask_prob` hides the **wrist** image tokens from the tip-target head's pooling
  on that fraction of training samples, so the head has to localize the target from the ego (head)
  camera and can learn to fuse the two views instead of leaning entirely on the wrist.
- It narrows the auxiliary head's key/value set only — the action head still sees every token, so
  it cannot destabilize policy training.
- It is meaningful rather than cosmetic: the backbone is causal and the views are concatenated
  **ego-first**, so ego token representations can never have attended to wrist tokens.
- **Check before trusting a result:** nothing supervises the ego view (the `heatmap_labels/`
  sidecar is wrist-only), the ego camera is natively 480px against the wrist's 720, and
  `image_target_size` is one global value — so above 480 the ego view is being upsampled. Confirm
  the target is resolvable in ego at all first.
- Note the interaction with Task B: `pool="mean"` cannot really fuse two views (it just averages),
  so this argues for one of the attention pools.

## 4. Order & notes
- C, E and F are implemented and land inert (defaults reproduce the previous behaviour exactly).
  A, B and D remain. B still needs a retrain to show effect; C/D ride along.
- A's code should land **before** P1 so the data regeneration "just works."

## 5. Do NOT
- Do not add rotation/flip image augmentation — it corrupts the tip-frame `rot6d` label.
- Do not chase `finalDist` with a stopping/progress-head change — that tail is contact physics.
- Do not just deepen/widen the MLP head without fixing pooling (Task B) — the spatial signal is
  already gone before the MLP.
