# Fine-Approach Perception Model (VLM-only heatmap finetune)

A second, specialized model for the last part of the drilling execution: instead of
regressing a 3D pose from far away, it looks at the **current wrist-camera frame only**
and predicts **where the wall target is in pixel coordinates** — plus whether it is
visible at all. It reuses the GR00T VLM backbone (Cosmos-Reason2-2B / Qwen3-VL) but has
**no action head**: the coarse approach keeps using your existing VLA checkpoints; this
model takes over at short range for precise servoing.

## What it predicts

| Output | Shape | Meaning |
|---|---|---|
| `heatmap` | (gh, gw) | Softmax over the VLM image-token grid (32 px/cell at 720 input) |
| `target_uv` | (2,) | Decoded target center: argmax cell + learned subpixel offset, in processed-image px |
| `visibility` | scalar | Sigmoid prob. that the target is in the image at all |
| `depth` | scalar | Target z in the camera frame, metres (optional head) |
| `wall_normal` | (3,) | Wall normal in the camera frame (optional head) |

Supervision comes from `g1_dataset/heatmap_labels/chunk-XXX/episode_XXXXXX.parquet`
(columns `u`, `v`, `visible`, `label_ok`, `depth_z`, `wall_normal_cam_*`), joined to
episodes by `frame_index`. **Training samples every sidecar row**
(`data.perception_labels_only=True`). Frames with `visible=False` train the visibility
head, while localization, offset, depth, and normal losses are masked out, so the model
learns to say "I don't see it" instead of hallucinating a location.

## Key mechanics

- **Keypoint-consistent augmentation**: the (u, v) label is passed through the *same*
  albumentations random-crop/resize as the image (`with_keypoints=True` in
  `gr00t/model/gr00t_n1d7/image_augmentations.py`). If the crop pushes the target out of
  frame, that sample's visibility label flips to 0.
- **Resolution-independent head**: coordinates are normalized to [0,1] over the
  processed image, so the Qwen smart-resize (720 → 704 internally) needs no special
  handling.
- **Losses**: heatmap soft cross-entropy vs a Gaussian target (`heatmap_target_sigma_px`),
  SmoothL1 subpixel offset at the GT cell, visibility BCE, optional depth SmoothL1 and
  wall-normal MSE. All masked terms are computed batch-wide and mask-weighted so DDP
  never sees unused parameters.
- **Backbone init**: `--base-model-path` accepts a **full VLA checkpoint** (your
  finetuned runs — action-head weights are ignored, backbone weights transferred) or a
  previous perception checkpoint. Prefer a run trained with `--tune-visual` at 720 px.

## Training

```bash
# Cluster (mirrors start_finetune.sh conventions):
RUN_NAME=perception_v1 BASE_MODEL_PATH=/path/to/your/vla/checkpoint bash start_finetune_perception.sh

# Or directly:
python gr00t/experiment/launch_finetune_perception.py \
    --base-model-path <vla-or-perception-checkpoint> \
    --dataset-path ./g1_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path register_perception_modality.py \
    --output-dir <dir> --tune-visual \
    --image-target-size 720 --image-crop-size 648
```

### Backbone init

The 2B VLM backbone (Cosmos-Reason2-2B) is **always** loaded from published weights —
it is never randomly initialised. `--base-model-path` only controls which GR00T-lineage
weights are overlaid on top of that architecture, and the perception head is **always
fresh**. Two modes:

| `--base-model-path` (or `BASE_MODEL_PATH=`) | Backbone | Use when |
|---|---|---|
| `nvidia/GR00T-N1.7-3B` (default) | GR00T foundation backbone | standard start |
| a finetuned VLA run, or a previous perception checkpoint | that run's backbone | reuse vision features already tuned on the task |

**The bare-Cosmos option is gone.** `scratch` / `none` / `""` used to skip the overlay
entirely, and that was never "training from scratch": the architecture is Cosmos-Reason2-2B
either way, because GR00T N1.7's VLM *is* a robot-finetuned Cosmos-Reason2-2B. All the
option did was discard GR00T's robot adaptation — and it did so **silently**, since a
missing `BASE_MODEL_PATH` produced a perfectly normal-looking run on un-adapted weights.
An empty value is now a hard error, raised before training starts.

To start a genuinely fresh perception run, a new `RUN_NAME` is all you need — an empty
output dir means training starts at step 0 with a fresh head:

```bash
RUN_NAME=perception_v5 bash start_finetune_perception.sh
# on Euler:
RUN_NAME=perception_v5 sbatch submit_perception_euler.sh
```

Note that `checkpoints-perception_v3_negatives` was trained with
`start_from_checkpoint: nvidia/GR00T-N1.7-3B` at `select_layer: 12` and 736/662 px — match
those if you want a comparable run, since `select_layer` now defaults to 16.

**After a model change you must bump `RUN_NAME`.** Reusing a name auto-resumes the
last checkpoint in its output dir, and a checkpoint written before the change lacks
the new parameters (e.g. the head's `text_*` cross-attention added by
`--use-text-conditioning`), so DeepSpeed's strict state-dict load dies with a
missing-keys error. Carry the old weights over as a warm start instead — matching
parameters load, new modules keep their initialization:

```bash
RUN_NAME=perception_v2 \
  BASE_MODEL_PATH=/groot/runs/perception_v1/checkpoint-<step> \
  bash start_finetune_perception.sh
```

That restarts at step 0 with a fresh optimizer and LR schedule, which is what you want
anyway: the old ones were built for the old architecture.

### The dataset is not pre-cut — every labelled frame is sampled

The `g1_dataset/data/**.parquet` (and the videos) are full episodes. Nothing is trimmed
on disk. `data.perception_labels_only=True` (set by the perception launcher) makes the
sharder sample every `frame_index` present in the `heatmap_labels/` sidecar, including
rows with `label_ok=False` or `visible=False`. This is deliberate: every row supplies a
visibility target, while the other perception losses use the visibility mask. Episodes
without a sidecar are excluded. Point the run at the full dataset; no separate cut
dataset is needed.

Monitor in wandb: `heatmap_loss`, `offset_loss`, `visibility_loss`, `depth_loss`,
`normal_loss`, and the direct quality metrics `uv_err_px_metric` (px error on visible
frames — at 0.12–0.2 m depth, 1 px ≈ 0.3–0.5 mm) and `visibility_acc_metric`.

## Inference / serving

This model has **no action head**, so `Gr00tPolicy` / `run_gr00t_server.py` (which call
`model.get_action` and decode a motor command) do **not** work with it. Use the
perception serving path instead — it implements the standard `BasePolicy.get_action`
interface over the **same** `get_action` ZMQ endpoint, so the existing
`gr00t.policy.server_client.PolicyClient` queries it unchanged; the returned "action"
dict just carries perception outputs.

### Serve

```bash
python gr00t/eval/run_perception_server.py \
    --model-path <perception-checkpoint-dir> \
    --embodiment-tag new_embodiment \
    --port 5555
```

### Query (standard client — this is what your robot loop calls)

```python
import numpy as np
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="localhost", port=5555, strict=False)
obs = {
    "video":    {"wrist_view": frame[None, None]},   # uint8 (B=1, T=1, H, W, C); native 720² square
    "language": {"annotation.human.action.task_description": [["drill in the red square"]]},
}
action, info = client.get_action(obs)      # options={"return_heatmap": True} to also get info["heatmap"]
action["target_uv_px"]           # (B, 2) target center in RAW wrist-camera px
action["visibility"]             # (B,)   sigmoid prob the target is in frame
action["visible"]                # (B,) bool  visibility > threshold
action["target_depth_m"]         # (B,)   camera-frame depth (if depth head trained)
action["wall_normal_cam"]        # (B, 3) unit wall normal, camera frame (if trained)
```

No `state` is required in the observation (the model ignores it; the policy feeds the
shared processor a zero state internally). Verified end to end on the `perceptionv1`
checkpoint: predicted vs ground-truth target ≈ **0.1 px**, visibility 0.9997, depth
within ~2 mm on a held frame.

### Direct (no server, for quick tests)

```python
from gr00t.policy.perception_policy import PerceptionPolicy
policy = PerceptionPolicy("new_embodiment", "<checkpoint-dir>", device="cuda")
action, info = policy.get_action(obs)      # same obs/return shape as above
```

Or drop to the model itself: `AutoModel.from_pretrained(dir)` + `model.predict(inputs)`
(inputs from `Gr00tN1d7PerceptionProcessor` in eval mode), then
`model.uv_processed_to_camera(pred["target_uv"], camera_size=(W, H))` to map the
processed-image pixel back to raw wrist-camera px (inverts the center-crop+resize eval
transform).

### From pixels to a servo target

Back-project `target_uv_px` with the wrist intrinsics (fx = fy = 388.09, cx = cy = 360 —
see `g1_dataset/heatmap_labels/labels_info.partial.json`) using the predicted `target_depth_m`,
or intersect the pixel ray with the wall plane from `wall_normal_cam` + FK. Gate
everything on `visibility` (e.g. hold the last confident estimate when it drops below ~0.5).

| Serving file | Role |
|---|---|
| `gr00t/policy/perception_policy.py` | `PerceptionPolicy` — `get_action` returns target uv / visibility / depth / normal |
| `gr00t/eval/run_perception_server.py` | Server entrypoint (registers the standard `get_action` endpoint) |

## Files

| File | Role |
|---|---|
| `gr00t/configs/model/gr00t_n1d7_perception.py` | `Gr00tN1d7PerceptionConfig` (head + loss knobs) |
| `gr00t/model/gr00t_n1d7/perception.py` | `Gr00tN1d7Perception` model + `TargetPerceptionHead` |
| `gr00t/model/gr00t_n1d7/processing_perception.py` | Keypoint-aware processor, emits `target_*` labels |
| `gr00t/model/gr00t_n1d7/setup_perception.py` | Pipeline: fresh processor, selective backbone weight transfer |
| `gr00t/data/dataset/lerobot_episode_loader.py` | Loads/joins `heatmap_labels/` sidecar (`target_uv`, …) |
| `gr00t/data/dataset/sharded_single_step_dataset.py` | all-sidecar-frame step filtering |
| `register_perception_modality.py` | Wrist-only modality config (video/state/action all `delta_indices=[0]`) |
| `gr00t/experiment/launch_finetune_perception.py` | CLI entry point (`PerceptionFinetuneConfig`) |
| `start_finetune_perception.sh` | Cluster launch script |

## Caveats

- Do **not** enable rotation augmentation: the keypoint would follow, but the
  wall-normal label would not.
- The perception processor intentionally ignores the checkpoint's stored processor
  config (it may carry old 256 px sizing); image resolution always comes from the CLI.
- `heatmap_target_sigma_px` trades sharpness vs training stability; if `uv_err_px_metric`
  plateaus high, try 5–6 px. The subpixel offset head is what carries precision below
  one 32 px cell.

## Generating actions from this perception

A checkpoint from this run is the recommended `--perception-model-path` init for the
**perception-conditioned action model**, which keeps this head and adds an action head
whose DiT is conditioned on the perception output cast into tokens. See
[PERCEPTION_ACTION_FINETUNE.md](PERCEPTION_ACTION_FINETUNE.md).
