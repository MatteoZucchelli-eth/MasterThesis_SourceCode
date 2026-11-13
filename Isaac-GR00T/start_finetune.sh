#!/bin/bash
set -euo pipefail

source .env

# Output run name. Keep this STABLE across resubmissions of the *same* experiment
# so training resumes from the last checkpoint (each Euler job is time-limited).
# Bump it (e.g. red_and_green_v2) to start a FRESH experiment — reusing a name
# whose checkpoint already hit --max-steps makes the trainer exit immediately.
# Override per run with:  RUN_NAME=red_and_green_v2 sbatch submit_euler.sh
RUN_NAME="${RUN_NAME:-red_and_green}"

# Dataset root, as seen from the training cwd (/workspace in the container).
# g1_dataset_red_2k: 925 episodes / 837k frames carved out of g1_dataset_general_2k,
# keeping ONLY the red-target episodes (all 5 instruction phrasings). All three
# coloured squares are still physically in frame, so the instruction still has to
# select the target — but target-colour generalization is out of the picture, which
# isolates tip-target precision. Use this for the pooling ablation.
#   g1_dataset_general_2k/g1_dataset_general_2k  2763 eps, all 3 colours (superset)
#   g1_dataset_small_general                      747 eps, all 3 colours
#   g1_dataset                                   1515 eps, single phrasing, red only
# Override with DATASET_DIR=... to train on a different capture.
DATASET_DIR="${DATASET_DIR:-./g1_dataset_red_2k/}"

# Modality config registered under NEW_EMBODIMENT. It must match the dataset's
# meta/modality.json: a group named there but missing from the parquet layout is
# only WARNED about by LeRobotEpisodeLoader._extract_joint_groups, so a mismatch
# silently narrows the state vector instead of failing.
#   register_new_modality.py         35-dim state captures (red_2k, negatives, ...)
#   register_autocollect_modality.py 23-dim right-arm-only capture (autocollect_v2)
MODALITY_CONFIG="${MODALITY_CONFIG:-register_new_modality.py}"

# Set to true to train the auxiliary tip-target head: predicts the target pose
# in the drill-tip frame as xyz(3) + rot6d(6) from the `next.tip_target_9d`
# label and feeds it back as a conditioning token to the action head
# (conditioning is on by default via --use-tip-target-conditioning).
#
# USE_TIP_TARGET_HEAD=false is the VANILLA GR00T N1.7 architecture: no auxiliary
# head, no conditioning token, loss == the flow-matching action loss alone. Every
# tip-target flag below is then inert (launch_finetune.py still forwards them into
# the model config, but Gr00tN1d7ActionHead never constructs the head). Use it for
# the baseline arm of "does the tip-target head actually buy anything".
USE_TIP_TARGET_HEAD="${USE_TIP_TARGET_HEAD:-true}"
TIP_TARGET_LOSS_WEIGHT=0.5       # overall tip-target loss multiplier vs the action loss
TIP_TARGET_XYZ_SCALE=0.1         # metres; scales xyz inside the head's prediction space

# Position loss split. The position error is rotated into the CAMERA frame and split
# into a LATERAL part (perpendicular to the optical axis) and a DEPTH part (along it),
# because the two are not equally learnable: 1 processed px is 0.381 mm of lateral
# error at the median 148 mm working depth, whereas depth has to be inferred from
# apparent size (the target sits at 148 +/- 31 mm, spanning 53-225 mm).
#
# The rotation matters. Kabsch over 44.7k (heatmap-sidecar u/v/depth_z,
# next.tip_target_9d) pairs -- rigid, rms 1.18 mm, since the wrist camera and the drill
# tip share a link -- puts the optical axis at [0.9037, -0.0012, -0.4281] in the TIP
# frame. So tip-frame z is NOT depth: depth = 0.9037*x - 0.4281*z. Splitting the
# label's own x,y vs z would leave 90% of the depth direction in the "lateral" term.
#
# 2/3 + 1/3 is NOT arbitrary: a rotation preserves the L2 norm, so those weights
# reproduce the previous single xyz MSE exactly. The split is inert until you change
# them -- which makes any reweighting a clean one-variable experiment. Raise DEPTH to
# spend more capacity on the hard direction; lower it to stop an under-determined
# depth signal from polluting the lateral estimate.
TIP_TARGET_LAT_LOSS_WEIGHT="${TIP_TARGET_LAT_LOSS_WEIGHT:-0.6666666666666666}"
TIP_TARGET_DEPTH_LOSS_WEIGHT="${TIP_TARGET_DEPTH_LOSS_WEIGHT:-0.3333333333333333}"
TIP_TARGET_ROT_LOSS_WEIGHT="${TIP_TARGET_ROT_LOSS_WEIGHT:-1.0}"   # rot6d MSE vs the position term

# Probability of hiding the WRIST image tokens from the tip-target head's pooling on a
# training sample, forcing it to localize the target from the ego (head) camera and so
# to learn to use both views rather than leaning entirely on the wrist. This narrows
# the AUXILIARY HEAD's view only -- the action head still sees every token, so it
# cannot destabilize policy training. It is meaningful rather than cosmetic because the
# backbone is causal and the views are concatenated ego-first, so ego token
# representations can never have attended to wrist tokens.
#
# 0 (default) disables it. Before trusting a result here, check that the target is
# resolvable in ego at all: nothing supervises the ego view (the heatmap_labels/
# sidecar is wrist-only), the ego camera is natively 480px against the wrist's 720,
# and IMAGE_TARGET_SIZE is one global value, so above 480 ego is being upsampled.
TIP_TARGET_WRIST_MASK_PROB="${TIP_TARGET_WRIST_MASK_PROB:-0}"

# Scheduled sampling of the conditioning token. Training teacher-forces the token on
# the LABEL, but at deployment the action head only ever sees the head's own
# prediction. Set TIP_TARGET_TF_GATE_MM to the mean position error (MILLIMETRES) the
# head must reach before conditioning starts being handed over to that prediction.
# Empty (default) = teacher-force for the whole run, the previous behaviour.
#
# The gate watches an EMA of the position error, needs it to hold under the threshold
# for TIP_TARGET_TF_GATE_PATIENCE steps, and is sticky -- it never re-closes. Crossing
# it ramps the per-sample handover probability 0 -> TIP_TARGET_TF_MAX_PRED_PROB over
# TIP_TARGET_TF_RAMP_STEPS rather than switching in one step. The decision is
# all-reduced across the 2 GPUs (ranks that disagreed would feed different conditioning
# tokens for the same optimizer step) and lives in the checkpoint, so resubmitting this
# job resumes the schedule instead of restarting it.
#
# Pick the threshold off a previous run's tip_target_pos_err_mm curve: it should be a
# value the head demonstrably reaches, not an aspiration. Too low and the gate never
# opens (harmless, identical to the old behaviour); too high and you hand the action
# head a conditioning token that is still mostly noise.
TIP_TARGET_TF_GATE_MM="${TIP_TARGET_TF_GATE_MM:-}"
TIP_TARGET_TF_RAMP_STEPS="${TIP_TARGET_TF_RAMP_STEPS:-2000}"
TIP_TARGET_TF_MAX_PRED_PROB="${TIP_TARGET_TF_MAX_PRED_PROB:-1.0}"
TIP_TARGET_TF_GATE_PATIENCE="${TIP_TARGET_TF_GATE_PATIENCE:-200}"
TIP_TARGET_TF_GATE_EMA="${TIP_TARGET_TF_GATE_EMA:-0.99}"

# How the VL token sequence is pooled into the vector fed to the head's predictor MLP:
#   attn_residual  num_queries learned queries cross-attending over all vision+language
#                  tokens, PLUS the masked-mean vector added back to every pooled query,
#                  with the attention's output projection zero-initialized (default)
#   mean           plain masked mean over all valid tokens
#
# The bare no-residual "attn" pool was REMOVED. Without the residual the pool is a
# single cross-attention layer with no skip and no FFN whose queries start random, so
# the softmax begins near-uniform and training starts at "a mean pool through an
# untrained 16.8M-parameter random projection" -- and the gradient reaching the backbone
# is routed by whatever those random queries happened to attend to. With the residual
# the head starts AT the mean-pool solution and the attention only learns a correction.
# An old bare-attn checkpoint still loads (same shapes) but will NOT reproduce.
#
# NOTE what the measured evidence actually says, before assuming mean is the safe pick.
# At 256px, paired evals in scratchpad/ have attn AHEAD on mean error:
#   head_to_head.csv        (480 samples)  attn 5.60 mm / med 3.51   mean 6.53 / 3.47
#   tip_eval_512_vs_256.csv (500 @256px)   attn 9.89 mm / med 4.82   mean 10.85 / 5.03
# mean only wins in the 512px rows of that second file, which are CONFOUNDED (12,662
# steps against 25,323 -- half the samples). attn_residual should dominate both.
#
# mean is not weight-compatible with attn_residual (different predictor input width), so
# give it its own RUN_NAME.
TIP_TARGET_POOL="${TIP_TARGET_POOL:-attn_residual}"
TIP_TARGET_NUM_QUERIES="${TIP_TARGET_NUM_QUERIES:-4}"   # attn_residual only
TIP_TARGET_ATTN_HEADS="${TIP_TARGET_ATTN_HEADS:-8}"     # attn_residual only

TIP_TARGET_TF_ARGS=""
[ -n "$TIP_TARGET_TF_GATE_MM" ] && TIP_TARGET_TF_ARGS="--tip-target-tf-gate-mm ${TIP_TARGET_TF_GATE_MM} \
    --tip-target-tf-ramp-steps ${TIP_TARGET_TF_RAMP_STEPS} \
    --tip-target-tf-max-pred-prob ${TIP_TARGET_TF_MAX_PRED_PROB} \
    --tip-target-tf-gate-patience ${TIP_TARGET_TF_GATE_PATIENCE} \
    --tip-target-tf-gate-ema ${TIP_TARGET_TF_GATE_EMA}"

TIP_TARGET_ARGS=""
[ "$USE_TIP_TARGET_HEAD" = true ] && TIP_TARGET_ARGS="--use-tip-target-head \
    --tip-target-loss-weight ${TIP_TARGET_LOSS_WEIGHT} \
    --tip-target-pool ${TIP_TARGET_POOL} \
    --tip-target-num-queries ${TIP_TARGET_NUM_QUERIES} \
    --tip-target-attn-heads ${TIP_TARGET_ATTN_HEADS} \
    --tip-target-lat-loss-weight ${TIP_TARGET_LAT_LOSS_WEIGHT} \
    --tip-target-depth-loss-weight ${TIP_TARGET_DEPTH_LOSS_WEIGHT} \
    --tip-target-rot-loss-weight ${TIP_TARGET_ROT_LOSS_WEIGHT} \
    --tip-target-wrist-mask-prob ${TIP_TARGET_WRIST_MASK_PROB} \
    --tip-target-xyz-scale ${TIP_TARGET_XYZ_SCALE} \
    ${TIP_TARGET_TF_ARGS}"

# Image resolution. These flags used to be INERT -- from_pretrained's override
# allowlist dropped image_target_size/image_crop_size/shortest_image_edge/
# crop_fraction, so every run before 2026-07-30 trained at the base model's 256px
# with crop_fraction 0.95 whatever these said. That is fixed; they now take effect,
# so they are PINNED to the values that reproduce the historical behaviour:
#   256 target with crop 243 -> crop fraction 243/256 = 0.949 ~= the old 0.95,
#   yielding 64 visual tokens per image and a 275-token sequence, as before.
# Raising this is a real experiment now, not a no-op. Reference points, measured:
#   target 384 crop 344 -> 144 tok/img, seq  595, ~54 native px per token
#   target 512 crop 460 -> 256 tok/img, seq 1043, ~40 native px per token
#   target 720 crop 648 -> 484 tok/img, seq 1955, ~30 native px per token
# MEASURED target size (scratchpad/ measure over 49,204 fine-phase visible frames of
# g1_dataset_autocollect_v2, from the corner_uv_* label columns): the target square is
# 42.2 px per side in the native 720px wrist frame at the median fine-phase range of
# 154 mm -- NOT the ~25 px this comment used to claim. In visual tokens (1 token = 32
# processed px) that is:
#     256 -> 0.47 tok    384 -> 0.70 tok    512 -> 0.94 tok    720 -> 1.32 tok
# Below ~1 token the target cannot be localized by WHICH token fires, only by that
# token's content, so 256 px really is coarse for this task. Two hard ceilings though:
# the wrist is natively 720 and the EGO IS NATIVELY 480, and image_target_size is a
# single global value applied to every view (the transform is built once in
# processing_gr00t_n1d7.py) -- so anything above 480 already upsamples the ego view and
# anything above 720 is pure upsampling. Cost is steep: VLM self-attention is O(L^2) in
# the token count, so 256->384 is 4.5x, 256->512 is 13.6x, 256->720 is 47x.
# Change it on its own, not alongside another variable, and watch memory.
#
# NOTE the 512-vs-256 numbers in scratchpad/tip_eval_512_vs_256.csv are CONFOUNDED:
# the 512 runs stopped at 12,662 steps against 25,323 for the 256 runs, i.e. half the
# samples. They are not evidence that resolution hurts.

# Batch. Effective batch = GLOBAL_BATCH_SIZE x GRAD_ACCUM_STEPS, and GLOBAL_BATCH_SIZE
# is split across the GPUs (32 over 2 GPUs = 16/GPU). Keep the product at 32 when
# trading one for the other, so the optimizer sees the same thing. Raising the image
# resolution multiplies visual tokens by (target/256)^2 -- 512px is 4x -- so this is
# the knob to drop first if a higher-resolution run OOMs:
#   GLOBAL_BATCH_SIZE=16 GRAD_ACCUM_STEPS=2   (same effective 32, half the activations)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

# Training length. Set NUM_EPOCHS to size the run by the dataset instead of by a step
# count: the launcher computes max_steps = ceil(epochs * trainable_steps / effective
# batch), where trainable_steps = sum(episode_length - action_horizon + 1). For
# g1_dataset_red_2k that is 810,321 steps, so at batch 32 one epoch = 25,323 steps.
#   NUM_EPOCHS=1 bash start_finetune.sh
# Leave NUM_EPOCHS empty to use MAX_STEPS verbatim.
# Note "epoch" is a sample-count target, not a true pass: batches are drawn from a
# shard buffer with replacement, so it means "each frame seen NUM_EPOCHS times in
# expectation". warmup_ratio scales with max_steps automatically; SAVE_STEPS does not,
# so it is derived below to give ~5 checkpoints either way.
NUM_EPOCHS="${NUM_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
if [ -n "$NUM_EPOCHS" ]; then
    TRAIN_LENGTH_ARGS="--num-epochs ${NUM_EPOCHS}"
    # ~810321 trainable steps for g1_dataset_red_2k; steps/epoch = that / effective batch.
    # Only a heuristic for checkpoint cadence -- the authoritative max_steps is computed
    # by launch_finetune.py from the dataset actually being used.
    SAVE_STEPS="${SAVE_STEPS:-$(python -c "import math;eff=${GLOBAL_BATCH_SIZE}*${GRAD_ACCUM_STEPS};print(max(200, math.ceil(${NUM_EPOCHS}*810321/eff/5/100)*100))")}"
else
    TRAIN_LENGTH_ARGS="--max-steps ${MAX_STEPS}"
    SAVE_STEPS="${SAVE_STEPS:-2000}"
fi
echo "[train] TRAIN_LENGTH_ARGS=${TRAIN_LENGTH_ARGS}  SAVE_STEPS=${SAVE_STEPS}"

IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-256}"
IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-243}"

# Generate a random port between 10000 and 65000 to avoid collisions
MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# Each experiment gets its own output dir keyed on RUN_NAME so that bumping
# RUN_NAME actually starts a fresh run. Reusing a name whose checkpoint already
# reached --max-steps makes HF Trainer return immediately (resume_from_checkpoint
# sees global_step >= max_steps), which looks like an instant, error-free "success".
OUTPUT_DIR="${GROOT_HOME:-/groot}/runs/${RUN_NAME}"

# --- Debug: surface the resolved configuration (goes to stdout / the .out log) ---
echo "[train] RUN_NAME=${RUN_NAME}"
echo "[train] DATASET_DIR=${DATASET_DIR}"
echo "[train] MODALITY_CONFIG=${MODALITY_CONFIG}"
echo "[train] USE_TIP_TARGET_HEAD=${USE_TIP_TARGET_HEAD} (false = vanilla GR00T N1.7)"
echo "[train] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train] MASTER_PORT=${MASTER_PORT}"
echo "[train] TIP_TARGET_ARGS=${TIP_TARGET_ARGS:-<none>}"
echo "[train] TIP_TARGET_POOL=${TIP_TARGET_POOL}  WRIST_MASK_PROB=${TIP_TARGET_WRIST_MASK_PROB}"
echo "[train] TIP_TARGET_LAT/DEPTH/ROT weights = ${TIP_TARGET_LAT_LOSS_WEIGHT} / ${TIP_TARGET_DEPTH_LOSS_WEIGHT} / ${TIP_TARGET_ROT_LOSS_WEIGHT}"
if [ -n "$TIP_TARGET_TF_GATE_MM" ]; then
    echo "[train] TEACHER FORCING: hand over to the head's own prediction once the EMA'd"
    echo "[train]   position error holds below ${TIP_TARGET_TF_GATE_MM} mm for ${TIP_TARGET_TF_GATE_PATIENCE} steps,"
    echo "[train]   then ramp to p=${TIP_TARGET_TF_MAX_PRED_PROB} over ${TIP_TARGET_TF_RAMP_STEPS} steps"
else
    echo "[train] TEACHER FORCING: always on (no gate) — set TIP_TARGET_TF_GATE_MM to hand over"
fi
echo "[train] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} (effective $((GLOBAL_BATCH_SIZE*GRAD_ACCUM_STEPS)))"
echo "[train] IMAGE_TARGET_SIZE=${IMAGE_TARGET_SIZE} IMAGE_CROP_SIZE=${IMAGE_CROP_SIZE}"
echo "[train]   -> requested crop fraction $(python -c "print(f'{${IMAGE_CROP_SIZE}/${IMAGE_TARGET_SIZE}:.3f}')"), ~$(python -c "print((${IMAGE_TARGET_SIZE}//32)**2)") visual tokens/image"
echo "[train]   (grep the training log for '[resolution]' to see what the processor ACTUALLY used)"
if [ -d "$OUTPUT_DIR" ]; then
    echo "[train] WARNING: output dir already exists. Existing checkpoints:"
    ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "[train]   (none)"
    echo "[train] If a checkpoint already hit --max-steps, training will resume-and-exit"
    echo "[train] immediately. Bump RUN_NAME to start fresh."
else
    echo "[train] Fresh run — no existing output dir."
fi
# --- Debug: verify the environment can actually run the entry point ---
echo "[train] cwd=$(pwd)"
echo "[train] which torchrun: $(command -v torchrun || echo MISSING)"
echo "[train] which python:   $(command -v python || echo MISSING)"
echo "[train] entry exists:   $(ls -l gr00t/experiment/launch_finetune.py 2>&1)"
echo "[train] gr00t import:   $(python -c 'import gr00t, torch; print("gr00t ok, torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())' 2>&1)"
echo "[train] Launching torchrun..."

# Run torchrun with stderr merged into stdout so EVERYTHING lands in the .out log,
# and capture the exit code instead of letting `set -e` swallow it silently.
set +e
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "${DATASET_DIR}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps ${SAVE_STEPS} \
    ${TRAIN_LENGTH_ARGS} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM_STEPS} \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.03 \
    --image-target-size ${IMAGE_TARGET_SIZE} \
    --image-crop-size ${IMAGE_CROP_SIZE} \
    --num-gpus 2 \
    --use-wandb \
    --tune-visual \
    $TIP_TARGET_ARGS 2>&1
TORCHRUN_EXIT=$?
set -e
echo "[train] torchrun exited with code: ${TORCHRUN_EXIT}"
exit "${TORCHRUN_EXIT}"
