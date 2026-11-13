#!/bin/bash
#SBATCH --job-name=groot-perception
#SBATCH --output=logs/groot-perception-%j.out
#SBATCH --error=logs/groot-perception-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --tmp=200G
# =============================================================================
# submit_perception_euler.sh — SLURM job script for the fine-approach
# PERCEPTION finetune (VLM backbone + target heatmap head) on ETH Euler.
#
# Usage:
#   cd ~/groot && sbatch submit_perception_euler.sh
#   # or, picking the run name and the VLA checkpoint to init the backbone from:
#   RUN_NAME=perception_v2 BASE_MODEL_PATH=/groot/runs/red_and_green/checkpoint-10000 \
#       sbatch submit_perception_euler.sh
#
# Retrieving the trained model afterwards (newest weights only; see submit_euler.sh):
#   scripts/fetch_euler_checkpoint.sh <run-name>
#
# Prerequisites: same container image and one-time setup as submit_euler.sh
# (see its header). Additionally:
#
#   1. Sync the perception dataset — INCLUDING the heatmap_labels/ sidecar — to
#      the work directory (from the repo root on your machine, where ./g1_dataset_small_general
#      holds data/ videos/ meta/ heatmap_labels/):
#        rsync -avP g1_dataset_small_general/ euler:/cluster/work/mrl/$USER/datasets/g1_dataset_small_general/
#
#   2. Make sure the codebase sync (rsync in submit_euler.sh header) has picked
#      up the new files; they are bind-mounted individually below:
#        start_finetune_perception.sh, setup_and_train_perception.sh,
#        register_perception_modality.py, gr00t/
#
#   3. BASE_MODEL_PATH must be reachable INSIDE the container and must be
#      GR00T-lineage. Scratch is mounted as /groot, so a finetuned VLA run lives at
#      e.g. /groot/runs/<run_name>/checkpoint-XXXX. An HF id like
#      nvidia/GR00T-N1.7-3B also works (downloads via HF_TOKEN from .env).
#      Keep the default unless you have a reason not to. Initialising from one of
#      your own VLA runs used to be the advice, on the assumption they were 720px;
#      they were not. Until commit a66a334 the processor silently dropped the
#      resolution overrides, so every VLA run trained at 256px and its vision
#      encoder is adapted to that, making it a worse init here than the GR00T
#      weights. Check a candidate's [resolution] log line before trusting it.
#
#      There is no "bare Cosmos-Reason2-2B" option any more. The backbone
#      ARCHITECTURE is always Cosmos-Reason2-2B — GR00T N1.7's VLM is that same 2B
#      model, robot-finetuned — so skipping the overlay never meant "from scratch",
#      it just discarded the robot adaptation, and it did so silently. An empty
#      BASE_MODEL_PATH is now a hard error.
# =============================================================================

set -euo pipefail

# --- Load modules ---
module load eth_proxy

# --- Paths (adjust to match your Euler setup) ---
PROJECT_DIR="$HOME/groot"
CONTAINER_TAR="/cluster/work/mrl/$USER/containers/groot-environment.tar.gz"
GROOT_DIR="/cluster/scratch/$USER/groot"
RESULTS_DIR="/cluster/project/mrl/$USER/results/groot/$SLURM_JOB_ID"
DATASET_HOST_DIR="${DATASET_HOST_DIR:-/cluster/work/mrl/$USER/datasets/g1_dataset_small_general}"

# Backbone init for the perception model (path as seen INSIDE the container).
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"

# --- Job info ---
echo "=== groot Perception Finetune Job ==="
echo "Job ID:          $SLURM_JOB_ID"
echo "Host:            $(hostname)"
echo "Date:            $(date)"
echo "GPU:             $CUDA_VISIBLE_DEVICES"
echo "TMPDIR:          $TMPDIR"
echo "RUN_NAME:        ${RUN_NAME:-perception_v1}"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "DATASET:         $DATASET_HOST_DIR"
echo "====================================="

# --- Create directories ---
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$GROOT_DIR/cache/checkpoints"
mkdir -p "$GROOT_DIR/cache/triton"
mkdir -p "$GROOT_DIR/cache/torch_kernels"
mkdir -p "$GROOT_DIR/data"
mkdir -p "$GROOT_DIR/runs"
mkdir -p "$RESULTS_DIR"

# --- Sanity: the dataset must ship the heatmap label sidecar ---
if [ ! -d "$DATASET_HOST_DIR/heatmap_labels" ]; then
    echo "[euler] ERROR: $DATASET_HOST_DIR/heatmap_labels not found."
    echo "[euler] The perception finetune trains on fine-phase frames from the"
    echo "[euler] heatmap_labels/ sidecar. Rsync the dataset including it (see header)."
    exit 1
fi

# --- Extract container to local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Copy dataset to local scratch (avoids slow NFS/Lustre reads) ---
# On top of the stats scan done on rank 0 (see submit_euler.sh), the perception
# sharding additionally reads 3 columns from every heatmap label parquet
# (~1500 files) to build the fine-phase step masks — local NVMe makes both fast.
# The container path, the bind target and the launcher's --dataset-path all derive
# from the host directory name, so pointing DATASET_HOST_DIR at another dataset is
# the only edit needed.
DATASET_NAME="$(basename "$DATASET_HOST_DIR")"
DATASET_LOCAL_DIR="$TMPDIR/$DATASET_NAME"
echo "[euler] Rsyncing dataset to local scratch (${DATASET_LOCAL_DIR})..."
time rsync -a --info=progress2 "$DATASET_HOST_DIR/" "$DATASET_LOCAL_DIR/"
echo "[euler] Dataset copy done."

# --- Launch with Apptainer ---
# Bindings mirror submit_euler.sh; differences:
#   - dataset mounts at /workspace/$DATASET_NAME, and DATASET_DIR is forwarded to
#     match, so DATASET_HOST_DIR is the only place the dataset name is written
#   - perception launch/setup scripts + wrist-only modality config are mounted
#   - RUN_NAME / BASE_MODEL_PATH and the run-shaping vars forwarded into the container
echo "[euler] Starting container..."
time apptainer exec --nv \
    --bind "$GROOT_DIR:/groot" \
    --bind "$PROJECT_DIR/.env:/workspace/.env" \
    --bind "$RESULTS_DIR:/results" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/gr00t:/workspace/gr00t" \
    --bind "$PROJECT_DIR/setup_and_train_perception.sh:/workspace/setup_and_train_perception.sh" \
    --bind "$PROJECT_DIR/start_finetune_perception.sh:/workspace/start_finetune_perception.sh" \
    --bind "$PROJECT_DIR/register_perception_modality.py:/workspace/register_perception_modality.py" \
    --bind "$DATASET_LOCAL_DIR:/workspace/$DATASET_NAME" \
    --env TRITON_CACHE_DIR=/groot/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/groot/cache/torch_kernels \
    --env XDG_CACHE_HOME=/groot/cache \
    --env RUN_NAME="${RUN_NAME:-perception_v1}" \
    --env BASE_MODEL_PATH="$BASE_MODEL_PATH" \
    --env HEATMAP_SIGMA_PX="${HEATMAP_SIGMA_PX:-}" \
    --env HEATMAP_LOSS_WEIGHT="${HEATMAP_LOSS_WEIGHT:-}" \
    --env OFFSET_LOSS_WEIGHT="${OFFSET_LOSS_WEIGHT:-}" \
    --env VISIBILITY_LOSS_WEIGHT="${VISIBILITY_LOSS_WEIGHT:-}" \
    --env VISIBILITY_NEG_WEIGHT="${VISIBILITY_NEG_WEIGHT:-}" \
    --env INCLUDE_COARSE_LABELLED="${INCLUDE_COARSE_LABELLED:-}" \
    --env DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-}" \
    --env WALL_NORMAL_LOSS_WEIGHT="${WALL_NORMAL_LOSS_WEIGHT:-}" \
    --env LEARNING_RATE="${LEARNING_RATE:-}" \
    --env DATASET_DIR="${DATASET_DIR:-./$DATASET_NAME/}" \
    --env IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-}" \
    --env IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-}" \
    --env GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}" \
    --env GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}" \
    --env NUM_EPOCHS="${NUM_EPOCHS:-}" \
    --env MAX_STEPS="${MAX_STEPS:-}" \
    --env SAVE_STEPS="${SAVE_STEPS:-}" \
    "$TMPDIR/groot-environment.sif" \
    bash /workspace/setup_and_train_perception.sh

echo "[euler] Job completed at $(date)"
