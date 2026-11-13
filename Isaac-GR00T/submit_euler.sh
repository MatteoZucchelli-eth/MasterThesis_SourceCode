#!/bin/bash
#SBATCH --job-name=groot-small-red-predict
#SBATCH --output=logs/groot-%j.out
#SBATCH --error=logs/groot-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --tmp=200G
# =============================================================================
# submit_euler.sh — SLURM job script for groot finetuning on ETH Euler
#
# Usage:
#   cd ~/groot && sbatch submit_euler.sh
#
# Prerequisites:
#   1. Build the Docker image locally (from repo root):
#        bash docker/build.sh
#      This produces the image tagged "gr00t".
#
#   2. Convert to Apptainer .sif and compress (rootful Docker):
#        apptainer build groot-environment.sif docker-daemon://gr00t:latest
#        tar -czf groot-environment.tar.gz groot-environment.sif
#        rm groot-environment.sif
#
#   3. Transfer to Euler:
#        # Send the heavy container image to the work directory
#    rsync -avP groot-environment.tar.gz \
#        euler:/cluster/work/mrl/$USER/containers/
#
#        # Sync the codebase to the home directory, excluding heavy/local folders.
#        # $HOME on Euler is quota-limited (~45 GB / 450k files), so keep datasets,
#        # checkpoints, and vendored deps out of it.
#        rsync -avP \
#            --exclude='.git' \
#            --exclude='__pycache__' \
#            --exclude='*.egg-info' \
#            --exclude='.venv' \
#            --exclude='venv' \
#            --exclude='checkpoints*' \
#            --exclude='g1_dataset*' \
#            --exclude='external_dependencies' \
#            --exclude='media' \
#            --exclude='demo_data' \
#            --exclude='wandb' \
#            --exclude='*.gz' \
#            . euler:~/groot/
#
#        # Sync the dataset directly to the high-capacity work directory.
#        # The trailing target name must match DATASET_HOST_DIR below.
#        rsync -avP g1_dataset_red_2k/ euler:/cluster/work/mrl/$USER/datasets/g1_dataset_red_2k/
#
#   RETRIEVING A TRAINED MODEL (after the job finishes):
#        scripts/fetch_euler_checkpoint.sh <run-name>
#      Fetches the NEWEST weights only: the final model at the run root if the run
#      finished, otherwise the highest checkpoint-N. It skips the optimizer/DeepSpeed/
#      resume state, and KEEPS experiment_cfg/ -- which holds
#      final_processor_config.json, the only reliable record of the image geometry
#      the run actually tokenized with. config.json is NOT a substitute for it: it
#      carries fields inherited from the base checkpoint that the processor
#      overrode, and the two do disagree. The script prints that geometry after
#      the transfer and fails if the file is missing.
#
#   4. One-time setup on Euler (run once after first transfer):
#        mkdir -p /cluster/work/mrl/$USER/containers
#        mkdir -p /cluster/project/mrl/$USER/results/groot
#        mkdir -p /cluster/scratch/$USER/groot/{cache/checkpoints,data,runs}
#        mkdir -p ~/groot/logs
#        # REQUIRED: the apptainer run below binds ~/groot/.env and will fail if
#        # the file is missing. Create it and fill in your tokens.
#        cp ~/groot/.env.sample ~/groot/.env  # then edit with your tokens
# =============================================================================

set -euo pipefail

# --- Load modules ---
module load eth_proxy


# --- Paths (adjust to match your Euler setup) ---
PROJECT_DIR="$HOME/groot"
CONTAINER_TAR="/cluster/work/mrl/$USER/containers/groot-environment.tar.gz"
GROOT_DIR="/cluster/scratch/$USER/groot"
RESULTS_DIR="/cluster/project/mrl/$USER/results/groot/$SLURM_JOB_ID"
DATASET_HOST_DIR="${DATASET_HOST_DIR:-/cluster/work/mrl/$USER/datasets/g1_dataset_red_2k}"

# The container path, the bind target and the launcher's --dataset-path all derive
# from the host directory's BASENAME, so pointing DATASET_HOST_DIR at another capture
# is the only edit needed. (This used to be hardcoded to g1_dataset_red_2k in three
# places, which meant a new dataset trained under the old name.)
DATASET_NAME="$(basename "$DATASET_HOST_DIR")"

# Modality config to register under NEW_EMBODIMENT — must match the dataset's
# meta/modality.json (see start_finetune.sh). Bound into the container by name.
MODALITY_CONFIG="${MODALITY_CONFIG:-register_new_modality.py}"
if [ ! -f "$PROJECT_DIR/$MODALITY_CONFIG" ]; then
    echo "[euler] ERROR: modality config $PROJECT_DIR/$MODALITY_CONFIG not found."
    echo "[euler] Did the codebase rsync pick it up? (see header)"
    exit 1
fi

# --- Job info ---
echo "=== groot Finetune Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "TMPDIR:    $TMPDIR"
echo "DATASET:   $DATASET_HOST_DIR  (-> /workspace/$DATASET_NAME)"
echo "MODALITY:  $MODALITY_CONFIG"
echo "RUN_NAME:  ${RUN_NAME:-red_and_green}"
echo "TIP HEAD:  ${USE_TIP_TARGET_HEAD:-<default: true>}"
echo "POOL:      ${TIP_TARGET_POOL:-<default: attn_residual>}  (attn_residual | mean)"
echo "WRIST MASK:${TIP_TARGET_WRIST_MASK_PROB:-<default: 0 = off>}"
echo "TF GATE:   ${TIP_TARGET_TF_GATE_MM:-<default: none, teacher-force throughout>}"
echo "========================="

# --- Create directories ---
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$GROOT_DIR/cache/checkpoints"
mkdir -p "$GROOT_DIR/cache/triton"
mkdir -p "$GROOT_DIR/cache/torch_kernels"
mkdir -p "$GROOT_DIR/data"
mkdir -p "$GROOT_DIR/runs"
mkdir -p "$RESULTS_DIR"

# --- Extract container to local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Copy dataset to local scratch (avoids slow NFS reads during stats generation) ---
# generate_stats/generate_rel_stats scans all parquet files on rank 0 before the NCCL
# barrier — on Lustre this takes ~4.5 s/file (~10 min for 131 files), exceeding the
# 600 s barrier timeout.  Reading from local NVMe reduces this to <10 s total.
DATASET_LOCAL_DIR="$TMPDIR/$DATASET_NAME"
echo "[euler] Rsyncing dataset to local scratch (${DATASET_LOCAL_DIR})..."
time rsync -a --info=progress2 "$DATASET_HOST_DIR/" "$DATASET_LOCAL_DIR/"
echo "[euler] Dataset copy done."

# --- Launch with Apptainer ---
# --nv:                 enable NVIDIA GPU support
# --bind /groot:        scratch storage for checkpoints, data, caches (read-write)
# --bind .env:          mount env file with secrets (HF_TOKEN, WANDB, etc.)
# --bind /results:      persistent results on project storage
# --env:                redirect triton/torch caches to writable scratch
echo "[euler] Starting container..."
time apptainer exec --nv \
    --bind "$GROOT_DIR:/groot" \
    --bind "$PROJECT_DIR/.env:/workspace/.env" \
    --bind "$RESULTS_DIR:/results" \
    --bind "$PROJECT_DIR/scripts:/workspace/scripts" \
    --bind "$PROJECT_DIR/gr00t:/workspace/gr00t" \
    --bind "$PROJECT_DIR/setup_and_train.sh:/workspace/setup_and_train.sh" \
    --bind "$PROJECT_DIR/start_finetune.sh:/workspace/start_finetune.sh" \
    --bind "$PROJECT_DIR/$MODALITY_CONFIG:/workspace/$MODALITY_CONFIG" \
    --bind "$DATASET_LOCAL_DIR:/workspace/$DATASET_NAME" \
    --env TRITON_CACHE_DIR=/groot/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/groot/cache/torch_kernels \
    --env XDG_CACHE_HOME=/groot/cache \
    --env RUN_NAME="${RUN_NAME:-red_and_green}" \
    --env TIP_TARGET_POOL="${TIP_TARGET_POOL:-}" \
    --env TIP_TARGET_LAT_LOSS_WEIGHT="${TIP_TARGET_LAT_LOSS_WEIGHT:-}" \
    --env TIP_TARGET_DEPTH_LOSS_WEIGHT="${TIP_TARGET_DEPTH_LOSS_WEIGHT:-}" \
    --env TIP_TARGET_ROT_LOSS_WEIGHT="${TIP_TARGET_ROT_LOSS_WEIGHT:-}" \
    --env TIP_TARGET_WRIST_MASK_PROB="${TIP_TARGET_WRIST_MASK_PROB:-}" \
    --env TIP_TARGET_TF_GATE_MM="${TIP_TARGET_TF_GATE_MM:-}" \
    --env TIP_TARGET_TF_RAMP_STEPS="${TIP_TARGET_TF_RAMP_STEPS:-}" \
    --env TIP_TARGET_TF_MAX_PRED_PROB="${TIP_TARGET_TF_MAX_PRED_PROB:-}" \
    --env TIP_TARGET_TF_GATE_PATIENCE="${TIP_TARGET_TF_GATE_PATIENCE:-}" \
    --env TIP_TARGET_TF_GATE_EMA="${TIP_TARGET_TF_GATE_EMA:-}" \
    --env NUM_EPOCHS="${NUM_EPOCHS:-}" \
    --env MAX_STEPS="${MAX_STEPS:-}" \
    --env SAVE_STEPS="${SAVE_STEPS:-}" \
    --env IMAGE_TARGET_SIZE="${IMAGE_TARGET_SIZE:-}" \
    --env IMAGE_CROP_SIZE="${IMAGE_CROP_SIZE:-}" \
    --env DATASET_DIR="${DATASET_DIR:-./$DATASET_NAME/}" \
    --env MODALITY_CONFIG="$MODALITY_CONFIG" \
    --env USE_TIP_TARGET_HEAD="${USE_TIP_TARGET_HEAD:-}" \
    --env GR00T_VIDEO_DELTA="${GR00T_VIDEO_DELTA:-}" \
    --env GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}" \
    --env GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}" \
    "$TMPDIR/groot-environment.sif" \
    bash /workspace/setup_and_train.sh

echo "[euler] Job completed at $(date)"
