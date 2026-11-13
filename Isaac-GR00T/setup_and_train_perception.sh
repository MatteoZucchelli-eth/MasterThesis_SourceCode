#!/bin/bash
# =============================================================================
# setup_and_train_perception.sh — One-shot script for the fine-approach
# PERCEPTION finetune (VLM backbone + target heatmap head).
# Works in both Docker and Apptainer (Euler cluster).
# Handles: .env loading, dataset verification (incl. heatmap_labels/), training.
#
# Usage:
#   ./setup_and_train_perception.sh
#
# Prerequisites:
#   - .env must exist (or .env.sample copied and edited)
#   - dataset bind-mounted at /workspace/g1_dataset_small_general, INCLUDING the heatmap_labels/
#     sidecar directory — training filters to fine-phase frames and will abort
#     without it.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

# --- 1. Source environment variables ---
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "[setup] Loading .env"
    set -a
    source "$PROJECT_DIR/.env"
    set +a
else
    echo "[ERROR] .env not found at $PROJECT_DIR/.env"
    echo "        Copy .env.sample to .env and fill in your tokens."
    exit 1
fi

: "${GROOT_HOME:?GROOT_HOME not set in .env}"
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
echo "[setup] GROOT_HOME=$GROOT_HOME"

# --- 2. Create directory structure ---
mkdir -p "$GROOT_HOME/cache/checkpoints/groot"
mkdir -p "$GROOT_HOME/runs"

# --- 3. Verify dataset (bind-mounted from host as /workspace/g1_dataset_small_general) ---
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/g1_dataset_small_general}"
export DATASET_DIR  # start_finetune_perception.sh honours it

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
    echo "[ERROR] Dataset not found at $DATASET_DIR"
    echo "        Ensure the dataset is bind-mounted into the container at $DATASET_DIR."
    exit 1
fi
if [ ! -d "$DATASET_DIR/heatmap_labels" ]; then
    echo "[ERROR] $DATASET_DIR/heatmap_labels not found."
    echo "        The perception finetune needs the heatmap label sidecar; rsync the"
    echo "        dataset INCLUDING heatmap_labels/ (see submit_perception_euler.sh header)."
    exit 1
fi
N_LABELS=$(find "$DATASET_DIR/heatmap_labels" -name 'episode_*.parquet' | wc -l)
echo "[setup] Dataset found at $DATASET_DIR (${N_LABELS} heatmap label files)"

# --- 4. Launch training ---
echo "[setup] All prerequisites ready. Launching perception training..."

exec bash "$PROJECT_DIR/start_finetune_perception.sh"
