#!/bin/bash
# =============================================================================
# setup_and_train.sh — One-shot script for Gr00t finetuning
# Works in both Docker and Apptainer (Euler cluster).
# Handles: .env loading, checkpoint download, data download, patching, training.
#
# Usage:
#   ./setup_and_train.sh
#
#
# Prerequisites:
#   - /hfm must be a writable directory (bind-mounted in Apptainer/Docker)
#   - .env must exist (or .env.sample copied and edited)
#   - HF_TOKEN must be set for downloading from HuggingFace
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

# Verify critical variables
: "${GROOT_HOME:?GROOT_HOME not set in .env}"
: "${HF_TOKEN:?HF_TOKEN not set in .env}"

echo "[setup] GROOT_HOME=$GROOT_HOME"

# --- 2. Create directory structure ---
echo "[setup] Ensuring directory structure under $GROOT_HOME ..."
mkdir -p "$GROOT_HOME/cache/checkpoints/groot"
mkdir -p "$GROOT_HOME/data/"

# --- 3. Verify dataset is available (bind-mounted from host as /workspace/g1_dataset_small_general) ---
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/g1_dataset_red_2k}"
export DATASET_DIR  # start_finetune.sh honours it
export DATASET_DIR

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
    echo "[ERROR] Dataset not found at $DATASET_DIR"
    echo "        Ensure the dataset is bind-mounted into the container at $DATASET_DIR."
    exit 1
else
    echo "[setup] Dataset found at $DATASET_DIR"
fi
# --- 4. Launch training ---
echo "[setup] All prerequisites ready. Launching training..."

exec bash "$PROJECT_DIR/start_finetune.sh"