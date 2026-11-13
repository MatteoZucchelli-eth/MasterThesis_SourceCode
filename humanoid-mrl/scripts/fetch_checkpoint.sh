#!/usr/bin/env bash
# Pull a trained GR00T checkpoint from Euler into the local groot workspace.
#   scripts/fetch_checkpoint.sh <remote checkpoint dir> [rsync args...]   (-n = dry run)
#   scripts/fetch_checkpoint.sh /cluster/scratch/mzucchelli/groot/runs/<run>/checkpoint-33028
#
# Lands in ~/groot_ws/Isaac-GR00T/checkpoints-<run>-<step>, which is what the
# serve compose file takes as MODEL_PATH.
#
# Allowlist of inference files only; resume-only training state stays on Euler.
set -euo pipefail

REMOTE="${1:?usage: $0 <remote checkpoint dir> [rsync args...]}"; shift
REMOTE="${REMOTE%/}"

STEP="$(basename "$REMOTE")"
RUN="$(basename "$(dirname "$REMOTE")")"
DEST="$HOME/groot_ws/Isaac-GR00T/checkpoints-${RUN}-${STEP#checkpoint-}"

mkdir -p "$DEST"
rsync -a --info=progress2 --partial "$@" \
  --include='/config.json' \
  --include='/embodiment_id.json' \
  --include='/processor_config.json' \
  --include='/statistics.json' \
  --include='/model.safetensors.index.json' \
  --include='/model*.safetensors' \
  --include='/experiment_cfg/' \
  --include='/experiment_cfg/**' \
  --exclude='*' \
  "euler:$REMOTE/" "$DEST/"

echo "done -> $DEST"
echo "serve it with: cd ~/groot_ws/Isaac-GR00T && MODEL_PATH=$(basename "$DEST") docker compose up -d"
