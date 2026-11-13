#!/usr/bin/env bash
# Push a dataset to Euler for finetuning.
#   dataset_tools/scripts/sync_to_euler.sh <dataset> [rsync args...]   (-n = dry run)
#
# Lands in /cluster/work/mrl/$USER/datasets/<name>/ ($HOME on Euler is
# quota-limited). Mirrors with --delete.
set -euo pipefail

DATASET="${1:?usage: $0 <dataset> [rsync args...]}"; shift
DATASET="${DATASET%/}"
[ -f "$DATASET/meta/info.json" ] || { echo "not a dataset: $DATASET" >&2; exit 1; }

HOST="${EULER_HOST:-euler}"
REMOTE="${EULER_DATASETS_DIR:-/cluster/work/mrl/${EULER_USER:-$USER}/datasets}/$(basename "$DATASET")"

ssh "$HOST" "mkdir -p '$REMOTE'"
rsync -a --info=progress2 --delete "$@" "$DATASET/" "$HOST:$REMOTE/"
# meta/ changes content without changing size, so re-sync it with --checksum.
rsync -a --checksum "$@" "$DATASET/meta/" "$HOST:$REMOTE/meta/"

echo "done -> $HOST:$REMOTE"