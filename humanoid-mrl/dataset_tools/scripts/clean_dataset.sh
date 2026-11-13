#!/usr/bin/env bash
# Delete slip episodes and episodes with an all-black camera stream.
#   dataset_tools/scripts/clean_dataset.sh <dataset> [--dry-run|--yes|...]
set -euo pipefail

DATASET="${1:?usage: dataset_tools/scripts/clean_dataset.sh <dataset> [args...]}"; shift
[ -f "$DATASET/meta/info.json" ] || { echo "not a dataset: $DATASET" >&2; exit 1; }
DATASET="$(cd "$DATASET" && pwd)"

REQUIRE="cv2 pyarrow" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

exec python -m dataset_tools.clean_dataset \
    "$DATASET" --report "$DATASET/rejected_episodes.csv" "$@"
