#!/usr/bin/env bash
# Generate wrist-camera target labels for a LeRobot dataset.
# Usage: dataset_tools/scripts/generate_pixel_labels.sh <dataset> [generator args...]
set -euo pipefail

DATASET="${1:?usage: dataset_tools/scripts/generate_pixel_labels.sh <dataset> [generator args...]}"
shift
[ -f "$DATASET/meta/info.json" ] || { echo "not a dataset: $DATASET" >&2; exit 1; }
DATASET="$(cd "$DATASET" && pwd)"

REQUIRE="cv2 pandas pinocchio" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

exec python -m dataset_tools.heatmap_labels.process_dataset \
    --dataset "$DATASET" "$@"
