#!/usr/bin/env bash
# Balance the wrist-camera target labels over tip-to-target distance.
# Usage: dataset_tools/scripts/balance_labels.sh <dataset> [balancer args...]
#   --report-only  print the before/after distribution, write nothing
#   --apply        swap the balanced labels in as heatmap_labels/
set -euo pipefail

DATASET="${1:?usage: dataset_tools/scripts/balance_labels.sh <dataset> [balancer args...]}"
shift
[ -d "$DATASET/heatmap_labels" ] || {
    echo "no heatmap_labels/ in $DATASET; run generate_pixel_labels.sh first" >&2
    exit 1
}
DATASET="$(cd "$DATASET" && pwd)"

REQUIRE="numpy pandas" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

exec python -m dataset_tools.heatmap_labels.balance_distance \
    --dataset "$DATASET" "$@"
