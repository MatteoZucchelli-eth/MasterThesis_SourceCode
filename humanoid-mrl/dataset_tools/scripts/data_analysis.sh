#!/usr/bin/env bash
# Run trajectory, target-overlay, and synchronization checks on one sample.
# Usage:
#   dataset_tools/scripts/data_analysis.sh <dataset> [out-dir]
# Environment: COUNT, SEED, STRIDE, SYNC_ALL=1.
set -euo pipefail

DATASET="${1:?usage: dataset_tools/scripts/data_analysis.sh <dataset> [out-dir]}"
DATASET="${DATASET%/}"
[ -d "$DATASET/data" ] || { echo "not a dataset (no data/): $DATASET" >&2; exit 1; }
DATASET="$(cd "$DATASET" && pwd)"

REQUIRE="pandas matplotlib cv2 pinocchio" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

OUT="${2:-$REPO/dataset_checks/$(basename "$DATASET")}"
COUNT="${COUNT:-10}"
STRIDE="${STRIDE:-60}"

SEED="${SEED:-$RANDOM$RANDOM}"
EPISODES="$(python - "$DATASET" "$COUNT" "$SEED" <<'PY'
import pathlib, random, sys
dataset, count, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
idx = sorted(int(f.stem.split("_")[1]) for f in dataset.glob("data/chunk-*/episode_*.parquet"))
if not idx:
    sys.exit(f"no episode parquets under {dataset / 'data'}")
print(" ".join(str(i) for i in sorted(random.Random(seed).sample(idx, min(count, len(idx))))))
PY
)"

ALL_EPISODES="$(python - "$DATASET" <<'PY'
import pathlib, sys
print(" ".join(str(int(f.stem.split("_")[1]))
               for f in sorted(pathlib.Path(sys.argv[1]).glob("data/chunk-*/episode_*.parquet"))))
PY
)"
SYNC_EPISODES="$EPISODES"
[ "${SYNC_ALL:-0}" = "1" ] && SYNC_EPISODES="$ALL_EPISODES"

mkdir -p "$OUT/trajectories" "$OUT/overlay"

{
    echo "dataset  : $DATASET"
    echo "episodes : $EPISODES"
    echo "sampled  : $COUNT of $(echo "$ALL_EPISODES" | wc -w), seed $SEED"
    echo "stride   : $STRIDE"
    echo "commit   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "date     : $(date -Is)"
} | tee "$OUT/manifest.txt"

echo
echo "=== trajectories -> $OUT/trajectories ==="
python dataset_tools/visualize_episodes.py \
    --dataset "$DATASET" --episodes $EPISODES --out "$OUT/trajectories" \
    2>&1 | tee "$OUT/trajectories.txt"

echo
echo "=== overlay -> $OUT/overlay ==="
python -m dataset_tools.heatmap_labels.overlay_check \
    --dataset "$DATASET" --episodes $EPISODES --stride "$STRIDE" --out "$OUT/overlay" \
    2>&1 | tee "$OUT/overlay.txt"

echo
echo "=== sync check -> $OUT/sync_check.txt ==="
python -m dataset_tools.heatmap_labels.sync_check \
    --dataset "$DATASET" --episodes $SYNC_EPISODES \
    2>&1 | tee "$OUT/sync_check.txt"

echo
echo "Done. $(find "$OUT" -name '*.png' | wc -l) images under $OUT"
echo "  trajectories/  $(find "$OUT/trajectories" -name '*.png' | wc -l) figures"
echo "  overlay/       $(find "$OUT/overlay" -name '*.png' | wc -l) frames"
