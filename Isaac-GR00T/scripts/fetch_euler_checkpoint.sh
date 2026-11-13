#!/bin/bash
# =============================================================================
# fetch_euler_checkpoint.sh — pull a trained checkpoint off ETH Euler, complete
# enough to rebuild the model exactly as it was trained.
#
# Usage:
#   scripts/fetch_euler_checkpoint.sh <run-name> [dest-dir]
#
# It always fetches the NEWEST weights the run produced, and only those:
#
#   1. the final model at the run root, if the run finished. trainer.save_model()
#      writes it AFTER the last periodic save, so it is strictly the latest.
#   2. otherwise the HIGHEST checkpoint-N (numeric, so 12662 beats 7800).
#
# Earlier checkpoints are never fetched unless you name one with --step.
# Destination defaults to ./checkpoints-<run>-<step>.
#
# Examples:
#   scripts/fetch_euler_checkpoint.sh red2k_mean_512
#   scripts/fetch_euler_checkpoint.sh perception_v2 ./checkpoints-perception-v2
#   scripts/fetch_euler_checkpoint.sh red2k_mean_512 --step 5200      # an older one
#
# Options / environment:
#   --final            Require the final model; fail if the run has not finished.
#   --latest-checkpoint  Take the highest checkpoint-N even when a final model
#                      exists. Use when you need optimizer/RNG state alongside it.
#   --step N           Fetch checkpoint-N specifically.
#   --keep-optimizer   Also download optimizer.pt (only needed to resume LOCALLY;
#                      resuming on Euler does not need a local copy).
#   --dry-run          Show what would transfer without writing anything.
#   EULER_HOST         ssh target (default: euler)
#   EULER_USER         remote user, for the scratch path (default: $USER)
#   EULER_RUNS_DIR     default: /cluster/scratch/$EULER_USER/groot/runs
#
# WHAT IS EXCLUDED, and why it is safe:
#   global_step*/ latest zero_to_fp32.py   DeepSpeed sharded optimizer state
#   optimizer.pt scheduler.pt rng_state_*  training-resume state
#   trainer_state.json training_args.bin   ditto
#   wandb_config.json                      logging metadata
# None of these are needed to load the model for inference or evaluation. What is
# KEPT is what matters: the weights plus config.json, processor_config.json,
# statistics.json, embodiment_id.json and — critically — experiment_cfg/, which
# holds final_processor_config.json. That file is the ONLY reliable record of the
# image geometry the run actually tokenized with; config.json can and does
# disagree with it (it carries fields inherited from the base checkpoint that the
# processor overrode). Never fetch a checkpoint without it.
#
# NOTE: excluding the resume state means the downloaded copy cannot resume
# training. That is deliberate — resuming happens on Euler, against the copy that
# is still there. Pass --keep-optimizer if you really want to resume locally.
# =============================================================================

set -euo pipefail

SSH_HOST="${EULER_HOST:-euler}"
REMOTE_USER="${EULER_USER:-${USER:-$(id -un)}}"
RUNS_DIR="${EULER_RUNS_DIR:-/cluster/scratch/$REMOTE_USER/groot/runs}"

RUN_NAME=""
DEST=""
WANT_FINAL=0
WANT_STEP=""
FORCE_LATEST_CKPT=0
KEEP_OPTIMIZER=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --final)             WANT_FINAL=1; shift ;;
        --latest-checkpoint) FORCE_LATEST_CKPT=1; shift ;;
        --step)              WANT_STEP="$2"; shift 2 ;;
        --keep-optimizer) KEEP_OPTIMIZER=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)           sed -n '2,50p' "$0"; exit 0 ;;
        -*)                  echo "unknown option: $1" >&2; exit 2 ;;
        *)
            if [ -z "$RUN_NAME" ]; then RUN_NAME="$1"
            elif [ -z "$DEST" ]; then DEST="$1"
            else echo "unexpected argument: $1" >&2; exit 2
            fi
            shift ;;
    esac
done

if [ -z "$RUN_NAME" ]; then
    echo "usage: $0 <run-name> [dest-dir] [--final|--step N] [--keep-optimizer] [--dry-run]" >&2
    exit 2
fi

RUN_DIR="$RUNS_DIR/$RUN_NAME"

# --- Resolve what to fetch, entirely from the remote listing -----------------
echo "[fetch] host=$SSH_HOST  run=$RUN_DIR"
if ! ssh "$SSH_HOST" "test -d '$RUN_DIR'"; then
    echo "[fetch] ERROR: no such run directory on $SSH_HOST:" >&2
    echo "[fetch]   $RUN_DIR" >&2
    echo "[fetch] runs that do exist:" >&2
    ssh "$SSH_HOST" "ls -1 '$RUNS_DIR' 2>/dev/null" | sed 's/^/[fetch]   /' >&2 || true
    exit 1
fi

STEPS="$(ssh "$SSH_HOST" "ls -1d '$RUN_DIR'/checkpoint-* 2>/dev/null || true" \
         | sed 's#.*/checkpoint-##' | grep -E '^[0-9]+$' | sort -n || true)"
HIGHEST="$(printf '%s\n' "$STEPS" | tail -1)"
HAS_FINAL="$(ssh "$SSH_HOST" "ls -1 '$RUN_DIR'/*.safetensors 2>/dev/null | head -1 || true")"

if [ -n "$STEPS" ]; then
    echo "[fetch] checkpoints present: $(printf '%s' "$STEPS" | tr '\n' ' ')"
else
    echo "[fetch] checkpoints present: (none)"
fi
[ -n "$HAS_FINAL" ] && echo "[fetch] a final model (trainer.save_model) IS present at the run root"

# Selection order. Only ever the NEWEST weights the run produced: the final model
# when the run finished, otherwise the highest checkpoint-N. Earlier checkpoints are
# reachable only by asking for one explicitly with --step.
SRC_IS_ROOT=0

select_final() {
    SRC="$RUN_DIR/"
    # Gates the checkpoint-*/ and profiling/ excludes below. Must key off the
    # SOURCE, not off --final: the default path lands here too.
    SRC_IS_ROOT=1
    # Name the destination after the step the run ended at, matching the
    # checkpoints-<run>-<step> convention. conf.yaml records it; fall back to
    # "final" if it cannot be read.
    local steps
    steps="$(ssh "$SSH_HOST" \
        "grep -m1 -E '^[[:space:]]*max_steps:' '$RUN_DIR/experiment_cfg/conf.yaml' 2>/dev/null || true" \
        | grep -oE '[0-9]+' | head -1 || true)"
    LABEL="${steps:-final}"
}

if [ -n "$WANT_STEP" ]; then
    if ! printf '%s\n' "$STEPS" | grep -qx "$WANT_STEP"; then
        echo "[fetch] ERROR: checkpoint-$WANT_STEP does not exist on the remote." >&2
        exit 1
    fi
    SRC="$RUN_DIR/checkpoint-$WANT_STEP/"
    LABEL="$WANT_STEP"
    echo "[fetch] selected: checkpoint-$WANT_STEP (explicitly requested)"
elif [ "$WANT_FINAL" = "1" ]; then
    if [ -z "$HAS_FINAL" ]; then
        echo "[fetch] ERROR: --final requested but no *.safetensors at the run root." >&2
        echo "[fetch] The run has not finished. Omit --final to take checkpoint-$HIGHEST." >&2
        exit 1
    fi
    select_final
    echo "[fetch] selected: the FINAL model at the run root (step ${LABEL})"
elif [ "$FORCE_LATEST_CKPT" = "1" ]; then
    if [ -z "$HIGHEST" ]; then
        echo "[fetch] ERROR: --latest-checkpoint requested but no checkpoint-* under $RUN_DIR." >&2
        exit 1
    fi
    SRC="$RUN_DIR/checkpoint-$HIGHEST/"
    LABEL="$HIGHEST"
    echo "[fetch] selected: checkpoint-$HIGHEST (highest; --latest-checkpoint overrode the final model)"
elif [ -n "$HAS_FINAL" ]; then
    # DEFAULT, run finished: save_model() writes the final weights to the run root
    # AFTER the last periodic save, so this is strictly the newest thing there is.
    select_final
    echo "[fetch] selected: the FINAL model at the run root (step ${LABEL}) -- newer than checkpoint-$HIGHEST"
elif [ -n "$HIGHEST" ]; then
    # DEFAULT, run unfinished or killed: newest periodic save.
    SRC="$RUN_DIR/checkpoint-$HIGHEST/"
    LABEL="$HIGHEST"
    echo "[fetch] selected: checkpoint-$HIGHEST (highest; no final model, so the run has not finished)"
else
    echo "[fetch] ERROR: $RUN_DIR holds neither a final model nor any checkpoint-*." >&2
    exit 1
fi

[ -z "$DEST" ] && DEST="./checkpoints-${RUN_NAME}-${LABEL}"

# --- Transfer ---------------------------------------------------------------
EXCLUDES=(
    --exclude='global_step*/'      # DeepSpeed sharded optimizer state
    --exclude='latest'
    --exclude='zero_to_fp32.py'
    --exclude='scheduler.pt'
    --exclude='rng_state_*.pth'
    --exclude='trainer_state.json'
    --exclude='training_args.bin'
    --exclude='wandb_config.json'
)
# The run root also holds subdirectories that are not part of the final model.
[ "$SRC_IS_ROOT" = "1" ] && EXCLUDES+=(--exclude='checkpoint-*/' --exclude='profiling/')
[ "$KEEP_OPTIMIZER" = "1" ] || EXCLUDES+=(--exclude='optimizer.pt')

RSYNC=(rsync -avP "${EXCLUDES[@]}" "$SSH_HOST:$SRC" "$DEST")
[ "$DRY_RUN" = "1" ] && RSYNC=(rsync -avn "${EXCLUDES[@]}" "$SSH_HOST:$SRC" "$DEST")

echo "[fetch] $SSH_HOST:$SRC"
echo "[fetch]   -> $DEST"
# rsync will not create nested parents on its own.
[ "$DRY_RUN" = "1" ] || mkdir -p "$DEST"
"${RSYNC[@]}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[fetch] dry run — nothing written."
    exit 0
fi

# --- Verify the copy is actually rebuildable ---------------------------------
TRUTH="$DEST/experiment_cfg/final_processor_config.json"
if [ ! -f "$TRUTH" ]; then
    echo "[fetch] WARNING: $TRUTH is missing." >&2
    echo "[fetch] Without it there is no reliable record of the image geometry this" >&2
    echo "[fetch] run tokenized with — config.json is NOT a substitute." >&2
    exit 1
fi

echo "[fetch] ---------------------------------------------------------------"
echo "[fetch] EFFECTIVE geometry this checkpoint was trained with"
echo "[fetch] (from experiment_cfg/final_processor_config.json — the ground truth)"
python3 - "$TRUTH" <<'PY'
import json, sys

cfg = json.load(open(sys.argv[1]))


def val(key):
    # vars(processor) was dumped with str(), so everything is a string.
    raw = cfg.get(key, "None")
    return None if raw in ("None", None) else raw


target, crop = val("image_target_size"), val("image_crop_size")
edge, frac = val("shortest_image_edge"), val("crop_fraction")
print(f"[fetch]   image_target_size  = {target}")
print(f"[fetch]   image_crop_size    = {crop}")
print(f"[fetch]   shortest_image_edge= {edge}")
print(f"[fetch]   crop_fraction      = {frac}")

side = None
if edge is not None:
    side = int(float(edge))
elif target is not None:
    side = int(float(target.strip("()[] ").split(",")[0]))
if side:
    # Backbone patches are 16 px with spatial_merge_size 2 -> 32 px per token.
    print(f"[fetch]   => {side}x{side} input, ~{(side // 32) ** 2} visual tokens per image")
if edge is not None or frac is not None:
    print("[fetch]   NB: shortest_image_edge/crop_fraction are set, so they WON over")
    print("[fetch]       image_target_size/image_crop_size. Pre-a66a334 runs look like")
    print("[fetch]       this (256 / 0.95) regardless of what the launch script asked for.")
PY
echo "[fetch] ---------------------------------------------------------------"
echo "[fetch] done: $DEST"
