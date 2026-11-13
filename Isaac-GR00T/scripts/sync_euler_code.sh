#!/bin/bash
# =============================================================================
# sync_euler_code.sh — push this working tree to ETH Euler.
#
# The counterpart of scripts/fetch_euler_checkpoint.sh, and the executable form
# of the rsync recipe in the submit_euler.sh header.
#
# Usage:
#   scripts/sync_euler_code.sh [--dry-run] [-y] [--no-delete]
#
# WHY --delete IS THE DEFAULT: the sync excludes .git, so the remote copy has no
# way to tell it is behind. Without --delete a file you deleted (or a chunk of
# uncommitted work you later dropped) lives on at ~/groot forever and runs there
# instead of the code you are looking at. That is exactly how a stale
# _load_heatmap_labels() kept reading a `fine_phase` column that this tree no
# longer writes. Deletions are listed and confirmed before anything is written.
#
# Remote-only paths (logs/, .env, checkpoints, datasets, wandb) are excluded, so
# --delete never touches them.
#
# After each sync the script writes ~/groot/.sync_stamp with the local HEAD sha,
# dirty flag and timestamp, and prints the PREVIOUS stamp first — so you can see
# at a glance how stale the remote was before you overwrote it.
#
# Options / environment:
#   --dry-run       Show what would transfer and what would be deleted; write nothing.
#   -y, --yes       Do not prompt, even when files would be deleted.
#   --no-delete     Transfer only; leave remote-only files in place (NOT advised).
#   EULER_HOST      ssh target (default: euler)
#   EULER_CODE_DIR  remote code dir (default: ~/groot) — must match PROJECT_DIR
#                   in submit_euler.sh, which bind-mounts gr00t/ and scripts/
#                   from it into /workspace.
# =============================================================================

set -euo pipefail

SSH_HOST="${EULER_HOST:-euler}"
REMOTE_DIR="${EULER_CODE_DIR:-~/groot}"

DRY_RUN=0
ASSUME_YES=0
DELETE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        -y|--yes)    ASSUME_YES=1; shift ;;
        --no-delete) DELETE=0; shift ;;
        -h|--help)   sed -n '2,36p' "$0"; exit 0 ;;
        *)           echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Heavy or machine-local paths, and paths that exist only on Euler. Everything
# listed here is protected from --delete on the receiver as well.
EXCLUDES=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='*.egg-info'
    --exclude='.venv'
    --exclude='venv'
    --exclude='.pytest_cache'
    --exclude='checkpoints*'
    --exclude='g1_dataset*'
    --exclude='external_dependencies'
    --exclude='media'
    --exclude='demo_data'
    --exclude='wandb'
    --exclude='vla_eval_results'
    --exclude='scratchpad'
    --exclude='*.gz'
    --exclude='*.sif'
    # ~450 MB of aarch64 flash-attn wheels for Orin/Thor/Spark installs.
    # Euler runs the prebuilt x86 container, so they are dead weight against
    # the ~45 GB $HOME quota.
    --exclude='scripts/deployment/*/wheels'
    # Remote-only, must survive --delete:
    --exclude='logs'      # SLURM writes job output here
    --exclude='.env'      # remote secrets; submit_euler.sh fails without it
)

RSYNC_OPTS=(-avP "${EXCLUDES[@]}")
[ "$DELETE" = "1" ] && RSYNC_OPTS+=(--delete)

echo "[sync] $REPO_ROOT"
echo "[sync]   -> $SSH_HOST:$REMOTE_DIR"

# --- How stale is the remote right now? -------------------------------------
PREV="$(ssh "$SSH_HOST" "cat $REMOTE_DIR/.sync_stamp 2>/dev/null || true")"
if [ -n "$PREV" ]; then
    echo "[sync] remote currently holds: $PREV"
else
    echo "[sync] remote has no .sync_stamp (never synced by this script)"
fi

HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
DIRTY=""
[ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" +dirty"
echo "[sync] sending: $HEAD_SHA$DIRTY"
[ -n "$DIRTY" ] && echo "[sync] NOTE: uncommitted changes are included — that is how untracked work reaches Euler, and how it gets stranded there when you drop it locally."

# --- Preview, and confirm any deletions -------------------------------------
PLAN="$(rsync -n "${RSYNC_OPTS[@]}" ./ "$SSH_HOST:$REMOTE_DIR/")"
DELETIONS="$(printf '%s\n' "$PLAN" | grep '^deleting ' || true)"

if [ -n "$DELETIONS" ]; then
    echo "[sync] these remote files are NOT in your tree and will be DELETED:"
    printf '%s\n' "$DELETIONS" | sed 's/^deleting /[sync]   - /'
fi

if [ "$DRY_RUN" = "1" ]; then
    printf '%s\n' "$PLAN"
    echo "[sync] dry run — nothing written."
    exit 0
fi

if [ -n "$DELETIONS" ] && [ "$ASSUME_YES" = "0" ]; then
    read -r -p "[sync] proceed? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "[sync] aborted."; exit 1 ;;
    esac
fi

# --- Transfer ----------------------------------------------------------------
rsync "${RSYNC_OPTS[@]}" ./ "$SSH_HOST:$REMOTE_DIR/"

# __pycache__ is excluded from the transfer, which also protects it from --delete.
# Left alone, it accumulates .pyc files for modules whose source is long gone (a
# perception_action.pyc outliving perception_action.py, say) and makes a freshly
# synced tree look stale to anyone grepping it. Clear it; the run rebuilds what
# it needs.
ssh "$SSH_HOST" "find $REMOTE_DIR -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true"

STAMP="$HEAD_SHA$DIRTY  synced $(date -u '+%Y-%m-%dT%H:%M:%SZ') from $(hostname)"
ssh "$SSH_HOST" "printf '%s\n' '$STAMP' > $REMOTE_DIR/.sync_stamp"

echo "[sync] done. remote now holds: $STAMP"
echo "[sync] submit with:  ssh $SSH_HOST 'cd ~/groot && sbatch submit_euler.sh'"
