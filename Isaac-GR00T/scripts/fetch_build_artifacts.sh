#!/bin/bash
# fetch_build_artifacts.sh — Fetch the build inputs that .gitignore keeps out of this repo.
#
# `uv.lock` (.gitignore) and `wheels/` (.gitignore) are not tracked here, but the
# Dockerfiles need them: `uv sync --frozen` requires a lockfile, and the aarch64
# torchcodec/flash-attn wheels have no PyPI equivalent. Upstream NVIDIA/Isaac-GR00T
# force-adds them, so we pull them from the release this tree was branched from.
#
# Idempotent: files already on disk are left alone. Safe to call on every build.
#
# Usage:
#   scripts/fetch_build_artifacts.sh [--profile=default|orin|spark|thor] [--all-wheels]
#
# Env overrides:
#   GROOT_UPSTREAM_URL  upstream remote (default: https://github.com/NVIDIA/Isaac-GR00T.git)
#   GROOT_UPSTREAM_REF  commit to pull from (default: the n1.7-release commit)
#   GROOT_CACHE_DIR     where the upstream mirror is cached (default: ~/.cache/gr00t)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

UPSTREAM_URL="${GROOT_UPSTREAM_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
# Pinned to the commit tagged n1.7-release. Every pyproject.toml in this tree is
# byte-identical to that commit, so its lockfiles resolve correctly here. Later
# upstream commits moved to Python 3.12 / torch 2.9 and are NOT compatible.
UPSTREAM_REF="${GROOT_UPSTREAM_REF:-23ace64f17aa5015259b8609d371eb61a357c776}"
CACHE_DIR="${GROOT_CACHE_DIR:-$HOME/.cache/gr00t}"
MIRROR_DIR="$CACHE_DIR/upstream-isaac-gr00t"

profile="default"
all_wheels=0
for arg in "$@"; do
    case $arg in
        --profile=*) profile="${arg#--profile=}" ;;
        --all-wheels) all_wheels=1 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "fetch_build_artifacts.sh: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Artifact manifest
# ──────────────────────────────────────────────────────────────────────────────
# One entry per line: "<repo-relative path>|<expected sha256, or - for text files>"
# The sha256 values are the git-lfs object ids recorded at UPSTREAM_REF.

WHEELS_DIR="scripts/deployment"

artifacts_default=(
    "uv.lock|-"
    "$WHEELS_DIR/dgpu/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl|69ad4512d5ddc3fdde44623f095ce60098a72d65216c7e4b010197a34b213e05"
)
# 406 MB, and only ever installed on an aarch64 host (GB200/Grace Hopper). With a
# lockfile present, an x86_64 `uv sync --frozen` never materializes it, so it is
# skipped unless the host needs it or --all-wheels is passed.
artifacts_default_aarch64=(
    "$WHEELS_DIR/dgpu/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl|7127cf58a7642d7350527a57daf14b8d1a8301ccd2805eaea1897f5e85535f30"
)
artifacts_orin=(
    "$WHEELS_DIR/orin/uv.lock|-"
    "$WHEELS_DIR/orin/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl|74f85bfa1e30610a3c314fe39bca71bb2f0285e8e7308c60176219a6a3e3bf1a"
)
# Thor reuses the Spark wheels (see scripts/deployment/thor/install_deps.sh).
artifacts_spark_wheels=(
    "$WHEELS_DIR/spark/wheels/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl|2777f80d348e213ae6cd65f288485189cb49e1a2ff7bfbceeb47cdef43301e2b"
    "$WHEELS_DIR/spark/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl|4dcd2ce933cfdbbeeda0bb4d35cece373035bdf341f0c58c1176c36f71e53954"
)

wanted=()
case "$profile" in
    default)
        wanted=("${artifacts_default[@]}")
        if [ "$all_wheels" -eq 1 ] || [ "$(uname -m)" = "aarch64" ]; then
            wanted+=("${artifacts_default_aarch64[@]}")
        fi
        ;;
    orin)  wanted=("${artifacts_orin[@]}") ;;
    spark) wanted=("$WHEELS_DIR/spark/uv.lock|-" "${artifacts_spark_wheels[@]}") ;;
    thor)  wanted=("$WHEELS_DIR/thor/uv.lock|-" "${artifacts_spark_wheels[@]}") ;;
    all)
        wanted=("${artifacts_default[@]}" "${artifacts_default_aarch64[@]}"
                "${artifacts_orin[@]}"
                "$WHEELS_DIR/spark/uv.lock|-" "$WHEELS_DIR/thor/uv.lock|-"
                "${artifacts_spark_wheels[@]}")
        ;;
    *) echo "fetch_build_artifacts.sh: unknown profile '$profile'" >&2; exit 2 ;;
esac

# ──────────────────────────────────────────────────────────────────────────────
# Anything already on disk needs no work
# ──────────────────────────────────────────────────────────────────────────────

missing=()
for entry in "${wanted[@]}"; do
    [ -s "$REPO_ROOT/${entry%%|*}" ] || missing+=("$entry")
done

if [ ${#missing[@]} -eq 0 ]; then
    echo "Build artifacts already present (profile: $profile) — nothing to fetch."
    exit 0
fi

echo "Missing ${#missing[@]} build artifact(s) for profile '$profile'; fetching from upstream."

for tool in git git-lfs; do
    if ! command -v "$tool" &>/dev/null; then
        echo "Error: $tool is required to fetch build artifacts but is not installed." >&2
        echo "  Install with: sudo apt-get install -y $tool" >&2
        exit 1
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Cached upstream mirror (blobs fetched on demand, nothing checked out)
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$CACHE_DIR"
if [ ! -d "$MIRROR_DIR" ]; then
    echo "Cloning upstream metadata into $MIRROR_DIR (no blobs, no checkout)..."
    git clone --filter=blob:none --no-checkout "$UPSTREAM_URL" "$MIRROR_DIR"
fi

if ! git -C "$MIRROR_DIR" cat-file -e "${UPSTREAM_REF}^{commit}" 2>/dev/null; then
    # Cached mirror predates the pinned commit — refresh all refs, then try the
    # bare SHA (allowed only if the server permits reachable-SHA1 fetches).
    echo "Fetching $UPSTREAM_REF from $UPSTREAM_URL..."
    git -C "$MIRROR_DIR" fetch --filter=blob:none --tags origin \
        || git -C "$MIRROR_DIR" fetch --filter=blob:none origin "$UPSTREAM_REF"
    git -C "$MIRROR_DIR" cat-file -e "${UPSTREAM_REF}^{commit}" 2>/dev/null || {
        echo "Error: commit $UPSTREAM_REF not found in $UPSTREAM_URL" >&2
        exit 1
    }
fi

# ──────────────────────────────────────────────────────────────────────────────
# Extract
# ──────────────────────────────────────────────────────────────────────────────

for entry in "${missing[@]}"; do
    path="${entry%%|*}"
    want_sha="${entry##*|}"
    dest="$REPO_ROOT/$path"
    tmp="$dest.tmp.$$"

    mkdir -p "$(dirname "$dest")"
    echo "  -> $path"

    if [ "$want_sha" = "-" ]; then
        git -C "$MIRROR_DIR" show "$UPSTREAM_REF:$path" > "$tmp"
    else
        # Stored via git-lfs upstream: `show` yields a pointer, `smudge` resolves it.
        git -C "$MIRROR_DIR" show "$UPSTREAM_REF:$path" \
            | git -C "$MIRROR_DIR" lfs smudge > "$tmp"

        got_sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
        if [ "$got_sha" != "$want_sha" ]; then
            rm -f "$tmp"
            echo "Error: checksum mismatch for $path" >&2
            echo "  expected $want_sha" >&2
            echo "  got      $got_sha" >&2
            exit 1
        fi
    fi

    [ -s "$tmp" ] || { rm -f "$tmp"; echo "Error: fetched $path is empty" >&2; exit 1; }
    mv "$tmp" "$dest"
done

echo "Build artifacts ready (profile: $profile)."
