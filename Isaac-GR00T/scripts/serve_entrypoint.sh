#!/usr/bin/env bash
# =============================================================================
# serve_entrypoint.sh — pick the correct GR00T eval server for the checkpoint.
#
# Invoked as the container command by docker-compose (see docker-compose.yml).
# Routes to:
#   - gr00t/eval/run_perception_server.py   for a perception checkpoint
#                                            (config.json model_type Gr00tN1d7Perception)
#   - gr00t/eval/run_gr00t_server.py         for everything else (the VLA)
#
# Controlled by environment variables (all set by the Makefile / compose):
#   MODEL_PATH       checkpoint dir or HF id            (default: general_3k)
#   EMBODIMENT_TAG   embodiment tag                     (default: NEW_EMBODIMENT)
#   PORT             server port                        (default: 5555)
#   SERVER           auto | vla | perception            (default: auto)
#                    'auto' inspects <MODEL_PATH>/config.json; override to force.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-general_3k}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
PORT="${PORT:-5555}"
SERVER="${SERVER:-auto}"

detect_server() {
    # Local checkpoints carry a config.json; a perception model_type routes to
    # the perception server. Anything else (incl. HF ids with no local config)
    # falls back to the VLA server.
    local cfg="${MODEL_PATH%/}/config.json"
    if [[ -f "$cfg" ]] && grep -q '"Gr00tN1d7Perception"' "$cfg"; then
        echo perception
    else
        echo vla
    fi
}

if [[ "$SERVER" == "auto" ]]; then
    SERVER="$(detect_server)"
    echo "[serve] auto-detected server type: $SERVER (from ${MODEL_PATH%/}/config.json)"
fi

case "$SERVER" in
    perception)
        echo "[serve] PERCEPTION server | model=$MODEL_PATH | tag=$EMBODIMENT_TAG | port=$PORT"
        exec python gr00t/eval/run_perception_server.py \
            --model-path "$MODEL_PATH" \
            --embodiment-tag "$EMBODIMENT_TAG" \
            --port "$PORT"
        ;;
    vla)
        echo "[serve] VLA server | model=$MODEL_PATH | tag=$EMBODIMENT_TAG | port=$PORT"
        exec python gr00t/eval/run_gr00t_server.py \
            --model-path "$MODEL_PATH" \
            --embodiment-tag "$EMBODIMENT_TAG" \
            --port "$PORT"
        ;;
    *)
        echo "[serve] ERROR: SERVER must be one of auto|vla|perception, got '$SERVER'" >&2
        exit 1
        ;;
esac
