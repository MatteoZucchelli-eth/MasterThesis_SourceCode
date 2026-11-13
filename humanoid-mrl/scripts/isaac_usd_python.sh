#!/usr/bin/env bash
# Run a standalone USD-authoring script under Isaac Sim's python.
#
# Adds the omni.usd.libs extension (pxr package and libusd_*.so) to PYTHONPATH
# and LD_LIBRARY_PATH, which kit only does once an app starts. The extension
# directory is globbed (versioned + hashed). Used by `make sync-cameras`.
#
#     docker compose -f docker/docker-compose.yaml run --rm --no-deps \
#         --entrypoint /workspace/scripts/isaac_usd_python.sh simulation-isaacsim \
#         /workspace/src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py
set -euo pipefail

ext=$(ls -d /isaac-sim/extscache/omni.usd.libs-*/ 2>/dev/null | head -1)
if [[ -z "${ext}" ]]; then
    echo "no omni.usd.libs under /isaac-sim/extscache -- is this the Isaac image?" >&2
    exit 2
fi
ext=${ext%/}

export PYTHONPATH="${ext}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ext}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec /isaac-sim/python.sh "$@"
