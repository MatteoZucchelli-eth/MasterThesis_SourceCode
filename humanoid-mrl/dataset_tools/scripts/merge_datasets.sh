#!/usr/bin/env bash
# Merge worker datasets and reindex their episodes, frames, tasks, and videos.
#   dataset_tools/scripts/merge_datasets.sh --out <output-root> <worker-root>...
set -euo pipefail

REQUIRE="pandas" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

exec python -m dataset_tools.merge_datasets "$@"
