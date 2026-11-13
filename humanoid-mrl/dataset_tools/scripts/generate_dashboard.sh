#!/usr/bin/env bash
# Generate aggregate dataset metrics, report, and dashboard.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_DATA_ROOT="${REPO}/data"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: dataset_tools/scripts/generate_dashboard.sh [data-root]"
  echo "Default: ${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
  exit 0
fi
(( $# <= 1 )) || { echo "Usage: dataset_tools/scripts/generate_dashboard.sh [data-root]" >&2; exit 2; }

DATASET_ROOT="${1:-${DATA_ROOT:-${DEFAULT_DATA_ROOT}}}"
TOOL_DIR="${REPO}/data_analysis"
# Generated artefacts live apart from the tool's source, and are gitignored.
OUTPUT_DIR="${TOOL_DIR}/result"

if [[ ! -d "${DATASET_ROOT}/big_targets" && ! -d "${DATASET_ROOT}/small_targets" ]]; then
  echo "Expected big_targets/ or small_targets/ under ${DATASET_ROOT}" >&2
  exit 1
fi

DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
REQUIRE="numpy pyarrow" . "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

mkdir -p "${OUTPUT_DIR}"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

python "${TOOL_DIR}/dq.py" \
  --scan "${DATASET_ROOT}" \
  --out "${TEMP_DIR}/metrics.json" \
  --dashboard "${TEMP_DIR}/dashboard.html" \
  >"${TEMP_DIR}/report.md"

python - "${TEMP_DIR}/metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    if not json.load(handle):
        raise SystemExit("No valid datasets found")
PY

mv "${TEMP_DIR}/metrics.json" "${OUTPUT_DIR}/metrics.json"
mv "${TEMP_DIR}/dashboard.html" "${OUTPUT_DIR}/dashboard.html"
mv "${TEMP_DIR}/report.md" "${OUTPUT_DIR}/report.md"

echo "Dashboard: ${OUTPUT_DIR}/dashboard.html"
echo "Report:    ${OUTPUT_DIR}/report.md"

if [[ "${OPEN_DASHBOARD:-1}" != "0" ]]; then
  if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
    xdg-open "${OUTPUT_DIR}/dashboard.html" >/dev/null 2>&1 &
  elif command -v open >/dev/null; then
    open "${OUTPUT_DIR}/dashboard.html"
  fi
fi
