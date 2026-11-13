#!/usr/bin/env bash
# Recompute the thesis success tables from recorded evaluation runs.
#
#   scripts/thesis_results.sh                 # the five systems of the thesis
#   scripts/thesis_results.sh <metrics-dir>   # any run of scripts/evaluate_vla.sh
#
# A task succeeds at lateral threshold tau when, for 2 s without a break, the
# lateral error is below tau, |depth error| <= 10 mm and the drill tilt < 20 deg.
# Python standard library only: no ROS, no GPU, no container.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUCCESS="${REPO_ROOT}/src/g1_control/g1_control/vla_success.py"
METRICS="${METRICS:-${REPO_ROOT}/metrics}"

count() {  # count <dir> <lateral-mm>  ->  "k/N"
  python3 "${SUCCESS}" "$1" --lateral-mm "$2" --depth-mm 10 --perp-deg 20 \
    --hold-s 2.0 | tail -1 | cut -d' ' -f1
}

row() {  # row <label> <contact> <dir>
  if [[ ! -d "$3" ]]; then
    printf '%-14s %-8s  (missing: %s)\n' "$1" "$2" "$3"
    return
  fi
  printf '%-14s %-8s %7s %7s %7s\n' "$1" "$2" \
    "$(count "$3" 10)" "$(count "$3" 5)" "$(count "$3" 2.5)"
}

printf '%-14s %-8s %7s %7s %7s\n' System Contact 10mm 5mm 2.5mm
if [[ $# -gt 0 ]]; then
  for dir in "$@"; do row "$(basename "${dir%/}")" - "${dir}"; done
  exit 0
fi
row Planner     on  "${METRICS}/split_vanilla_contact_120s"
row TTH         on  "${METRICS}/tiptarget_contact_120s"
# Run under vla_planner, but the handover never fired, so the TTH VLA drove the
# whole rollout: equivalent to ROLLOUT_STATE=vla.
row TTH         off "${METRICS}/agreemenT_NOCONTACT_splitdetector"
row Vanilla-480 off "${METRICS}/vanilla_vla_nocontact_120s"
row Vanilla-256 off "${METRICS}/vanilla256_nocontact_120s"
