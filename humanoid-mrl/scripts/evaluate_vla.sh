#!/usr/bin/env bash
# Run every validated Isaac Sim VLA evaluation scenario.
#
# Usage:
#   scripts/evaluate_vla.sh start <metrics-dir>
#   scripts/evaluate_vla.sh <metrics-dir>          # same as start
#   scripts/evaluate_vla.sh status <metrics-dir>
#   scripts/evaluate_vla.sh logs [sim|stack]
#   scripts/evaluate_vla.sh stop
#
# Tunables: SIM_GUI=1 shows Isaac Sim and opens RViz beside it (both headless
# by default; EVAL_RVIZ_CONFIG picks the layout); ROLLOUT_S=60 sets
# policy time per scenario; ROS_DOMAIN_ID=0 and SIM_CAMERA_PUBLISH_HZ=30 by
# default. ROLLOUT_STATE=vla|vla_planner picks the state each rollout runs in
# (vla_planner also needs a detector on 5556), and ENABLE_WALL_PLANE=1 turns
# the sim's head depth back on and fits the plane — required by vla_planner's
# default depth_source: stereo, and useless otherwise. DRILL_CONTACT=1 (the
# default here) keeps the drill bit's collider on, so the tip stops on the wall;
# DRILL_CONTACT=0 takes it off and the bit passes through the wall, matching
# the physics the datasets were collected under.
#
# EVAL_SCENARIO_FILTER picks a subset of the scenario table to run (see below).
# STOP_ON_COMPLETE=0 makes every rollout run the full ROLLOUT_S; by default a
# vla_planner rollout ends as soon as the planner reports it has held the
# finalize pose, so ROLLOUT_S can be budgeted for the slowest attempt.
#
# The model on 5555 is a vanilla action VLA with no perception head, so
# VLA_PLANNER hands over on the detector's first confident sighting and, since
# nothing else can supply a range, needs ENABLE_WALL_PLANE=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.eval.yaml"
PROJECT="vla-eval"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export DATASET_PATH=/tmp/vla-eval-unused
export SIM_CAMERA_PUBLISH_HZ="${SIM_CAMERA_PUBLISH_HZ:-30.0}"
export SIM_GUI_FLAG="--cfg.$([[ "${SIM_GUI:-0}" == 1 ]] && echo gui || echo no-gui)"
# SIM_GUI=1 enables the compose "gui" profile (RViz). Exported so every
# compose subcommand sees the same profiles.
if [[ "${SIM_GUI:-0}" == 1 ]]; then
  export COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}gui"
fi
# Which RViz layout the gui profile opens; config.rviz has the PerceptionOverlay.
export EVAL_RVIZ_CONFIG="${EVAL_RVIZ_CONFIG:-/workspace/config.rviz}"
# debug_viz_node (TF frames + wrist overlay for RViz); defaults to SIM_GUI.
export ENABLE_DEBUG_VIZ="${ENABLE_DEBUG_VIZ:-$([[ "${SIM_GUI:-0}" == 1 ]] && echo true || echo false)}"
# DRILL_CONTACT=1 turns the bit collider on (sim default is off); prepended to
# any explicit SIM_EXTRA_ARGS.
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:-}"
if [[ "${DRILL_CONTACT:-1}" == 1 ]]; then
  SIM_EXTRA_ARGS="--cfg.no-disable-drill-bit-collision ${SIM_EXTRA_ARGS}"
fi
export SIM_EXTRA_ARGS
export ROLLOUT_STATE="${ROLLOUT_STATE:-vla}"
export STOP_ON_COMPLETE="$([[ "${STOP_ON_COMPLETE:-1}" == 1 ]] && echo true || echo false)"
# Run only part of the scenario table, e.g.
#   EVAL_SCENARIO_FILTER="5:red-triangle,6:green-triangle,6:red-triangle"
# Entries are <id> or <id>:<colour>-<shape>; an id alone runs all three of its
# targets. Only the commanded cells are filtered; scenes are unchanged.
export EVAL_SCENARIO_FILTER="${EVAL_SCENARIO_FILTER:-}"
export ENABLE_WALL_PLANE="$([[ "${ENABLE_WALL_PLANE:-0}" == 1 ]] && echo true || echo false)"
# The wall plane needs head depth, which is off by default in evaluation.
if [[ "${ENABLE_WALL_PLANE}" == true ]]; then
  export G1_HEAD_DEPTH_TOPIC="${G1_HEAD_DEPTH_TOPIC:-/camera/d435/depth/image_rect_raw}"
fi

compose() { docker compose -p "${PROJECT}" -f "${COMPOSE_FILE}" "$@"; }

port_is_live() {
  timeout 2 bash -c "</dev/tcp/127.0.0.1/${1}" >/dev/null 2>&1
}

metrics_dir() {
  local requested=${1:?usage: evaluate_vla.sh start <metrics-dir>}
  mkdir -p "${requested}"
  METRICS_DIR="$(cd "${requested}" && pwd)"
  export METRICS_DIR
}

start() {
  if ! port_is_live 5555; then
    echo "No GR00T model is reachable on port 5555." >&2
    echo "Bring up the model, then run this command again." >&2
    exit 1
  fi
  # VLA_PLANNER needs the detector on fine_server_port (host network).
  if [[ "${ROLLOUT_STATE}" == vla_planner ]] && ! port_is_live 5556; then
    echo "ROLLOUT_STATE=vla_planner needs a detector on port 5556." >&2
    exit 1
  fi
  # The driver has no range estimate; the wall plane is the only depth source.
  if [[ "${ROLLOUT_STATE}" == vla_planner && "${ENABLE_WALL_PLANE}" != true ]]; then
    echo "ROLLOUT_STATE=vla_planner needs ENABLE_WALL_PLANE=1: the driver" >&2
    echo "has no perception head, so the wall plane is the only depth source." >&2
    exit 1
  fi
  metrics_dir "$1"
  # Refuse a folder with existing metrics unless EVAL_APPEND=1.
  if [[ -n "$(find "${METRICS_DIR}" -maxdepth 1 -name 'vla_metrics_*.csv' -print -quit)" ]]; then
    if [[ "${EVAL_APPEND:-0}" != 1 ]]; then
      echo "${METRICS_DIR} already holds VLA metrics from an earlier run." >&2
      echo "Use a fresh directory, move those aside, or set EVAL_APPEND=1." >&2
      exit 1
    fi
    echo "EVAL_APPEND=1: adding this run to the metrics already in ${METRICS_DIR}."
  fi
  export ROLLOUT_S="${ROLLOUT_S:-60.0}"
  echo "Building the evaluation images and mounted ROS workspace..."
  compose build
  compose run --rm --no-deps stack /bin/bash -lc '
    source /root/env.bash
    colcon build --parallel-workers 2 --packages-up-to \
      g1_simulation g1_control g1_state_machine
  '
  echo "Starting VLA evaluation in ${ROLLOUT_STATE} (drill" \
    "contact $([[ "${DRILL_CONTACT:-1}" == 1 ]] && echo on || echo off); metrics:" \
    "${METRICS_DIR}, ${ROLLOUT_S}s/scenario)..."
  compose up -d
  echo "Watch with: scripts/evaluate_vla.sh status '${METRICS_DIR}'"
}

status() {
  metrics_dir "$1"
  compose ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'
  if [[ -f "${METRICS_DIR}/evaluation_status.json" ]]; then
    python3 -m json.tool "${METRICS_DIR}/evaluation_status.json"
  else
    echo "Evaluation status: waiting for the stack to start."
  fi
}

case "${1:-}" in
  start) shift; start "$@" ;;
  status) shift; status "$@" ;;
  logs)
    shift
    if [[ -n "${1:-}" ]]; then
      compose logs -f --tail 100 "$1"
    else
      compose logs -f --tail 100
    fi
    ;;
  stop) compose down ;;
  -h|--help|"") sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) start "$@" ;;
esac
