#!/usr/bin/env bash
# Spawn / manage headless AUTO_COLLECT data-collection workers.
#
# Each worker is one docker compose project (collect-w<i>) built from
# docker/docker-compose.collect.yaml: a headless Isaac Sim plus the control +
# recorder stack, isolated by ROS_DOMAIN_ID and writing to its own dataset
# directory g1_dataset_worker_<i> at the repo root. Workers loop forever;
# stop them when you have enough episodes.
#
# Usage:
#   scripts/collect_workers.sh start <N>          # start workers 1..N (+ watchdog)
#   scripts/collect_workers.sh stop               # stop every collect worker
#   scripts/collect_workers.sh status             # container state + episode counts
#   scripts/collect_workers.sh logs <i> [service] # follow worker i's logs;
#                                                 # service: sim | stack (default: both)
#   scripts/collect_workers.sh watchdog           # run the RAM watchdog in the
#                                                 # foreground (start spawns it
#                                                 # in the background for you)
#
# Tunables (env):
#   DOMAIN_BASE  first ROS_DOMAIN_ID; worker i uses DOMAIN_BASE+i   (default 40)
#   SIM_CAMERA_PUBLISH_HZ  camera/render rate per sim               (default 30)
#   SIM_MEM_LIMIT / STACK_MEM_LIMIT  hard RAM cap per sim / stack
#                container (defaults 20g / 6g, see the compose file). Keep
#                N * (sum of both) below host RAM.
#   WATCHDOG_MIN_AVAIL_GB  emergency-stop every worker when host
#                MemAvailable stays below this (default 4)
#   SIM_GUI=1    run Isaac Sim with its window so you can watch the robot
#                (usually with a single worker: SIM_GUI=1 ... start 1).
#                Needs a host X server; if the window doesn't appear, run
#                `xhost +local:docker` first. Headless is the default.
#   DRILL_CONTACT=1  keep the drill bit's collider on, so the tip actually
#                touches the wall instead of passing through it. Same shorthand
#                the Makefile has; it prepends --cfg.no-disable-drill-bit-collision
#                to SIM_EXTRA_ARGS below (see config.py's
#                disable_drill_bit_collision).
#   SIM_EXTRA_ARGS  raw argv appended to the sim, for flags with no shorthand.
#                Composes with DRILL_CONTACT, which prepends to it.
#
# The state machine / controller / recorder logs are the `stack` service
# (`logs <i> stack`); Isaac Sim's own output is the `sim` service. The
# watchdog logs to logs/collect_watchdog.log.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker/docker-compose.collect.yaml"
DOMAIN_BASE="${DOMAIN_BASE:-40}"
SIM_CAMERA_PUBLISH_HZ="${SIM_CAMERA_PUBLISH_HZ:-30.0}"
WATCHDOG_MIN_AVAIL_GB="${WATCHDOG_MIN_AVAIL_GB:-4}"
WATCHDOG_PIDFILE="${REPO_ROOT}/logs/collect_watchdog.pid"
WATCHDOG_LOG="${REPO_ROOT}/logs/collect_watchdog.log"
# Caller's sim argv, read once so per-worker exports never stack DRILL_CONTACT's flag.
SIM_EXTRA_ARGS_BASE="${SIM_EXTRA_ARGS:-}"
if [[ "${DRILL_CONTACT:-0}" == "1" ]]; then
  SIM_EXTRA_ARGS_BASE="--cfg.no-disable-drill-bit-collision ${SIM_EXTRA_ARGS_BASE}"
fi

project_name() { echo "collect-w$1"; }

compose() {
  local worker=$1
  shift
  docker compose -p "$(project_name "${worker}")" -f "${COMPOSE_FILE}" "$@"
}

worker_env() {
  local worker=$1
  export ROS_DOMAIN_ID=$((DOMAIN_BASE + worker))
  export DATASET_PATH="/workspace/g1_dataset_worker_${worker}"
  export SIM_CAMERA_PUBLISH_HZ
  if [[ "${SIM_GUI:-0}" == "1" ]]; then
    export SIM_GUI_FLAG="--cfg.gui"
  else
    export SIM_GUI_FLAG="--cfg.no-gui"
  fi
  # Must be exported: the compose file interpolates it from the environment.
  export SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS_BASE}"
}

build_workspace() {
  # The stack bind-mounts the host checkout (incl. install/); build it once so
  # workers never race on the same install tree.
  echo "Building the mounted ROS workspace once..."
  compose 1 run --rm --no-deps stack /bin/bash -lc '
    source /root/env.bash
    colcon build --parallel-workers 2 \
      --packages-up-to g1_control g1_state_machine g1_teleoperation
  '
}

list_workers() {
  docker compose ls --all --format json 2>/dev/null |
    python3 -c '
import json, sys
data = json.load(sys.stdin)
names = sorted(
    p["Name"] for p in data if p["Name"].startswith("collect-w")
)
print("\n".join(names))
'
}

mem_available_gb() {
  awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo
}

watchdog_running() {
  [[ -f "${WATCHDOG_PIDFILE}" ]] && kill -0 "$(cat "${WATCHDOG_PIDFILE}")" 2>/dev/null
}

start_watchdog() {
  if watchdog_running; then
    echo "Watchdog already running (pid $(cat "${WATCHDOG_PIDFILE}"))."
    return
  fi
  mkdir -p "$(dirname "${WATCHDOG_LOG}")"
  nohup "${BASH_SOURCE[0]}" watchdog >>"${WATCHDOG_LOG}" 2>&1 &
  echo $! >"${WATCHDOG_PIDFILE}"
  echo "Watchdog started (pid $!, kills all workers if MemAvailable < ${WATCHDOG_MIN_AVAIL_GB}G; log: ${WATCHDOG_LOG})"
}

stop_watchdog() {
  if watchdog_running; then
    kill "$(cat "${WATCHDOG_PIDFILE}")" 2>/dev/null || true
    echo "Watchdog stopped."
  fi
  rm -f "${WATCHDOG_PIDFILE}"
}

# Last-resort brake: hard-kill every collect container when host RAM runs out.
# Complements the per-container mem_limit for the aggregate case.
emergency_stop() {
  echo "[$(date -Is)] EMERGENCY STOP: MemAvailable below ${WATCHDOG_MIN_AVAIL_GB}G — killing all collect workers"
  local ids
  ids=$(docker ps -q --filter "name=collect-w")
  if [[ -n "${ids}" ]]; then
    # Drop the restart policy first or `unless-stopped` resurrects them.
    # shellcheck disable=SC2086
    docker update --restart=no ${ids} || true
    # shellcheck disable=SC2086
    docker kill ${ids} || true
  fi
  local projects
  projects=$(list_workers)
  for p in ${projects}; do
    local i=${p#collect-w}
    worker_env "${i}"
    compose "${i}" down --timeout 5 || true
  done
  echo "[$(date -Is)] Emergency stop complete."
}

cmd_watchdog() {
  echo "[$(date -Is)] Watchdog up: interval 5s, threshold MemAvailable < ${WATCHDOG_MIN_AVAIL_GB}G"
  local breaches=0
  while true; do
    local avail
    avail=$(mem_available_gb)
    if (( avail < WATCHDOG_MIN_AVAIL_GB )); then
      breaches=$((breaches + 1))
      echo "[$(date -Is)] Low memory: ${avail}G available (${breaches}/2)"
    else
      breaches=0
    fi
    # Two consecutive breaches so a transient dip doesn't nuke a whole run.
    if (( breaches >= 2 )); then
      emergency_stop
      rm -f "${WATCHDOG_PIDFILE}"
      exit 0
    fi
    sleep 5
  done
}

cmd_start() {
  local n=${1:?usage: collect_workers.sh start <N>}
  if [[ ! "${n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: worker count must be a positive integer, got '${n}'." >&2
    exit 2
  fi
  if (( DOMAIN_BASE + n > 232 )); then
    echo "error: ROS_DOMAIN_ID must not exceed 232 (DOMAIN_BASE=${DOMAIN_BASE}, workers=${n})." >&2
    exit 2
  fi
  echo "Building images once (shared by all workers)..."
  worker_env 1
  compose 1 build
  build_workspace
  for i in $(seq 1 "${n}"); do
    worker_env "${i}"
    echo "Starting worker ${i} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, dataset ${DATASET_PATH})"
    compose "${i}" up -d
  done
  start_watchdog
  echo "Started ${n} worker(s). Watch with: scripts/collect_workers.sh status"
}

cmd_stop() {
  stop_watchdog
  local projects
  projects=$(list_workers)
  if [[ -z "${projects}" ]]; then
    echo "No collect workers running."
    return
  fi
  for p in ${projects}; do
    local i=${p#collect-w}
    worker_env "${i}"
    echo "Stopping ${p}..."
    compose "${i}" down
  done
}

cmd_status() {
  local projects
  projects=$(list_workers)
  if [[ -z "${projects}" ]]; then
    echo "No collect workers running."
  fi
  for p in ${projects}; do
    local i=${p#collect-w}
    worker_env "${i}"
    echo "── ${p} (ROS_DOMAIN_ID=$((DOMAIN_BASE + i))) ──"
    compose "${i}" ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'
  done
  echo
  if watchdog_running; then
    echo "Watchdog: running (pid $(cat "${WATCHDOG_PIDFILE}"), MemAvailable now $(mem_available_gb)G)"
  else
    echo "Watchdog: NOT running"
  fi
  # Surface containers the kernel OOM-killed (they hit their mem_limit; with
  # restart: unless-stopped they come back up, so this is the only trace).
  local oom
  oom=$(docker ps -a --filter "name=collect-w" --format '{{.Names}}' |
    xargs -r docker inspect --format '{{.Name}} OOMKilled={{.State.OOMKilled}} exit={{.State.ExitCode}}' 2>/dev/null |
    grep 'OOMKilled=true' || true)
  if [[ -n "${oom}" ]]; then
    echo "OOM-killed containers (raise SIM_MEM_LIMIT/STACK_MEM_LIMIT or run fewer workers):"
    echo "${oom}" | sed 's/^/  /'
  fi
  echo
  echo "Episode counts:"
  shopt -s nullglob
  for d in "${REPO_ROOT}"/g1_dataset_worker_*; do
    local count=0
    [[ -f "${d}/meta/episodes.jsonl" ]] && count=$(grep -c . "${d}/meta/episodes.jsonl" || true)
    echo "  $(basename "${d}"): ${count} episodes"
  done
}

cmd_logs() {
  local i=${1:?usage: collect_workers.sh logs <worker-index> [sim|stack]}
  local service=${2:-}
  worker_env "${i}"
  # shellcheck disable=SC2086 — an empty service means "all services".
  compose "${i}" logs -f --tail 100 ${service}
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  watchdog) shift; cmd_watchdog "$@" ;;
  *)
    grep '^#' "$0" | sed -n '2,42p' | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
