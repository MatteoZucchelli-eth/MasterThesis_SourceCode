#!/usr/bin/env bash
# Run the unit tests of the ROS packages. CPU only: no GPU, no Isaac Sim.
#
#   scripts/run_tests.sh                 # all packages with tests
#   scripts/run_tests.sh g1_control      # selected packages
#
# Runs in the sim-base image (build it once with
# `ROS_DOMAIN_ID=0 DATASET_PATH=unused docker compose -f
# docker/docker-compose.collect.yaml build stack`). The repo
# is mounted read-only and built in a copy inside the container, so nothing on
# the host changes. Lint tests (flake8, pep257, copyright) are left to
# pre-commit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGES="${*:-g1_control g1_common g1_sensors g1_simulation g1_teleoperation g1_cam_drill_interface}"

if ! docker image inspect sim-base >/dev/null 2>&1; then
  echo "Image sim-base not found. Build it first:" >&2
  echo "  ROS_DOMAIN_ID=0 DATASET_PATH=unused docker compose -f docker/docker-compose.collect.yaml build stack" >&2
  exit 1
fi

docker run --rm --network none -v "${REPO_ROOT}:/workspace:ro" \
  -e PACKAGES="${PACKAGES}" sim-base bash -lc '
  set -e
  cp -r /workspace /tmp/ws && cd /tmp/ws && rm -rf build install log
  source /opt/ros/*/setup.bash
  source /root/venv/bin/activate
  echo "Building ${PACKAGES}..."
  colcon build --parallel-workers 4 --packages-up-to ${PACKAGES} >/tmp/build.log 2>&1 \
    || { tail -40 /tmp/build.log; exit 1; }
  source install/setup.bash
  status=0
  for pkg in ${PACKAGES}; do
    echo "=== ${pkg}"
    python -m pytest -q -p no:cacheprovider "src/${pkg}/test" \
      -k "not flake8 and not pep257 and not copyright" || status=1
  done
  exit ${status}
'
