#!/bin/bash

set -euo pipefail

DOCS_DIR=${ROS2_WS}/docs

sphinx-build -c "${DOCS_DIR}" -b html "${DOCS_DIR}" "${DOCS_DIR}/html"
chown 1000:1000 -R "${DOCS_DIR}/html"
