#!/bin/bash

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# shellcheck source=/dev/null
source "/opt/ros/jazzy/setup.bash"

DEP_WS_SETUP_BASH="${ROS2_WS_DEP}/install/setup.bash"
if [[ -f $DEP_WS_SETUP_BASH ]]; then
    # shellcheck source=/dev/null
    source "$DEP_WS_SETUP_BASH"
else
    echo "File '${DEP_WS_SETUP_BASH}' not available, not sourcing"
fi

WS_SETUP_BASH="${ROS2_WS}/install/setup.bash"
if [[ -f $WS_SETUP_BASH ]]; then
    # shellcheck source=/dev/null
    source "$WS_SETUP_BASH"
else
    echo "File '${WS_SETUP_BASH}' not available, not sourcing"
fi
