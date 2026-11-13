#!/bin/bash
# shellcheck source=/dev/null
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# shellcheck source=/dev/null
source "/opt/humanoid-mrl-msgs/setup.bash"

if [[ "$1" == "--source-only" ]]; then
    return 0 2>/dev/null || exit 0
fi

exec "$@"
