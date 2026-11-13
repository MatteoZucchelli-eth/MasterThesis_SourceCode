#!/usr/bin/env bash

set -euo pipefail

# Start the non-interactive support nodes in another terminal first:
# ros2 launch g1_state_machine head_camera_calibration_support.launch.py
#
# At every motionless arm placement press s, collect for a few seconds, then
# press t before moving. Repeat over diverse arm configurations and press e to
# write ~/.ros/head_camera_extrinsics.yaml. The run compares head RGB, hand RGB
# and mocap only.
exec ros2 run g1_sensors head_camera_calibration \
  --ros-args \
  -p tag_size_m:=0.085 \
  -p tag_id:=0 \
  -p wall_frame:=head_calibration_tag
