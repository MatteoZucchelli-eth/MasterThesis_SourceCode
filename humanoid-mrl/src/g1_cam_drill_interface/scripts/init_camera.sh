#!/bin/bash

sleep 5

# Device is passed in from the launch file
DEVICE="${1:-/dev/video0}"

# If one would like to use auto exposure use:
# v4l2-ctl -d "$DEVICE" --set-ctrl=auto_exposure=0 # Auto mode

# If one would like to use manual exposure use:
v4l2-ctl -d "$DEVICE" --set-ctrl=auto_exposure=1 # Manual mode

# Units of 100 us (30 ms). Must stay below the 33.3 ms frame period of
# framerate: 30.0 in hand_camera.yaml, or the camera drops its frame rate
# (verify with `ros2 topic hz /camera/hand/color/image_raw`). For more
# brightness prefer raising `gain` (1..40, currently 1).
v4l2-ctl -d "$DEVICE" --set-ctrl=exposure_time_absolute=300


echo "Camera parameters finally set to: "
v4l2-ctl -d "$DEVICE" --list-ctrls
