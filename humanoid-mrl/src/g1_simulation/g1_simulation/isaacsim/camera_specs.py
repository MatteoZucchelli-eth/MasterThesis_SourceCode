# MIT License
"""Simulated camera specs (from the real hardware's camera_info), importable without
Isaac.

Consumed by :func:`.camera_graph.build_camera_graph`.
"""
from __future__ import annotations

# Render product size [px], matching the hardware (Isaac derives vFOV from the
# aspect ratio). The policy's square crop is cut downstream by g1_common.camera.
HAND_CAMERA_SIZE_PX = (1280, 720)  # See3CAM_24CUG on the drill mount, 16:9
HEAD_CAMERA_SIZE_PX = (640, 480)  # d435 colour as the driver publishes it, 4:3

# Focal length [px] from the real camera_info
# (`ros2 topic echo <cam>/color/camera_info --once`).
HAND_CAMERA_FX_PX = 690.1127251872208  # See3CAM, hand_camera_info.yaml (calibrated)
HEAD_CAMERA_FX_PX = 606.370361328125  # d435 factory intrinsics at 640x480

# Principal point [px] from the same camera_info (off-centre on real lenses).
HAND_CAMERA_CENTRE_PX = (639.8322628786954, 369.7137298773904)
HEAD_CAMERA_CENTRE_PX = (308.32464599609375, 246.02682495117188)


def policy_crop_side(size_px: tuple[int, int]) -> int:
    """Side of the square frame a policy is actually shown, for a raw camera size.

    Recorder and controller take a centred ``min(width, height)`` crop, so the
    head's 640x480 reaches the policy as 480x480.
    """
    return min(int(size_px[0]), int(size_px[1]))


def policy_crop_origin_x(size_px: tuple[int, int]) -> float:
    """Left edge of that centred crop in raw pixels — the shift applied to cx."""
    return (int(size_px[0]) - policy_crop_side(size_px)) / 2.0
