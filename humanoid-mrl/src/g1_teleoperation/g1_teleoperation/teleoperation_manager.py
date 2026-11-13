# MIT License
"""Facade for VR (Quest) teleoperation: arm tracking and twist mapping."""
from __future__ import annotations

import time

import numpy as np

from .arm_tracking import ArmTracker
from .xr_unix_receiver import XRUnixReceiver


class TeleoperationManager:
    def __init__(self):
        self._receiver = XRUnixReceiver("/xr_socks/xr_pose.sock")
        self._receiver.start()

        self._max_command_age_sec = 1.0

        self._tracker = ArmTracker()

        # Arm-tracking state (toggled by the left-X button)
        self._arm_tracking_active = False
        self._use_default_pose = True

        # Button rising-edge detection
        self._prev_left_x = 0
        self._prev_left_y = 0

    # ── Public interface (used by g1_control) ─────────────────────────────────

    @property
    def use_default_pose(self) -> bool:
        """True when the controller should drive upper-body to the default pose.
        False when arm tracking was deactivated via X (hold last position)."""
        return self._use_default_pose

    @property
    def needs_robot_poses(self) -> bool:
        """True after activation until robot palm poses are provided for scaling."""
        return self._tracker.needs_robot_poses

    def set_robot_reference_poses(
        self, left_4x4: np.ndarray, right_4x4: np.ndarray
    ) -> None:
        """Provide the robot palm poses that complete calibration (scale/align)."""
        self._tracker.set_robot_references(left_4x4, right_4x4)

    def reset_arm_tracking(self, use_default_pose: bool = True) -> None:
        """Reset arm-tracking state to inactive.

        use_default_pose=True  — robot moves to the default pose (normal entry / Y button).
        use_default_pose=False — robot holds its current pose (e.g. entering from VLA).
        """
        self._arm_tracking_active = False
        self._use_default_pose = use_default_pose
        self._tracker.reset()

    def get_teleop_commands(self):
        """Return (base twist, palm-tracking dict or None) for this control tick."""
        payload = self._receiver.get_payload()
        if payload is None:
            return np.zeros(3), None

        if "timestamp" in payload:
            if time.time() - payload["timestamp"] > self._max_command_age_sec:
                return np.zeros(3), None

        self._update_arm_tracking(payload)

        palm_tracking = None
        if self._arm_tracking_active:
            palm_tracking = self._tracker.palm_targets(payload)

        return self._twist_from(payload), palm_tracking

    # ── Button handling ───────────────────────────────────────────────────────

    def _update_arm_tracking(self, payload) -> None:
        """Toggle arm tracking on/off with the left-X button (rising-edge).
        Press left-Y to stop and reset arm tracking."""
        left_x = payload.get("left_x", 0)
        left_y = payload.get("left_y", 0)

        if left_x == 1 and self._prev_left_x == 0:
            if not self._arm_tracking_active:
                self._arm_tracking_active = True
                self._use_default_pose = False
                self._tracker.calibrate(payload)
            else:
                self._arm_tracking_active = False
                self._use_default_pose = False  # hold last tracked position
                self._tracker.reset()
                print("Arm tracking deactivated.")

        if left_y == 1 and self._prev_left_y == 0:
            self.reset_arm_tracking()
            print("Arm tracking reset via Y button.")

        self._prev_left_x = left_x
        self._prev_left_y = left_y

    # ── Twist mapping ─────────────────────────────────────────────────────────

    @staticmethod
    def _twist_from(payload) -> np.ndarray:
        """Map triggers/sticks to a base twist [vx, vy, wz]."""
        lin_vel = ((-payload.get("right_trigger")) + payload.get("left_trigger")) / 10
        ang_vel = -1 * payload.get("right_joystick")[0]
        lat_vel = 2 * (payload.get("left_squeeze") - payload.get("right_squeeze"))
        return np.array([0.5 * lin_vel, 0.25 * lat_vel, 2 * ang_vel])
