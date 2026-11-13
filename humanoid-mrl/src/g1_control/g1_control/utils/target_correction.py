# MIT License
"""Slew-limited offset between a planned target pose and its live estimate.

The target hangs off ``right_drill_link``, so its lookup carries the FK error at
the configuration it was read in; executing elsewhere misses by
``E(q_execute) - E(q_measure)``. Re-reading every tick cancels that term. This
class holds the correction from the planned target to the live one. It is not a
feedback loop; the rate/magnitude limits only guard against bad frames.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


def _step_toward(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    """Move ``current`` at most ``max_step`` towards ``goal``."""
    delta = goal - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm < 1e-12:
        return goal.copy()
    return current + delta * (max_step / norm)


def _clamp(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``v`` down to ``max_norm`` if it is longer than that."""
    norm = float(np.linalg.norm(v))
    return v.copy() if norm <= max_norm else v * (max_norm / norm)


def _step_toward_rotation(
    current: np.ndarray, goal: np.ndarray, max_step_rad: float
) -> np.ndarray:
    """Rotate ``current`` at most ``max_step_rad`` towards ``goal``."""
    delta = pin.log3(goal @ current.T)
    norm = float(np.linalg.norm(delta))
    if norm <= max_step_rad or norm < 1e-12:
        return goal.copy()
    return pin.exp3(delta * (max_step_rad / norm)) @ current


class TargetCorrection:
    """Rate- and magnitude-limited SE3 offset from a reference target to a live one.

    Offset and poses share a frame (pelvis); the correction is a left-multiplication.
    """

    def __init__(
        self,
        *,
        max_rate_m_s: float,
        max_pos_m: float,
        max_rate_rad_s: float,
        max_ang_rad: float,
    ) -> None:
        self._max_rate_m_s = float(max_rate_m_s)
        self._max_pos_m = float(max_pos_m)
        self._max_rate_rad_s = float(max_rate_rad_s)
        self._max_ang_rad = float(max_ang_rad)
        self._reference: pin.SE3 | None = None
        self.reset(None)

    def reset(self, reference: pin.SE3 | None) -> None:
        """Anchor to the target pose the plan was built from; zero the offset."""
        self._reference = None if reference is None else pin.SE3(reference)
        self._translation = np.zeros(3)
        self._rotation = np.eye(3)
        self._raw_translation = np.zeros(3)
        self._raw_rotation = np.eye(3)

    def update(self, observed: pin.SE3, dt: float) -> None:
        """Fold in the latest target estimate, moving no faster than the limits.

        Not calling it holds the last correction.
        """
        if self._reference is None or dt <= 0.0:
            return

        self._raw_translation = observed.translation - self._reference.translation
        self._raw_rotation = observed.rotation @ self._reference.rotation.T

        self._translation = _clamp(
            _step_toward(
                self._translation, self._raw_translation, self._max_rate_m_s * dt
            ),
            self._max_pos_m,
        )
        # log3/exp3 each tick also re-orthonormalises the rotation.
        rotated = _step_toward_rotation(
            self._rotation, self._raw_rotation, self._max_rate_rad_s * dt
        )
        self._rotation = pin.exp3(_clamp(pin.log3(rotated), self._max_ang_rad))

    def apply(self, pose: pin.SE3) -> pin.SE3:
        """``pose`` displaced by the current correction."""
        return pin.SE3(
            self._rotation @ pose.rotation, pose.translation + self._translation
        )

    @property
    def corrected_target(self) -> pin.SE3 | None:
        """The reference target with the correction folded in, or None unanchored.

        Uses the limited (slew-rated, clamped) correction, independent of whether
        the consumer applies it.
        """
        if self._reference is None:
            return None
        return self.apply(self._reference)

    @property
    def applied(self) -> tuple[float, float]:
        """Correction actually in force, as (mm, deg)."""
        return self._magnitude(self._translation, self._rotation)

    @property
    def raw(self) -> tuple[float, float]:
        """Correction the estimate is asking for, before the limits, as (mm, deg)."""
        return self._magnitude(self._raw_translation, self._raw_rotation)

    @staticmethod
    def _magnitude(
        translation: np.ndarray, rotation: np.ndarray
    ) -> tuple[float, float]:
        return (
            float(np.linalg.norm(translation)) * 1e3,
            float(np.degrees(np.linalg.norm(pin.log3(rotation)))),
        )
