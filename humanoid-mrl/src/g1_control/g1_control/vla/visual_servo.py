# MIT License
"""Image-plane proportional control for the wrist-camera drill alignment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0.0 or norm <= limit:
        return value
    return value * (float(limit) / norm)


@dataclass(frozen=True)
class VisualServoResult:
    offset_wall_m: np.ndarray
    velocity_wall_m_s: np.ndarray
    error_px: np.ndarray | None
    valid: bool
    reason: str = ""


class ImagePlaneVisualServo:
    """Integrate a bounded P velocity that centres the target on a setpoint pixel.

    Controls two orthonormal wall-plane directions; a numerical image Jacobian
    maps them to pixel error.

    ``setpoint_of(target_tip)`` returns the point the target must land on,
    normally the drill-axis station level with the target
    (``planner.drill_entry_point``). Comparing points at the same axial station
    removes camera parallax, so zero pixel error means the bit enters the wall
    at the target.
    """

    def __init__(
        self,
        *,
        kp_s: float,
        max_rate_m_s: float,
        max_offset_m: float,
        jacobian_step_m: float,
        min_singular_px_per_m: float,
        ema_alpha: float,
        max_error_px: float,
        projector,
        setpoint_of,
    ) -> None:
        self._kp_s = float(kp_s)
        self._max_rate_m_s = float(max_rate_m_s)
        self._max_offset_m = float(max_offset_m)
        self._step_m = float(jacobian_step_m)
        self._min_singular = float(min_singular_px_per_m)
        # Unused: the planner freezes the 3D target upstream; pixel error is
        # instantaneous.
        self._alpha = float(ema_alpha)
        self._max_error_px = float(max_error_px)
        self._projector = projector
        self._setpoint_of = setpoint_of
        if self._step_m <= 0.0:
            raise ValueError("visual-servo Jacobian step must be positive")
        self.reset()

    def reset(self) -> None:
        self._offset_wall_m = np.zeros(2)
        self._target_uv_px: np.ndarray | None = None
        # Recomputed every update: the setpoint moves with the target.
        self._setpoint_uv_px: np.ndarray | None = None

    @property
    def offset_wall_m(self) -> np.ndarray:
        return self._offset_wall_m.copy()

    @property
    def setpoint_uv_px(self) -> np.ndarray | None:
        """Pixel the target is being driven onto, as of the last update.

        Projection of ``setpoint_of(target)``, i.e. where the bit will enter the
        wall. None before the first valid update.
        """
        return None if self._setpoint_uv_px is None else self._setpoint_uv_px.copy()

    @property
    def target_uv_px(self) -> np.ndarray | None:
        """Most recent projection of the stationary pelvis-frame target."""
        return None if self._target_uv_px is None else self._target_uv_px.copy()

    def update(
        self,
        *,
        target_tip_m: np.ndarray,
        tip_rotation_pelvis: np.ndarray,
        wall_basis_pelvis: np.ndarray,
        dt: float,
        new_observation: bool = True,
    ) -> VisualServoResult:
        """Consume one current target-in-tip estimate and advance the command.

        ``wall_basis_pelvis`` is 3x2. Moving the eye-in-hand tip by ``+dq``
        moves a static target by ``-dq`` in the tip frame, which supplies the
        sign of the finite-difference Jacobian below.
        """
        target_tip = np.asarray(target_tip_m, dtype=float).reshape(3)
        rotation = np.asarray(tip_rotation_pelvis, dtype=float).reshape(3, 3)
        basis = np.asarray(wall_basis_pelvis, dtype=float).reshape(3, 2)

        target = self._projector.project_tip_point(target_tip)
        setpoint = self._projector.project_tip_point(self._setpoint_of(target_tip))
        self._target_uv_px = (
            None
            if target is None
            else np.asarray(target.uv_px, dtype=float).copy()
        )
        self._setpoint_uv_px = (
            None
            if setpoint is None
            else np.asarray(setpoint.uv_px, dtype=float).copy()
        )
        if target is None or setpoint is None:
            return self._invalid("camera projection is unavailable")
        if target.depth_m <= 0.0 or setpoint.depth_m <= 0.0:
            return self._invalid("target or drill axis station is behind the camera")
        if not target.inside() or not setpoint.inside():
            return self._invalid(
                "target or drill axis station projects outside the frame"
            )

        error = np.asarray(target.uv_px - setpoint.uv_px, dtype=float)
        if float(np.linalg.norm(error)) > self._max_error_px:
            return self._invalid("pixel error exceeds the plausibility gate")

        # Differentiate the error (target minus setpoint): both pixels move when
        # the drill axis is not square to the wall.
        basis_tip = rotation.T @ basis
        jacobian = np.empty((2, 2), dtype=float)
        for column in range(2):
            moved_tip = target_tip - self._step_m * basis_tip[:, column]
            moved = self._projector.project_tip_point(moved_tip)
            moved_setpoint = self._projector.project_tip_point(
                self._setpoint_of(moved_tip)
            )
            if (
                moved is None
                or moved_setpoint is None
                or moved.depth_m <= 0.0
                or moved_setpoint.depth_m <= 0.0
            ):
                return self._invalid("cannot perturb the target projection")
            moved_error = np.asarray(
                moved.uv_px - moved_setpoint.uv_px, dtype=float
            )
            jacobian[:, column] = (moved_error - error) / self._step_m

        singular = np.linalg.svd(jacobian, compute_uv=False)
        if not np.all(np.isfinite(singular)) or singular[-1] < self._min_singular:
            return self._invalid("image Jacobian is singular")

        velocity = -self._kp_s * np.linalg.pinv(jacobian) @ error
        velocity = _clip_norm(velocity, self._max_rate_m_s)
        self._offset_wall_m = _clip_norm(
            self._offset_wall_m + max(0.0, float(dt)) * velocity,
            self._max_offset_m,
        )
        return VisualServoResult(
            offset_wall_m=self.offset_wall_m,
            velocity_wall_m_s=velocity,
            error_px=error.copy(),
            valid=True,
        )

    def _invalid(self, reason: str) -> VisualServoResult:
        return VisualServoResult(
            offset_wall_m=self.offset_wall_m,
            velocity_wall_m_s=np.zeros(2),
            error_px=None,
            valid=False,
            reason=reason,
        )
