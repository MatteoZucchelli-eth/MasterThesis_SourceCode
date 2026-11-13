import importlib.util
import sys
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).parents[1] / "g1_control" / "vla" / "visual_servo.py"
)
_SPEC = importlib.util.spec_from_file_location("visual_servo_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ImagePlaneVisualServo = _MODULE.ImagePlaneVisualServo


def _drill_entry_point(target_tip):
    """Copy of ``g1_control.vla.planner.drill_entry_point`` (module is loaded
    standalone)."""
    point = np.asarray(target_tip, dtype=float).reshape(3)
    return np.array([point[0], 0.0, 0.0])


class _Projection:
    def __init__(self, uv, depth, inside=True):
        self.uv_px = np.asarray(uv, dtype=float)
        self.depth_m = float(depth)
        self._inside = inside

    def inside(self):
        return self._inside


class _EyeInHandProjector:
    """Pinhole camera 100 mm behind the tip, looking along tip +x."""

    def project_tip_point(self, point_tip):
        x, y, z = np.asarray(point_tip, dtype=float)
        depth = x + 0.1
        if depth <= 0.0:
            return _Projection([0.0, 0.0], depth)
        return _Projection(
            [360.0 + 500.0 * y / depth, 360.0 + 500.0 * z / depth], depth
        )


class _OffAxisProjector:
    """Wrist camera geometry: 126 mm behind and 72.5 mm above ``drill_tip_link``
    (URDF)."""

    BACK_M = 0.126
    UP_M = 0.0725

    def project_tip_point(self, point_tip):
        x, y, z = np.asarray(point_tip, dtype=float)
        depth = x + self.BACK_M
        if depth <= 0.0:
            return _Projection([0.0, 0.0], depth)
        return _Projection(
            [360.0 + 690.0 * y / depth, 360.0 + 690.0 * (z - self.UP_M) / depth],
            depth,
        )


def _servo(**overrides):
    projector = overrides.pop("projector", _EyeInHandProjector())
    setpoint_of = overrides.pop("setpoint_of", _drill_entry_point)
    values = dict(
        kp_s=2.0,
        max_rate_m_s=0.05,
        max_offset_m=0.03,
        jacobian_step_m=0.0005,
        min_singular_px_per_m=100.0,
        ema_alpha=1.0,
        max_error_px=200.0,
        projector=projector,
        setpoint_of=setpoint_of,
    )
    values.update(overrides)
    return ImagePlaneVisualServo(**values)


def test_servo_converges_in_both_wall_plane_coordinates():
    servo = _servo()
    projector = _EyeInHandProjector()
    initial_target = np.array([0.01, 0.008, -0.005])
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    result = None
    for _ in range(400):
        offset = servo.offset_wall_m
        target_now = initial_target - basis @ offset
        result = servo.update(
            target_tip_m=target_now,
            tip_rotation_pelvis=np.eye(3),
            wall_basis_pelvis=basis,
            dt=0.02,
        )

    assert result is not None and result.valid
    assert np.linalg.norm(result.error_px) < 0.05
    np.testing.assert_allclose(servo.offset_wall_m, [0.008, -0.005], atol=2e-5)


def test_zero_alpha_still_reports_current_error_for_convergence_gate():
    servo = _servo(ema_alpha=0.0)
    projector = _EyeInHandProjector()
    initial_target = np.array([0.01, 0.006, 0.0])
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    errors = []
    result = None
    for _ in range(300):
        target_now = initial_target - basis @ servo.offset_wall_m
        result = servo.update(
            target_tip_m=target_now,
            tip_rotation_pelvis=np.eye(3),
            wall_basis_pelvis=basis,
            dt=0.02,
            new_observation=False,
        )
        errors.append(float(np.linalg.norm(result.error_px)))

    assert result is not None and result.valid
    assert errors[-1] < 0.05
    assert errors[-1] < errors[0] * 0.01


def test_servo_rate_and_total_travel_are_bounded():
    servo = _servo(max_rate_m_s=0.002, max_offset_m=0.003)
    projector = _EyeInHandProjector()
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    first = servo.update(
        target_tip_m=[0.01, 0.02, 0.0],
        tip_rotation_pelvis=np.eye(3),
        wall_basis_pelvis=basis,
        dt=0.02,
    )
    assert np.linalg.norm(first.velocity_wall_m_s) <= 0.002 + 1e-12
    assert np.linalg.norm(first.offset_wall_m) <= 0.002 * 0.02 + 1e-12

    for _ in range(500):
        servo.update(
            target_tip_m=[0.01, 0.02, 0.0],
            tip_rotation_pelvis=np.eye(3),
            wall_basis_pelvis=basis,
            dt=0.02,
        )
    assert np.linalg.norm(servo.offset_wall_m) <= 0.003 + 1e-12


def test_implausible_pixel_error_does_not_move_the_command():
    servo = _servo(max_error_px=10.0)
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
    result = servo.update(
        target_tip_m=[0.01, 0.02, 0.0],
        tip_rotation_pelvis=np.eye(3),
        wall_basis_pelvis=basis,
        dt=0.02,
    )
    assert not result.valid
    assert "plausibility" in result.reason
    np.testing.assert_allclose(result.offset_wall_m, [0.0, 0.0])


def test_setpoint_tracks_the_target_instead_of_the_tip_origin():
    """The setpoint pixel follows the target's station down the drill axis."""
    servo = _servo(projector=_OffAxisProjector())
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    near = servo.update(
        target_tip_m=[0.01, 0.0, 0.0],
        tip_rotation_pelvis=np.eye(3),
        wall_basis_pelvis=basis,
        dt=0.02,
    )
    near_uv = servo.setpoint_uv_px
    assert near.valid and near_uv is not None

    servo.update(
        target_tip_m=[0.05, 0.0, 0.0],
        tip_rotation_pelvis=np.eye(3),
        wall_basis_pelvis=basis,
        dt=0.02,
    )
    far_uv = servo.setpoint_uv_px
    assert far_uv is not None
    assert abs(float(far_uv[1] - near_uv[1])) > 20.0

    servo.reset()
    assert servo.setpoint_uv_px is None


def test_convergence_leaves_no_parallax_offset_at_a_standoff():
    """Converging at a standoff puts the bit on the target with no parallax offset."""
    standoff_m = 0.015
    servo = _servo(projector=_OffAxisProjector())
    # Perfectly aimed drill one standoff away: any offset is servo error.
    initial_target = np.array([standoff_m, 0.0, 0.0])
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    result = None
    for _ in range(400):
        target_now = initial_target - basis @ servo.offset_wall_m
        result = servo.update(
            target_tip_m=target_now,
            tip_rotation_pelvis=np.eye(3),
            wall_basis_pelvis=basis,
            dt=0.02,
        )

    assert result is not None and result.valid
    assert np.linalg.norm(result.error_px) < 0.05
    np.testing.assert_allclose(servo.offset_wall_m, [0.0, 0.0], atol=1e-4)


def test_tip_origin_setpoint_is_what_introduced_the_parallax():
    """A tip-origin setpoint settles at the predicted parallax offset U*d/B."""
    standoff_m = 0.015
    servo = _servo(
        projector=_OffAxisProjector(), setpoint_of=lambda _target: np.zeros(3)
    )
    initial_target = np.array([standoff_m, 0.0, 0.0])
    basis = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))

    for _ in range(400):
        target_now = initial_target - basis @ servo.offset_wall_m
        servo.update(
            target_tip_m=target_now,
            tip_rotation_pelvis=np.eye(3),
            wall_basis_pelvis=basis,
            dt=0.02,
        )

    # Tip origin and target project alike when U/B == (o + U)/(d + B), i.e.
    # o = U*d/B (camera is rigid with the tip).
    predicted_m = _OffAxisProjector.UP_M * standoff_m / _OffAxisProjector.BACK_M
    assert abs(predicted_m - 0.0086) < 1e-4
    np.testing.assert_allclose(servo.offset_wall_m, [0.0, predicted_m], atol=2e-4)
