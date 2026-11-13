# MIT License
"""Unit tests for g1_control.utils pure helpers."""

import numpy as np
import pytest

from g1_control.utils.utils import (
    create_action_buffer,
    get_gravity_orientation,
    load_config,
    pr_slice,
    project_on_sphere,
)
from g1_definitions import definitions


def test_project_on_sphere_lands_on_sphere():
    center = np.array([1.0, 2.0, 3.0])
    radius = 2.5
    point = center + np.array([10.0, 0.0, 0.0])
    out = project_on_sphere(center, radius, point)
    assert np.isclose(np.linalg.norm(out - center), radius)
    # Stays on the ray from center towards point.
    assert np.allclose(out, center + np.array([radius, 0.0, 0.0]))


def test_get_gravity_orientation_identity_quat():
    # Identity quaternion -> gravity points straight down in body frame.
    g = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(g, [0.0, 0.0, -1.0])


def test_get_gravity_orientation_is_unit_norm():
    rng = np.random.default_rng(0)
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    g = get_gravity_orientation(q)
    assert np.isclose(np.linalg.norm(g), 1.0)


def test_get_gravity_orientation_180_about_x_flips_z():
    # Rotation by pi about x: quat = [0, 1, 0, 0]. Gravity z flips sign.
    g = get_gravity_orientation(np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.allclose(g, [0.0, 0.0, 1.0])


def test_create_action_buffer_shapes():
    buf = create_action_buffer(chunk_size=7)
    assert set(buf) == {definitions.RIGHT_EEF_KEY, "upper_body_joint_chunk"}
    assert buf[definitions.RIGHT_EEF_KEY].shape == (7, 9)
    assert buf["upper_body_joint_chunk"].shape == (7, 28)
    assert all(v.dtype == np.float32 for v in buf.values())
    assert all(np.count_nonzero(v) == 0 for v in buf.values())


def test_pr_slice_matches_definition():
    from g1_definitions import definitions

    arr = np.arange(40)
    for group in ("left_arm", "right_arm", "waist"):
        start, end = definitions.JOINT_PR_SLICES[group]
        assert np.array_equal(pr_slice(arr, group), arr[start:end])


def test_load_config_parses_arrays(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "simulation_dt: 0.002\n"
        "motor_kps: [1, 2, 3]\n"
        "default_angles: [0.1, 0.2]\n"
        "name: g1\n"
    )
    out = load_config(str(cfg))
    assert isinstance(out["motor_kps"], np.ndarray)
    assert out["motor_kps"].dtype == np.float32
    assert np.allclose(out["default_angles"], [0.1, 0.2])
    # Non-array keys pass through untouched.
    assert out["name"] == "g1"
    assert out["simulation_dt"] == 0.002


def test_project_on_sphere_zero_direction_raises():
    # Degenerate input (point == center) divides by zero -> nan, not a crash.
    out = project_on_sphere(np.zeros(3), 1.0, np.zeros(3))
    assert np.isnan(out).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
