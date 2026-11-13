# MIT License
"""Unit tests for the tip-target estimator (pure numpy, no URDF needed)."""

import numpy as np
import pytest

from g1_control.vla.tip_estimate import (
    mat_to_pose_9d,
    pose_9d_to_mat,
    rot6d_to_matrix,
    TipTargetEstimator,
)


def _pose_9d(position, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return np.concatenate([position, rotation[:, 0], rotation[:, 1]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ── 9D pose helpers ──────────────────────────────────────────────────────────


def test_pose_9d_mat_round_trip():
    pose = _pose_9d(np.array([0.1, -0.2, 0.3]), _rot_z(0.7))
    assert np.allclose(mat_to_pose_9d(pose_9d_to_mat(pose)), pose)


def test_rot6d_gram_schmidt_restores_orthonormality():
    rot6d = np.concatenate([_rot_z(0.3)[:, 0], _rot_z(0.3)[:, 1]]) + 0.01
    R = rot6d_to_matrix(rot6d)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


# ── TipTargetEstimator ───────────────────────────────────────────────────────


def test_estimator_anchors_prediction_into_pelvis_frame():
    """tip pose ∘ tip-frame prediction must reproduce the pelvis-frame target."""
    tip_rot = _rot_z(0.5)
    tip_pos = np.array([0.4, -0.1, 0.2])
    target_pos = np.array([0.55, 0.05, 0.25])
    # Prediction as the training label defines it: T_tip^-1 * T_target.
    pred_pos = tip_rot.T @ (target_pos - tip_pos)
    pred_rot = tip_rot.T  # target orientation = identity in pelvis frame

    est = TipTargetEstimator(ema_alpha=1.0)
    est.update((_pose_9d(pred_pos, pred_rot), _pose_9d(tip_pos, tip_rot), 0.0, 1), 0.0)

    assert np.allclose(est.position(), target_pos, atol=1e-9)
    fused = est.pose_9d()
    assert np.allclose(rot6d_to_matrix(fused[3:]), np.eye(3), atol=1e-9)


def test_estimator_ema_and_seq_dedup():
    est = TipTargetEstimator(ema_alpha=0.5)
    anchor = _pose_9d(np.zeros(3))
    est.update((_pose_9d(np.array([1.0, 0.0, 0.0])), anchor, 0.0, 1), 0.0)
    # Same seq: must be ignored no matter the payload.
    est.update((_pose_9d(np.array([9.0, 9.0, 9.0])), anchor, 0.0, 1), 0.5)
    assert np.allclose(est.position(), [1.0, 0.0, 0.0])
    # New seq: folded in at alpha=0.5.
    est.update((_pose_9d(np.array([2.0, 0.0, 0.0])), anchor, 1.0, 2), 1.0)
    assert np.allclose(est.position(), [1.5, 0.0, 0.0])


def test_estimator_seed_starts_from_a_prior_without_faking_freshness():
    est = TipTargetEstimator(ema_alpha=0.5)
    est.seed(_pose_9d(np.array([1.0, 0.0, 0.0])), last_seq=7)
    assert est.has_estimate
    assert np.allclose(est.position(), [1.0, 0.0, 0.0])
    # A seed is not a measurement: age stays infinite.
    assert est.age(now=0.0) == float("inf")

    anchor = _pose_9d(np.zeros(3))
    # A sample with the seeded seq is already in the seed and is ignored.
    est.update((_pose_9d(np.array([5.0, 0.0, 0.0])), anchor, 0.0, 7), 0.0)
    assert np.allclose(est.position(), [1.0, 0.0, 0.0])

    est.update((_pose_9d(np.array([2.0, 0.0, 0.0])), anchor, 1.0, 8), 1.0)
    assert np.allclose(est.position(), [1.5, 0.0, 0.0])
    assert est.age(now=1.0) == pytest.approx(0.0)


def test_estimator_age_and_reset():
    est = TipTargetEstimator()
    assert est.age(now=10.0) == float("inf")
    assert est.position() is None
    est.update((_pose_9d(np.zeros(3)), _pose_9d(np.zeros(3)), 0.0, 1), 2.0)
    assert est.age(now=3.5) == pytest.approx(1.5)
    est.reset()
    assert not est.has_estimate
