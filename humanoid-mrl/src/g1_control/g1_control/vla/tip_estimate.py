# MIT License
"""Anchoring and fusion of the VLA's tip->target predictions.

A ``tip_target`` prediction is the target pose in the drill-tip frame as of
the snapshot that produced it.

* Pose helpers convert between the 9D layout (xyz(3) + rot6d(6), matching
  ``RobotModel.se3_to_9d_pose``) and 4x4 transforms; :func:`anchored_target_9d`
  places a prediction in the pelvis frame using its own anchor.
* :class:`TipTargetEstimator` EMA-fuses successive anchored predictions (removes
  noise, not systematic bias).

Pure numpy; the caller converts to SE3 for IK.
"""
from __future__ import annotations

import numpy as np



def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6D rotation (two matrix columns) back to a full matrix."""
    x = rot6d[0:3] / np.linalg.norm(rot6d[0:3])
    y = rot6d[3:6] - np.dot(x, rot6d[3:6]) * x
    y = y / np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def pose_9d_to_mat(pose_9d: np.ndarray) -> np.ndarray:
    """9D pose (xyz + rot6d) to a 4x4 homogeneous transform."""
    mat = np.eye(4)
    mat[:3, :3] = rot6d_to_matrix(np.asarray(pose_9d[3:9], dtype=float))
    mat[:3, 3] = pose_9d[:3]
    return mat


def mat_to_pose_9d(mat: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform to a 9D pose (xyz + rot6d)."""
    return np.concatenate([mat[:3, 3], mat[:3, 0], mat[:3, 1]])


def rms_spread(positions) -> float:
    """RMS distance of a set of points from their own mean [m].

    Zero for fewer than two points, which callers must treat as unproven.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape[0] < 2:
        return 0.0
    deviations = positions - positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(deviations**2, axis=1))))


def anchored_target_9d(pred_9d: np.ndarray, anchor_tip_9d: np.ndarray) -> np.ndarray:
    """Place a tip-frame prediction in the pelvis frame with its own anchor.

    The anchor is the tip pose fed to that inference, so tip motion during the
    server round trip does not smear the result.
    """
    return mat_to_pose_9d(
        pose_9d_to_mat(np.asarray(anchor_tip_9d, dtype=float))
        @ pose_9d_to_mat(np.asarray(pred_9d, dtype=float))
    )


class TipTargetEstimator:
    """Fuses the 1 Hz tip-frame target predictions into a pelvis-frame estimate.

    Predictions are anchored with their own snapshot's tip pose (see
    :func:`anchored_target_9d`), then averaged in the static pelvis frame.
    """

    def __init__(self, ema_alpha: float = 0.4) -> None:
        # Weight of a NEW prediction in the moving average (1.0 = no filtering).
        self._alpha = float(ema_alpha)
        self.reset()

    def reset(self) -> None:
        self._position: np.ndarray | None = None
        self._rot6d: np.ndarray | None = None
        self._last_seq: int | None = None
        self._last_update_time: float | None = None

    def update(
        self,
        anchored: tuple[np.ndarray, np.ndarray, float, int] | None,
        now: float,
    ) -> None:
        """Fold in the latest anchored prediction, if it is new.

        ``anchored`` is ``(pred_tip_target_9d, anchor_tip_9d, stamp, seq)`` as
        returned by ``VLAClient.get_tip_target_anchored``: the tip-frame
        prediction, the pelvis-frame tip pose fed to that inference, its
        timestamp and a monotonically increasing sequence number.
        """
        if anchored is None:
            return
        pred_9d, anchor_tip_9d, _stamp, seq = anchored
        if seq == self._last_seq:
            return
        self._last_seq = seq

        target_9d = anchored_target_9d(pred_9d, anchor_tip_9d)
        position = target_9d[:3]
        rot6d = target_9d[3:9]

        if self._position is None:
            self._position = position
            self._rot6d = rot6d
        else:
            a = self._alpha
            self._position = (1.0 - a) * self._position + a * position
            # rot6d_to_matrix re-orthonormalises on read-out (small spread).
            self._rot6d = (1.0 - a) * self._rot6d + a * rot6d
        self._last_update_time = now

    def seed(self, pose_9d: np.ndarray, last_seq: int | None = None) -> None:
        """Start the average from an estimate produced elsewhere.

        ``last_seq`` is the sequence number that estimate already consumed, so
        it is not folded in twice. Age stays infinite until a new prediction
        arrives: a seed is not a measurement.
        """
        pose_9d = np.asarray(pose_9d, dtype=float)
        self._position = pose_9d[:3].copy()
        self._rot6d = pose_9d[3:9].copy()
        self._last_seq = last_seq
        self._last_update_time = None

    @property
    def has_estimate(self) -> bool:
        return self._position is not None

    def age(self, now: float) -> float:
        """Seconds since the last accepted prediction (inf before the first)."""
        if self._last_update_time is None:
            return float("inf")
        return now - self._last_update_time

    def pose_9d(self) -> np.ndarray | None:
        """Fused target pose in the pelvis frame, or None before any update."""
        if self._position is None:
            return None
        rotation = rot6d_to_matrix(self._rot6d)
        return np.concatenate([self._position, rotation[:, 0], rotation[:, 1]])

    def position(self) -> np.ndarray | None:
        if self._position is None:
            return None
        return self._position.copy()
