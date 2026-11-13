# MIT License
"""Shared tip-frame target-estimate plumbing for the VLA clients.

Used by both :mod:`g1_control.vla.client` and
:mod:`g1_control.vla.perception_client`:

* :class:`TipTargetState` — locked latest-prediction slot behind
  ``get_tip_target`` / ``get_tip_target_anchored``.
* :class:`TipTargetLifter` — lifts a 2D :class:`PerceptionDetection` to a
  tip-frame pose via the hand-camera intrinsics and the constant
  ``hand_camera_optical_frame -> drill_tip_link`` extrinsic.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from g1_control.utils.ros_helpers import lookup_se3

from .perception import (
    CameraIntrinsics,
    FrameProjection,
    PerceptionDetection,
    target_pose_in_camera,
)
from .tip_estimate import mat_to_pose_9d, pose_9d_to_mat

logger = logging.getLogger(__name__)


class TipTargetState:
    """Thread-safe slot holding the newest tip-frame target prediction.

    The inference thread writes with :meth:`publish`; the control loop reads
    with :meth:`latest` / :meth:`anchored`. ``seq`` increments per prediction and
    never resets, so consumers can deduplicate on it across sessions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pred: np.ndarray | None = None
        self._anchor: np.ndarray | None = None
        self._stamp: float = 0.0
        self._seq: int = 0

    def reset(self) -> None:
        """Drop the previous session's prediction (the sequence keeps counting)."""
        with self._lock:
            self._pred = None
            self._anchor = None

    def publish(self, pred_tip_9d: np.ndarray, anchor_tip_9d: np.ndarray) -> None:
        """Store a prediction and the tip anchor of the inference that made it."""
        with self._lock:
            self._pred = np.asarray(pred_tip_9d, dtype=float).copy()
            self._anchor = np.asarray(anchor_tip_9d, dtype=float).copy()
            self._stamp = time.monotonic()
            self._seq += 1

    def latest(self) -> np.ndarray | None:
        """Newest tip-frame target pose (9D), or None before the first one."""
        with self._lock:
            return None if self._pred is None else self._pred.copy()

    def anchored(self) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """``(pred_tip_9d, anchor_tip_9d, stamp, seq)``, or None before the first."""
        with self._lock:
            if self._pred is None or self._anchor is None:
                return None
            return (self._pred.copy(), self._anchor.copy(), self._stamp, self._seq)


class DetectionSlot:
    """Thread-safe newest raw detection, plus a count of every one seen.

    The count distinguishes "detections gated out" from "detector stopped".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detection: PerceptionDetection | None = None
        self._count = 0

    def reset(self) -> None:
        """Drop the previous session's detection and restart the count."""
        with self._lock:
            self._detection = None
            self._count = 0

    def publish(self, detection: PerceptionDetection) -> None:
        """Store a detection, gated out or not — this counts every arrival."""
        with self._lock:
            self._detection = detection
            self._count += 1

    def latest(self) -> PerceptionDetection | None:
        with self._lock:
            return self._detection

    @property
    def count(self) -> int:
        """Detections seen this session, whatever the gate made of them."""
        with self._lock:
            return self._count


class TipTargetLifter:
    """Lifts one 2D detection to a target pose in the drill-tip frame.

    Back-projects ``(u, v, depth)`` with the hand-camera intrinsics, orients by
    ``wall_normal_cam`` (see :func:`~g1_control.vla.perception.target_pose_in_camera`)
    and composes with the constant ``hand_camera_optical_frame -> drill_tip_link``
    extrinsic, looked up from TF once and cached.
    """

    def __init__(
        self,
        tf_buffer,
        camera_frame: str,
        tip_frame: str,
        intrinsics: CameraIntrinsics | None = None,
        min_visibility: float = 0.0,
    ) -> None:
        self._intr = intrinsics
        self._tf_buffer = tf_buffer
        self._camera_frame = camera_frame
        self._tip_frame = tip_frame
        self._min_visibility = float(min_visibility)
        self._extrinsic_tip_cam: np.ndarray | None = None

    def set_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        """Set the intrinsics of the prepared (undistorted, square-cropped) frame.

        Late-bound because they depend on CameraInfo; a plain reference swap, so
        an in-flight lift keeps its matrix.
        """
        self._intr = intrinsics

    def lift(self, detection: PerceptionDetection) -> np.ndarray | None:
        """Tip-frame 9D target pose, or None below ``min_visibility`` / before
        the camera geometry is up.

        ``min_visibility`` is a floor on the detector's ``visibility`` scalar
        (0.0 = off); its usefulness is checkpoint-specific. On perception_v4
        (680 frames, rightfreq vs ``heatmap_labels_full``):

            visibility >= 0.99   median 0.68 px error,  1.1% worse than 40 px
            0.90 .. 0.99         median 1.00 px,        2.0%
            0.50 .. 0.90         median 1.52 px,       17.4%
            < 0.10               median 262 px,        78.6%

        Rejected detections are still published to :class:`DetectionSlot` by the
        caller.
        """
        if detection.visibility < self._min_visibility:
            logger.debug(
                "[perception] detection at %s gated out: visibility %.2f < %.2f",
                np.round(detection.uv_px, 1),
                detection.visibility,
                self._min_visibility,
            )
            return None

        if self._intr is None:
            logger.warning(
                "[perception] detection arrived but no CameraInfo on the wrist "
                "stream yet; dropping it"
            )
            return None

        extrinsic = self._extrinsic()
        if extrinsic is None:
            logger.warning(
                "[perception] detection arrived but %s->%s TF not available "
                "yet; dropping it",
                self._camera_frame,
                self._tip_frame,
            )
            return None

        pred_tip_9d = mat_to_pose_9d(
            extrinsic @ target_pose_in_camera(detection, self._intr)
        )
        logger.debug(
            "[perception] uv=%s vis=%.2f depth=%.3f m -> tip->target xyz (mm): %s",
            np.round(detection.uv_px, 1),
            detection.visibility,
            detection.depth_m,
            np.round(pred_tip_9d[:3] * 1000.0, 1),
        )
        return pred_tip_9d

    def project_tip_pose(self, pred_tip_9d: np.ndarray) -> FrameProjection | None:
        """Where a tip-frame target pose falls in the prepared wrist frame.

        Inverse of :meth:`lift`; None while intrinsics or extrinsic are missing.
        """
        return self.project_tip_point(pose_9d_to_mat(pred_tip_9d)[:3, 3])

    def project_tip_point(self, point_tip) -> FrameProjection | None:
        """Where a tip-frame *point* falls in the prepared wrist frame.

        Position-only variant of :meth:`project_tip_pose`.
        """
        extrinsic = self._extrinsic()
        if self._intr is None or extrinsic is None:
            return None
        # _extrinsic() is T(tip <- cam); invert to map tip -> cam.
        cam_tip = np.linalg.inv(extrinsic)
        point = np.asarray(point_tip, dtype=float).reshape(3)
        return FrameProjection.of(cam_tip[:3, :3] @ point + cam_tip[:3, 3], self._intr)

    def extrinsic_tip_cam(self) -> np.ndarray | None:
        """The cached ``T(tip <- cam)``, or None until TF has it.

        Shared with :class:`~g1_control.vla.planner.StereoDepthDonor` so both use
        the same matrix and availability.
        """
        return self._extrinsic()

    def _extrinsic(self) -> np.ndarray | None:
        """Constant ``hand_camera_optical_frame -> drill_tip_link`` transform, cached."""
        if self._extrinsic_tip_cam is not None:
            return self._extrinsic_tip_cam
        # lookup_se3(target, source) = target pose in source = T(source<-target):
        # here that is T(tip<-cam), which maps a camera-frame point to the tip.
        se3 = lookup_se3(self._tf_buffer, self._camera_frame, self._tip_frame)
        if se3 is None:
            return None
        self._extrinsic_tip_cam = np.asarray(se3.homogeneous, dtype=float)
        logger.info(
            "[perception] cached %s->%s extrinsic (|t|=%.1f mm)",
            self._camera_frame,
            self._tip_frame,
            float(np.linalg.norm(self._extrinsic_tip_cam[:3, 3])) * 1000.0,
        )
        return self._extrinsic_tip_cam
