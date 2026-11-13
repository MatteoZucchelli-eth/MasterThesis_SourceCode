# MIT License
"""Pure primitives of the VLA_PLANNER state: approach -> stop -> check -> plan.

Pure numpy, unit-testable without ROS or pinocchio (``test/test_vla_planner.py``):

* :class:`TargetPoseAccumulator` — batch-averages anchored predictions taken
  while the tip is held still; its dispersion gates accumulate -> finalize.
* :class:`StereoDepthDonor` — optionally rebuilds each prediction on the
  measured wall plane (detector bearing, measured range).
* :func:`finalize_knots` — knot sequence for the final approach trajectory.
* :class:`FinalizeResidual` — target-to-tip miss split about the drill axis.

The approach gate lives in :mod:`g1_control.vla.handover`.

``anchored`` tuples are ``(pred_tip_target_9d, anchor_tip_9d, stamp, seq)`` as
returned by ``VLAClient.get_tip_target_anchored``. 9D pose: xyz(3) + rot6d(6),
matching ``RobotModel.se3_to_9d_pose``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Re-exported from :mod:`g1_control.vla.depth_donor` for consumers of this module.
from .depth_donor import (  # noqa: F401
    DEPTH_SOURCE_FINE,
    DEPTH_SOURCE_STEREO,
    DEPTH_SOURCES,
    DepthBiasedSource,
    DepthDonation,
    DepthDonor,
    Ranging,
    Sighting,
    StereoDepthDonor,
    WallPlaneSlot,
)

from .tip_estimate import (
    anchored_target_9d,
    mat_to_pose_9d,
    pose_9d_to_mat,
    rms_spread,
    rot6d_to_matrix,
)


@dataclass
class VlaPlannerConfig:
    """Tuning of the VLA_PLANNER state (distances in metres, times in seconds).

    Near the target (< 8 cm) the tip head's per-prediction error is ~10 mm
    median (~ -3 mm bias); beyond ~12 cm predictions are tens of mm off.
    """

    approach_distance: float = 0.125
    """Max distance of the standoff batch target before the attempt is
    abandoned as the head's far mode [m]. Applied only after the transit."""
    approach_consecutive: int = 3
    """Consecutive judged detections that must clear the gate before the arm
    stops. Debounces distractor locks; must stay above 1."""
    approach_timeout_s: float = 90.0
    """Abandon the attempt if the gate has not fired after this long [s].
    Restarts from the default pose (counts against ``abort_max_restarts``)."""

    approach_detection_max_age_s: float = 2.0
    """The close-range estimate must be fresher than this [s]."""
    approach_rearm_s: float = 5.0
    """After the gate condition fails, how long before a new streak may start
    [s]."""

    accumulate_settle_tol_mm: float = 2.0
    """Max FK(measured q) to FK(commanded q) tip distance before detections
    enter the batch [mm].

    Also the hard gate before ACCUMULATING starts another motion leg or
    declares a successful hold; the settle timeout never waives it."""
    accumulate_settle_hold_s: float = 1.0
    """Continuous time tracking must remain inside the tolerance before an
    accumulated estimate may start another motion leg."""
    accumulate_settle_min_s: float = 2.0
    """Minimum wait before the first detection is accepted [s]; an immediate
    tolerance pass can mean the controller has not reacted yet. Keep well under
    ``accumulate_settle_timeout_s``."""
    accumulate_settle_timeout_s: float = 5.0
    """Start accumulating anchored detections after this long waiting to settle.

    Samples are anchored to their own tip pose, so a standing offset does not
    bias the estimate; the next motion leg still waits for
    ``accumulate_settle_tol_mm``. Not counted in ``accumulate_timeout_s``."""
    accumulate_min_samples: int = 30
    """Fresh predictions to average before the estimate counts as converged.

    Must fit within ``accumulate_timeout_s`` at the detector rate (30 samples
    take ~3 s at 10 Hz, 10 s at 3 Hz)."""
    accumulate_max_spread_mm: float = 12.0
    """Position RMS spread about the mean below which the estimate counts as
    converged [mm]."""
    accumulate_outlier_mm: float = 50.0
    """Discard a detection this far from the median of the accepted ones [mm]."""
    accumulate_timeout_s: float = 12.0
    """Stop accumulating after this long: proceed with the mean if there are
    samples, abort to HOLDING otherwise."""
    depth_source: str = DEPTH_SOURCE_FINE
    """Where the close-range estimate's range comes from:

    * ``"fine"`` — the detector's own ``target_depth_m``.
    * ``"stereo"`` — the head D435 wall plane, via :class:`StereoDepthDonor`.

    Both keep the detector's bearing. Applies to the ACCUMULATING batch and the
    finalize live correction, not the approach gate.

    ``"stereo"`` requires the ``g1_sensors`` wall_plane node on an aligned head
    depth stream; without a plane no sample is built."""
    depth_max_age_s: float = 2.0
    """Max age of the donated range, measured from the last source sequence
    change on the caller's clock [s]."""
    fine_depth_bias_mm: float = 0.0
    """Close-range detector depth correction [mm]. Positive moves the target
    further along its wrist-camera ray. Separate from the head D435 correction."""
    depth_max_shift_mm: float = 60.0
    """Skip a sample the donated depth would move further than this [mm].

    Rejects the detector's far-mode depth answers; sized above its normal range
    error (tens of mm) and below the far mode."""

    finalize_standoff_m: float = 0.075
    """Where the transit merges onto the estimated wall normal — the length
    of the straight-in push (same role as ``auto_collect_standoff_m``)."""
    finalize_standoff_rise_m: float = 0.005
    """Raise the standoff this far up the wall, tilting the push [m].

    The drill stays along the normal; the tip approaches slightly from above.
    0 gives a straight-in push. Tilt is ``atan(rise / standoff)``."""
    finalize_wall_offset_m: float = 0.005
    """Stop this far short of the estimated target along the wall normal [m]."""
    finalize_wall_lateral_offset_m: float = 0.0
    """Translate the target along wall-local +y (pelvis +y/left for a
    forward-facing wall) [m]."""
    finalize_wall_up_offset_m: float = 0.0
    """Translate the target upward in the wall plane (negative moves it down) [m]."""
    finalize_transit_duration_s: float = 6.0
    """Seconds for the current-pose -> standoff leg, over which the drill is
    also turned onto the wall normal."""
    finalize_insert_duration_s: float = 8.0
    """Seconds for the standoff -> target straight-in push (slow, so the arm
    integral nulls the droop before arrival)."""
    finalize_align_drill: bool = True
    """Align the drill axis (tip x) to the estimated target's x-axis, fading in
    over the transit leg. False keeps the FK-captured orientation."""

    visual_servo_enabled: bool = False
    """Enable image-plane alignment at the re-measured standoff, before the push.

    Proportional feedback aligns the predicted target pixel with the projected
    drill-tip pixel without deliberately changing depth."""
    visual_servo_kp_s: float = 1.0
    """Proportional gain from image error to wall-plane velocity [1/s]."""
    visual_servo_max_rate_mm_s: float = 3.0
    """Maximum norm of the commanded wall-plane velocity."""
    visual_servo_max_offset_mm: float = 25.0
    """Maximum wall-plane displacement from the standoff pose."""
    visual_servo_jacobian_step_mm: float = 0.5
    """Finite-difference translation used to build the 2x2 image Jacobian."""
    visual_servo_min_singular_px_per_m: float = 100.0
    """Reject camera geometry whose weakest wall-plane response is smaller."""
    visual_servo_ema_alpha: float = 0.4
    """Weight of a new sample in the servo's anchored 3D target estimate
    (0 = hold the accumulated standoff estimate)."""
    visual_servo_max_error_px: float = 120.0
    """Reject a target this far from the drill-tip pixel as a bad sighting."""
    visual_servo_success_px: float = 5.0
    """Pixel-error radius that starts the success hold timer."""
    visual_servo_success_hold_s: float = 0.5
    """Continuous time inside ``visual_servo_success_px`` before insertion."""
    visual_servo_timeout_s: float = 12.0
    """Give up and restart when image alignment cannot converge in this time."""

    finalize_correct_target: bool = False
    """Track the live target estimate (``FK(q) . tip^-1 . target``) through the
    finalize instead of flying the frozen ACCUMULATING pose.

    False disables the live tracker entirely. During a visual-servo insertion
    an enabled tracker is measurement-only."""
    finalize_correction_ema_alpha: float = 0.4
    """Weight of a new prediction in the live estimate the correction follows."""
    finalize_correction_max_age_s: float = 3.0
    """Stop updating the correction once the newest detection is older than
    this; the correction then holds its last value."""
    finalize_correction_rate_mm_s: float = 5.0
    """Slew rate of the applied correction (glitch protection, not a gain)."""
    finalize_correction_max_mm: float = 25.0
    """Magnitude clamp on the applied correction [mm]."""
    finalize_correction_gate_mm: float = 50.0
    """Ignore a live prediction further than this from the initial planned
    target (0 disables) [mm].

    Gates the correction's input, keeping far-mode answers out of the EMA;
    ``finalize_correction_max_mm`` only clamps its output."""
    finalize_correction_rate_deg_s: float = 3.0
    """Slew rate of the orientation half of the correction."""
    finalize_correction_max_deg: float = 3.0
    """Magnitude clamp on the orientation half."""

    finalize_shadow_fine_only: bool = True
    """Run a second, never-applied live tracker fed the detector's own depth.

    Published as TF ``vla_tip_target_live_fine_only`` and summarised per leg,
    to compare depth sources within one run. Inert with ``depth_source`` "fine"."""

    abort_max_restarts: int = 2
    """How many times a run may abandon its attempt and start over.

    Shared by approach timeouts and standoff batches past
    ``approach_distance``; the arm returns to its default pose (``RECOVERING``).
    0 turns both aborts into plain give-ups."""
    recover_timeout_s: float = 10.0
    """Restart the approach anyway if the arm has not reached its default pose
    within this [s]."""
    complete_hold_s: float = 2.0
    """Seconds the finalize pose must be held, inside
    ``accumulate_settle_tol_mm``, before the run is logged as complete. The arm
    then holds."""


class TargetPoseAccumulator:
    """Batch-averages anchored predictions into a pelvis-frame target estimate.

    Stationary-tip counterpart of
    :class:`~g1_control.vla.tip_estimate.TipTargetEstimator`: a plain mean of
    every fresh prediction, plus its dispersion for the convergence check.
    """

    #: Accepted samples before outlier rejection starts.
    _MIN_REFERENCE = 3

    def __init__(self, max_outlier_m: float = float("inf")) -> None:
        self._max_outlier_m = float(max_outlier_m)
        self.reset()

    def reset(self) -> None:
        self._positions: list[np.ndarray] = []
        self._rot6ds: list[np.ndarray] = []
        self._last_seq: int | None = None
        self._rejected = 0

    def add(self, anchored: tuple | None) -> bool:
        """Fold in the latest anchored prediction; True if it was kept."""
        if anchored is None:
            return False
        pred_9d, anchor_tip_9d, _stamp, seq = anchored
        if seq == self._last_seq:
            return False
        self._last_seq = seq

        target_9d = anchored_target_9d(pred_9d, anchor_tip_9d)
        if self._is_outlier(target_9d[:3]):
            self._rejected += 1
            return False
        self._positions.append(target_9d[:3])
        self._rot6ds.append(target_9d[3:9])
        return True

    def _is_outlier(self, position: np.ndarray) -> bool:
        """Whether a sample is too far from the component-wise median of the
        accepted ones (median, so outliers cannot widen their own acceptance)."""
        if len(self._positions) < self._MIN_REFERENCE:
            return False
        reference = np.median(np.asarray(self._positions), axis=0)
        return float(np.linalg.norm(position - reference)) > self._max_outlier_m

    @property
    def rejected(self) -> int:
        """Samples discarded as not belonging to the cluster."""
        return self._rejected

    @property
    def count(self) -> int:
        return len(self._positions)

    @property
    def last_seq(self) -> int | None:
        """Sequence number of the newest prediction folded in, if any
        (see ``TipTargetEstimator.seed``)."""
        return self._last_seq

    def position(self) -> np.ndarray | None:
        """Mean target position (pelvis frame), or None before any sample."""
        if not self._positions:
            return None
        return np.mean(self._positions, axis=0)

    def pose_9d(self) -> np.ndarray | None:
        """Mean target pose (pelvis frame), or None before any sample.

        rot6d columns are averaged and re-orthonormalised (Gram-Schmidt),
        valid for small orientation spread.
        """
        if not self._positions:
            return None
        rotation = rot6d_to_matrix(np.mean(self._rot6ds, axis=0))
        return np.concatenate([self.position(), rotation[:, 0], rotation[:, 1]])

    def position_spread(self) -> float:
        """RMS distance of the samples from their mean [m] (0 for < 2 samples)."""
        return rms_spread(self._positions)

    def rotation_spread_deg(self) -> float:
        """RMS geodesic angle of the samples from the mean rotation [deg]."""
        if len(self._rot6ds) < 2:
            return 0.0
        mean_rot = rot6d_to_matrix(np.mean(self._rot6ds, axis=0))
        angles = []
        for rot6d in self._rot6ds:
            relative = mean_rot.T @ rot6d_to_matrix(rot6d)
            cos = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(np.arccos(cos))
        return float(np.degrees(np.sqrt(np.mean(np.square(angles)))))

    def converged(self, min_samples: int, max_spread_m: float) -> bool:
        """Whether the estimate is ready to plan on."""
        return self.count >= max(
            1, int(min_samples)
        ) and self.position_spread() <= float(max_spread_m)


class LiveTargetTracker:
    """One live target estimate and the limited correction it drives.

    Chain: anchored prediction -> depth donation -> plausibility gate -> EMA ->
    rate-limited offset (:class:`TargetCorrection`). The planner runs two: the
    applied one and a shadow that differs only in its donor.
    """

    def __init__(
        self,
        *,
        correction,
        donor: "DepthDonor | None",
        ema_alpha: float,
        gate_m: float,
        max_age_s: float,
        logger=None,
        label: str = "",
    ) -> None:
        # Lazy import: g1_control.utils imports from this package.
        from g1_control.vla.tip_estimate import TipTargetEstimator

        self._correction = correction
        self._donor = donor
        self._estimator = TipTargetEstimator(ema_alpha=ema_alpha)
        self._gate_m = float(gate_m)
        self._max_age_s = float(max_age_s)
        #: None silences per-run logging (used by the shadow tracker).
        self._logger = logger
        self._label = label
        self._planned_target: np.ndarray | None = None
        self._gated_out = 0
        self._last_t: float | None = None

    def reset(self, estimate_9d, last_seq: int | None) -> None:
        """Anchor to the estimate a plan was just built from.

        Seeds the EMA with it; ``last_seq`` prevents re-counting the sample
        that estimate already consumed.
        """
        estimate_9d = np.asarray(estimate_9d, dtype=float)
        self._correction.reset(_pin().SE3(pose_9d_to_mat(estimate_9d)))
        self._estimator.reset()
        if self._donor is not None:
            self._donor.reset()
        self._planned_target = estimate_9d[:3].copy()
        self._gated_out = 0
        self._estimator.seed(estimate_9d, last_seq)
        self._last_t = None

    def update(self, now: float, fine_anchored) -> None:
        """Fold this tick's prediction into the estimate and the correction.

        Called every tick. A stale estimate stops feeding the correction, which
        then holds its last value.
        """
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        self._estimator.update(self._admit(fine_anchored, now), now)
        if self._estimator.age(now) > self._max_age_s:
            return
        estimate_9d = self._estimator.pose_9d()
        if estimate_9d is not None:
            self._correction.update(_pin().SE3(pose_9d_to_mat(estimate_9d)), dt)

    def _admit(self, fine_anchored, now: float):
        """The newest prediction, if within ``gate_m`` of the initial planned
        target; rejections are counted, and only the first is logged."""
        anchored = self._donate(fine_anchored, now)
        if anchored is None or self._planned_target is None or self._gate_m <= 0.0:
            return anchored
        position = anchored_target_9d(anchored[0], anchored[1])[:3]
        offset = float(np.linalg.norm(position - self._planned_target))
        if offset <= self._gate_m:
            return anchored
        self._gated_out += 1
        if self._gated_out == 1 and self._logger is not None:
            self._logger.info(
                f"[vla-planner]{self._tag} live estimate {offset * 1e3:.0f} mm "
                f"from the planned target (gate {self._gate_m * 1e3:.0f} mm) — "
                "not folding it into the correction. Expected once the target "
                "leaves the wrist view on the way in; further ones are counted, "
                "not logged."
            )
        return None

    def _donate(self, fine_anchored, now: float):
        """The live prediction re-ranged by the donor (identity with no donor);
        None when the donor cannot vouch for it."""
        if self._donor is None or fine_anchored is None:
            return fine_anchored
        return self._donor.fuse(fine_anchored, now).anchored

    @property
    def _tag(self) -> str:
        return f" [{self._label}]" if self._label else ""

    @property
    def corrected_target(self):
        """Planned estimate plus the limited correction in force; None before
        :meth:`reset`. Independent of whether the caller applies it."""
        return self._correction.corrected_target

    @property
    def corrected_target_9d(self):
        """:attr:`corrected_target` as a flat 9D pose, or None."""
        target = self.corrected_target
        if target is None:
            return None
        return mat_to_pose_9d(target.homogeneous)

    def apply(self, pose):
        """``pose`` displaced by the correction in force."""
        return self._correction.apply(pose)

    @property
    def applied(self) -> tuple[float, float]:
        """Correction actually in force, as (mm, deg)."""
        return self._correction.applied

    @property
    def raw(self) -> tuple[float, float]:
        """Correction the estimate is asking for before the limits, as (mm, deg)."""
        return self._correction.raw

    @property
    def gated_out(self) -> int:
        """Predictions rejected as implausible this leg."""
        return self._gated_out

    @property
    def donor(self) -> "DepthDonor | None":
        return self._donor


def _pin():
    """pinocchio, imported on first use so this module imports without it."""
    import pinocchio

    return pinocchio


def wall_normal_of(estimate_9d: np.ndarray) -> np.ndarray:
    """The estimated wall normal: the target frame's x-axis."""
    return rot6d_to_matrix(np.asarray(estimate_9d, dtype=float)[3:9])[:, 0]


def wall_up_of(estimate_9d: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """ "Above" on the wall surface: ``up`` (default pelvis +z) projected into
    the wall plane.

    Falls back to the estimate's z-axis on a near-horizontal wall.
    """
    estimate_9d = np.asarray(estimate_9d, dtype=float)
    rotation = rot6d_to_matrix(estimate_9d[3:9])
    world_up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, float)
    normal = rotation[:, 0]
    in_plane = world_up - float(np.dot(world_up, normal)) * normal
    norm = float(np.linalg.norm(in_plane))
    return in_plane / norm if norm > 1e-3 else rotation[:, 2]


def wall_lateral_of(estimate_9d: np.ndarray) -> np.ndarray:
    """Unit wall-local +y (wall-up x normal) in the estimated wall plane."""
    up = wall_up_of(estimate_9d)
    normal = wall_normal_of(estimate_9d)
    lateral = np.cross(up, normal)
    return lateral / np.linalg.norm(lateral)


def finalize_aim(
    estimate_9d: np.ndarray,
    wall_offset_m: float,
    wall_lateral_offset_m: float = 0.0,
    wall_up_offset_m: float = 0.0,
) -> np.ndarray:
    """Final tip position after applying the three wall-local offsets."""
    estimate_9d = np.asarray(estimate_9d, dtype=float)
    return (
        estimate_9d[:3]
        - wall_offset_m * wall_normal_of(estimate_9d)
        + wall_lateral_offset_m * wall_lateral_of(estimate_9d)
        + wall_up_offset_m * wall_up_of(estimate_9d)
    )


def finalize_knots(
    tip_position: np.ndarray,
    estimate_9d: np.ndarray,
    standoff_m: float,
    wall_offset_m: float,
    rise_m: float = 0.0,
    wall_lateral_offset_m: float = 0.0,
    wall_up_offset_m: float = 0.0,
) -> list[np.ndarray]:
    """Knot sequence for the finalize trajectory: tip -> standoff -> aim.

    The estimate's x-axis is the wall normal (into the wall). The standoff sits
    ``standoff_m`` back along it, raised ``rise_m`` up the wall; the aim stops
    ``wall_offset_m`` short of the target. The lateral/up offsets translate
    both standoff and aim in the wall plane.
    """
    estimate_9d = np.asarray(estimate_9d, dtype=float)
    wall_normal = wall_normal_of(estimate_9d)
    wall_up = wall_up_of(estimate_9d)
    translated_target = finalize_aim(
        estimate_9d,
        0.0,
        wall_lateral_offset_m,
        wall_up_offset_m,
    )
    standoff = translated_target - standoff_m * wall_normal
    if rise_m:
        standoff = standoff + rise_m * wall_up
    return [
        np.asarray(tip_position, dtype=float).copy(),
        standoff,
        finalize_aim(
            estimate_9d,
            wall_offset_m,
            wall_lateral_offset_m,
            wall_up_offset_m,
        ),
    ]


def drill_entry_point(target_tip: np.ndarray) -> np.ndarray:
    """Point ``(x, 0, 0)`` on the drill axis at the target's station, in the
    drill-tip frame (``drill_tip_link`` +x is the drill axis)."""
    point = np.asarray(target_tip, dtype=float).reshape(3)
    return np.array([point[0], 0.0, 0.0])


@dataclass(frozen=True)
class FinalizeResidual:
    """Post-finalize miss, split about the drill axis.

    Lateral and along-axis misses are reported separately since they can
    cancel in a plain distance.
    """

    lateral_m: float
    """Miss perpendicular to the drill axis — where the bit will enter the wall
    relative to the target."""

    along_m: float
    """Signed miss along the drill axis, intended wall offset removed:
    positive = stopped short, negative = went past."""

    distance_m: float
    """Plain ``‖target - tip‖``, wall offset included; compared against
    ``approach_distance`` for the abort test."""

    lateral_px: float | None = None
    """:attr:`lateral_m` in wrist-camera pixels of the prepared frame
    (independent of range). ~15% anisotropic because the camera views the tip
    ~31° off the drill axis. None when camera geometry is unavailable."""

    @classmethod
    def of(
        cls,
        target_tip: np.ndarray,
        wall_offset_m: float,
        lateral_px: float | None = None,
        desired_target_tip: np.ndarray | None = None,
    ) -> "FinalizeResidual":
        """Build from the measured target in the drill-tip frame (+x = axis).

        ``desired_target_tip`` is where the target should appear at a translated
        aim; defaults to ``(wall_offset_m, 0, 0)``.
        """
        point = np.asarray(target_tip, dtype=float).reshape(3)
        desired = (
            np.array([float(wall_offset_m), 0.0, 0.0])
            if desired_target_tip is None
            else np.asarray(desired_target_tip, dtype=float).reshape(3)
        )
        return cls(
            lateral_m=float(np.linalg.norm(point[1:] - desired[1:])),
            along_m=float(point[0] - desired[0]),
            distance_m=float(np.linalg.norm(point)),
            lateral_px=None if lateral_px is None else float(lateral_px),
        )
