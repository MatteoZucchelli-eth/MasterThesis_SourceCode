# MIT License
"""Replace a monocular detection's range with one from another source.

A wrist detection is a ray (accurate bearing) plus a depth (inaccurate). The
donor keeps the ray and intersects it with a wall plane; :class:`StereoDepthDonor`
uses the plane measured by the head d435 (g1_sensors' wall_plane_node).

Pure numpy (no ROS, no pinocchio).
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass

import numpy as np

from g1_common.plane import Plane

from .handover import SeqFreshness
from .tip_estimate import pose_9d_to_mat

DEPTH_SOURCE_FINE = "fine"
"""Accumulate the close-range detector's estimate as it comes."""

DEPTH_SOURCE_STEREO = "stereo"
"""Rebuild each detection on the wall plane the head d435 measured.

Needs the head depth stream (off by default on the robot, see
docker/head_camera.yaml); builds nothing when no plane is available."""

DEPTH_SOURCES = (DEPTH_SOURCE_FINE, DEPTH_SOURCE_STEREO)


@dataclass(frozen=True)
class DepthDonation:
    """What a :class:`DepthDonor` made of one detection."""

    anchored: tuple | None = None
    """``(pred_tip_target_9d, anchor_tip_9d, stamp, seq)`` of the detection at
    the donated range, keeping the fine sample's anchor, stamp and seq. None
    when nothing was built."""
    reason: str = ""
    """Why no sample was built; empty if one was or no detection was passed."""
    fine_depth_m: float = 0.0
    """Optical-axis depth the detector reported."""
    fused_depth_m: float = 0.0
    """Optical-axis depth the sample was rebuilt at."""
    shift_mm: float = 0.0
    """3D displacement of the target (what the ``max_shift_m`` gate judges)."""
    donor_seq: int | None = None
    """Sequence number of the range's source."""


@dataclass(frozen=True)
class Sighting:
    """One detection, decomposed into what a range donor needs from it."""

    ray: np.ndarray
    """(3,) Target position in the camera optical frame (also the ray direction)."""
    normal_cam: np.ndarray
    """(3,) The detection's own predicted wall normal, in the camera frame."""
    anchor_9d: np.ndarray
    """Drill-tip pose in the pelvis frame at the inference that produced this."""
    extrinsic: np.ndarray
    """4x4 ``T(tip <- cam)``, the same one the lift back-projected with."""
    stamp: float
    seq: int

    @property
    def depth_m(self) -> float:
        """Optical-axis depth the detector reported."""
        return float(self.ray[2])

    @property
    def cam_from_pelvis(self) -> np.ndarray:
        """4x4 ``T(cam <- pelvis)`` through this sample's own anchor."""
        return np.linalg.inv(pose_9d_to_mat(self.anchor_9d) @ self.extrinsic)


@dataclass(frozen=True)
class Ranging:
    """A subclass's answer to "how far along this ray?"."""

    scale: float | None = None
    """Multiple of :attr:`Sighting.ray` the range puts the target at. None when
    no range could be derived, in which case ``reason`` says why."""
    reason: str = ""
    donor_seq: int | None = None
    """Sequence number of whatever supplied the range, for the tally."""


class DepthDonor(abc.ABC):
    """Rebuilds a close-range detection at a range from another source.

    Subclasses implement :meth:`_range`; the base handles plausibility limits
    and a per-batch tally. Nothing is built (no fallback to the detector depth)
    when there is no extrinsic, no or stale range, a shift past
    ``max_shift_m``, or a degenerate ray.
    """

    #: Camera-frame depth below which a rebuilt point is rejected.
    _MIN_DEPTH_M = 0.02

    #: Smallest |cos| between view ray and wall normal (~78 deg off the normal).
    _MIN_RAY_COS = 0.2

    #: Named in the log lines, so a run says which range it was flown at.
    _SOURCE = "another source"

    def __init__(
        self,
        extrinsic_provider,
        max_age_s: float,
        max_shift_m: float,
    ) -> None:
        # Returns T(tip <- cam) or None; the same lookup the lifter uses.
        self._extrinsic_provider = extrinsic_provider
        self._max_age_s = float(max_age_s)
        self._max_shift_m = float(max_shift_m)
        self._freshness = SeqFreshness()
        self.reset()

    def reset(self) -> None:
        """Drop the previous batch's tally (a fresh round is a fresh batch)."""
        self._freshness.reset()
        self._fused = 0
        self._skipped = 0
        self._donor_seqs: set = set()
        self._depth_deltas: list[float] = []
        self._last_fused_seq: int | None = None
        self._last_skipped_seq: int | None = None

    # ── The template ─────────────────────────────────────────────────────────

    def fuse(self, fine: tuple | None, now: float):
        """Rebuild ``fine`` at this donor's range, or explain why it could not.

        ``now`` is the caller's clock, used only to age the range source.
        """
        self._observe(now)
        if fine is None:
            return DepthDonation()

        extrinsic = self._extrinsic_provider()
        if extrinsic is None:
            return self._skip(fine, "the camera->tip extrinsic is not up yet")

        pred_9d, anchor_9d, stamp, seq = fine
        pred_9d = np.asarray(pred_9d, dtype=float)
        extrinsic = np.asarray(extrinsic, dtype=float)
        # The detection, in the camera frame it was made in.
        target_cam = np.linalg.inv(extrinsic) @ pose_9d_to_mat(pred_9d)
        if float(np.linalg.norm(target_cam[:3, 3])) < 1e-9:
            return self._skip(fine, "the detection sits at the optical centre")

        sighting = Sighting(
            ray=target_cam[:3, 3],
            normal_cam=target_cam[:3, 0],
            anchor_9d=np.asarray(anchor_9d, dtype=float),
            extrinsic=extrinsic,
            stamp=stamp,
            seq=seq,
        )
        ranging = self._range(sighting, now)
        if ranging.scale is None:
            return self._skip(fine, ranging.reason or "no range was available")
        if ranging.scale <= 0.0:
            # Range lies behind the optical centre along this ray.
            return self._skip(
                fine,
                f"{self._SOURCE}'s range is behind the detection's ray, not "
                "along it",
            )

        fused_cam = ranging.scale * sighting.ray
        if float(fused_cam[2]) < self._MIN_DEPTH_M:
            return self._skip(
                fine,
                f"the donated range puts the target {float(fused_cam[2]) * 1e3:.0f} "
                "mm from the lens",
            )
        shift_m = float(np.linalg.norm(fused_cam - sighting.ray))
        if shift_m > self._max_shift_m:
            return self._skip(
                fine,
                f"{self._SOURCE} and the detector disagree by "
                f"{shift_m * 1e3:.0f} mm — further than a shared target explains",
            )

        # Only the position moves; orientation stays the detection's own.
        fused_tip = (extrinsic @ np.append(fused_cam, 1.0))[:3]
        donation = DepthDonation(
            anchored=(
                np.concatenate([fused_tip, pred_9d[3:9]]),
                sighting.anchor_9d,
                stamp,
                seq,
            ),
            fine_depth_m=sighting.depth_m,
            fused_depth_m=float(fused_cam[2]),
            shift_mm=shift_m * 1e3,
            donor_seq=ranging.donor_seq,
        )
        self._record(donation, seq)
        return donation

    # ── What subclasses answer ───────────────────────────────────────────────

    @abc.abstractmethod
    def _range(self, sighting: Sighting, now: float) -> Ranging:
        """How far along ``sighting.ray`` this donor's range puts the target."""

    @abc.abstractmethod
    def _observe(self, now: float) -> None:
        """Advance the source's freshness clock; called every tick, detection or not."""

    def _plane_scale(self, sighting: Sighting, plane: Plane) -> Ranging:
        """Intersect the detection's ray with ``plane`` (camera frame)."""
        if plane.ray_cos(sighting.ray) < self._MIN_RAY_COS:
            return Ranging(reason="the wall is too edge-on to the detection's ray")
        scale = plane.intersect_ray(sighting.ray)
        if scale is None:
            return Ranging(reason="the detection's ray does not meet the wall")
        return Ranging(scale=scale)

    # ── Tally ────────────────────────────────────────────────────────────────

    def _skip(self, fine: tuple, reason: str) -> DepthDonation:
        """Build nothing; count once per detection (it is retried on later ticks)."""
        seq = fine[3]
        if seq != self._last_skipped_seq:
            self._last_skipped_seq = seq
            self._skipped += 1
        return DepthDonation(reason=reason)

    def _record(self, donation: DepthDonation, seq: int) -> None:
        """Tally a built sample, once per detection (see :meth:`_skip`)."""
        if seq == self._last_fused_seq:
            return
        self._last_fused_seq = seq
        self._fused += 1
        self._donor_seqs.add(donation.donor_seq)
        self._depth_deltas.append(donation.fused_depth_m - donation.fine_depth_m)

    @property
    def fused(self) -> int:
        """Detections rebuilt at the donated range this batch."""
        return self._fused

    @property
    def skipped(self) -> int:
        """Detections no sample could be built from this batch."""
        return self._skipped

    @property
    def donor_count(self) -> int:
        """Distinct range readings the batch rests on (range scatter ~ 1/sqrt of
        this)."""
        return len(self._donor_seqs)

    def summary(self) -> str:
        """One line on what the donor did, for the batch's log."""
        if not self._fused:
            return f"no sample took {self._SOURCE}'s range ({self._skipped} skipped)"
        mean_delta_mm = float(np.mean(self._depth_deltas)) * 1e3
        parts = [
            f"{self._fused} samples took {self._SOURCE}'s range from "
            f"{self.donor_count} distinct estimates, moving depth "
            f"{mean_delta_mm:+.1f} mm on average"
        ]
        if self._skipped:
            parts.append(f"{self._skipped} skipped")
        return "; ".join(parts)


class StereoDepthDonor(DepthDonor):
    """Rebuilds a detection on the wall plane the head d435 measured.

    The plane arrives in a body frame (``g1_interfaces/WallPlane``) and is moved
    into the detection's camera frame through its own anchor. No fallback.
    """

    _SOURCE = "the measured wall"

    def __init__(
        self,
        extrinsic_provider,
        plane_provider,
        max_age_s: float,
        max_shift_m: float,
    ) -> None:
        # Returns the newest (Plane, seq) in the reference frame, or None.
        self._plane_provider = plane_provider
        super().__init__(extrinsic_provider, max_age_s, max_shift_m)

    def _observe(self, now: float) -> None:
        """Age the plane."""
        plane = self._plane_provider()
        # SeqFreshness reads the seq from element [3].
        self._freshness.update(
            None if plane is None else (None, None, None, plane[1]), now
        )

    def _range(self, sighting: Sighting, now: float) -> Ranging:
        plane = self._plane_provider()
        if plane is None:
            return Ranging(reason="no wall plane has been measured yet")
        wall, seq = plane
        age = self._freshness.update((None, None, None, seq), now)
        if age > self._max_age_s:
            return Ranging(reason=f"the measured wall plane is {age:.1f} s old")

        ranging = self._plane_scale(
            sighting, wall.transformed(sighting.cam_from_pelvis)
        )
        if ranging.scale is None:
            return ranging
        return Ranging(scale=ranging.scale, donor_seq=seq)


class WallPlaneSlot:
    """Thread-safe newest measured wall plane with a never-resetting sequence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plane: Plane | None = None
        self._frame: str = ""
        self._seq = 0

    def publish(self, plane: Plane, frame_id: str = "") -> None:
        with self._lock:
            self._plane = plane
            self._frame = frame_id
            self._seq += 1

    def reset(self) -> None:
        """Forget the plane, keeping the sequence (called on state re-entry)."""
        with self._lock:
            self._plane = None

    def latest(self) -> tuple[Plane, int] | None:
        """``(plane, seq)``, or None before the first plane arrives."""
        with self._lock:
            if self._plane is None:
                return None
            return (self._plane, self._seq)

    @property
    def frame_id(self) -> str:
        """Frame of the newest plane, for the caller's log line."""
        with self._lock:
            return self._frame


def depth_biased_9d(pred_9d, extrinsic, bias_m: float):
    """Move a tip-frame target along its camera ray by ``bias_m`` of depth."""
    if extrinsic is None or not bias_m:
        return pred_9d
    pred_9d = np.asarray(pred_9d, dtype=float)
    extrinsic = np.asarray(extrinsic, dtype=float)
    origin = extrinsic[:3, 3]
    ray = pred_9d[:3] - origin
    # Column 2 is camera optical +z expressed in the tip frame.
    depth = float(extrinsic[:3, 2] @ ray)
    if depth <= 0.0 or depth + float(bias_m) <= 0.0:
        return pred_9d
    biased = pred_9d.copy()
    biased[:3] += ray * (float(bias_m) / depth)
    return biased


class DepthBiasedSource:
    """A target source whose positions include a camera-depth calibration."""

    def __init__(self, client, bias_m: float, extrinsic_provider) -> None:
        self._client = client
        self._bias_m = float(bias_m)
        self._extrinsic_provider = extrinsic_provider

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_tip_target(self):
        pred_9d = self._client.get_tip_target()
        if pred_9d is None:
            return None
        return self._bias(pred_9d)

    def get_tip_target_anchored(self):
        anchored = self._client.get_tip_target_anchored()
        if anchored is None:
            return None
        pred_9d, anchor_9d, stamp, seq = anchored
        return (self._bias(pred_9d), anchor_9d, stamp, seq)

    def _bias(self, pred_9d):
        return depth_biased_9d(pred_9d, self._extrinsic_provider(), self._bias_m)
