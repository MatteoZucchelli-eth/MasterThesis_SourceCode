# MIT License
"""Pure trajectory math for the scripted drill-tip approach.

No ROS or robot-model dependencies; unit-tested in test/test_auto_collect_state.py.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin


def quintic(u: float) -> float:
    """Minimum-jerk time scaling: s(0)=0, s(1)=1, zero velocity+accel at both ends."""
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def quintic_hermite(tau: float, p1: float, v0: float, v1: float) -> float:
    """Quintic Hermite from 0 to ``p1`` over ``tau`` in [0, 1].

    Endpoint velocities are ``v0`` and ``v1`` (per unit tau); endpoint accelerations
    are pinned to zero, so chaining two of these with a shared velocity gives a C2
    schedule — continuous position, speed and acceleration across the knot.

    With ``v0 = v1 = 0`` this reduces exactly to :func:`quintic` scaled by ``p1``.
    """
    t2 = tau * tau
    t3 = t2 * tau
    t4 = t3 * tau
    t5 = t4 * tau
    h_p1 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h_v0 = tau - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h_v1 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    return p1 * h_p1 + v0 * h_v0 + v1 * h_v1


def unit(v: np.ndarray) -> np.ndarray:
    """Normalise ``v``, returning a zero vector for a degenerate (zero-length) input."""
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def _slerp_direction(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Spherically interpolate between unit vectors ``a`` (s=0) and ``b`` (s=1)."""
    dot = float(np.clip(a @ b, -1.0, 1.0))
    angle = np.arccos(dot)
    if np.sin(angle) < 1e-9:
        # Parallel (anti-parallel does not occur: the drill faces the target).
        return b.copy() if s >= 1.0 else a.copy()
    return (np.sin((1.0 - s) * angle) * a + np.sin(s * angle) * b) / np.sin(angle)


def levelled_drill_rotation(
    drill_axis: np.ndarray,
    up: np.ndarray,
    roll_rad: float = 0.0,
) -> np.ndarray | None:
    """Tip rotation on ``drill_axis`` with the drill held upright about it.

    Pins the roll: tip y is set perpendicular to ``up``, so the handle (tip -z)
    hangs straight down. ``roll_rad`` then cants the frame about the drill axis.

    Returns ``None`` when ``drill_axis`` is parallel to ``up`` (roll undefined).
    """
    axis = unit(np.asarray(drill_axis, dtype=float))
    y = np.cross(np.asarray(up, dtype=float), axis)
    norm = float(np.linalg.norm(y))
    if norm < 1e-6:
        return None
    y = y / norm
    rotation = np.column_stack((axis, y, np.cross(axis, y)))
    if roll_rad:
        # Post-multiplied: cant about the frame's own x (the drill axis).
        rotation = rotation @ np.asarray(pin.exp3(np.array([roll_rad, 0.0, 0.0])))
    return rotation


def _slerp_rotation(r0: np.ndarray, r1: np.ndarray, blend: float) -> np.ndarray:
    """Geodesic interpolation from ``r0`` (blend 0) to ``r1`` (blend 1) in SO(3).

    One constant-rate turn about a single axis, so a blend that fades the whole
    orientation — axis *and* roll — never asks the IK to absorb a step.
    """
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return np.asarray(r0, dtype=float).copy()
    if blend >= 1.0:
        return np.asarray(r1, dtype=float).copy()
    omega = pin.log3(np.asarray(r0, dtype=float).T @ np.asarray(r1, dtype=float))
    return np.asarray(r0, dtype=float) @ np.asarray(pin.exp3(blend * omega))


def aligned_drill_rotation(
    drill_axis: np.ndarray,
    reference_rotation: np.ndarray,
    blend: float,
    up: np.ndarray | None = None,
    roll_rad: float = 0.0,
) -> np.ndarray:
    """Tip rotation whose x-axis (the drill axis) is aligned to ``drill_axis``.

    ``blend`` ramps from the reference rotation (0) to the aligned one (1), so
    the alignment can be faded in along the trajectory.

    Roll about the drill axis: without ``up`` it follows the reference
    rotation's y-axis; with ``up`` (the wall's up-direction) the endpoint is
    :func:`levelled_drill_rotation` (optionally canted by ``roll_rad``), reached
    by a single geodesic turn.
    """
    reference_rotation = np.asarray(reference_rotation, dtype=float)
    axis = np.asarray(drill_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    if up is not None:
        levelled = levelled_drill_rotation(axis, up, roll_rad)
        if levelled is not None:
            return _slerp_rotation(reference_rotation, levelled, blend)

    x = _slerp_direction(
        reference_rotation[:, 0], axis, float(np.clip(blend, 0.0, 1.0))
    )
    y = reference_rotation[:, 1] - (reference_rotation[:, 1] @ x) * x
    y = y / np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


class CubicBezier:
    """A cubic Bézier segment, sampled by arc length rather than by parameter.

    An arc-length lookup table (built once) makes
    ``point_at_arclength_fraction`` advance uniformly along the path. Control
    points evenly spaced on the chord give an exact straight line.
    """

    # Arc-length LUT resolution; speed ripple ~0.02%.
    _LUT_SAMPLES = 2049

    def __init__(self, p0, p1, p2, p3) -> None:
        self.control_points = np.stack([p0, p1, p2, p3]).astype(np.float64)

        self._lut_s = np.linspace(0.0, 1.0, self._LUT_SAMPLES)
        points = np.array([self._at(s) for s in self._lut_s])
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._lut_arclength = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self._lut_arclength[-1])

    def _at(self, s: float) -> np.ndarray:
        """Point at Bézier parameter ``s`` (exact at the endpoints)."""
        r = 1.0 - s
        basis = np.array([r * r * r, 3.0 * r * r * s, 3.0 * r * s * s, s * s * s])
        return basis @ self.control_points

    def tangent_at(self, s: float) -> np.ndarray:
        """Unit tangent at Bézier parameter ``s`` (direction of travel)."""
        s = float(np.clip(s, 0.0, 1.0))
        r = 1.0 - s
        p = self.control_points
        derivative = 3.0 * (
            r * r * (p[1] - p[0]) + 2.0 * r * s * (p[2] - p[1]) + s * s * (p[3] - p[2])
        )
        return unit(derivative)

    def point_at_arclength_fraction(self, fraction: float) -> np.ndarray:
        """Point ``fraction`` of the way along the curve *by distance*, in [0, 1]."""
        if self.length <= 1e-12:
            return self.control_points[3].copy()
        distance = float(np.clip(fraction, 0.0, 1.0)) * self.length
        s = float(np.interp(distance, self._lut_arclength, self._lut_s))
        return self._at(s)


class TipTrajectory:
    """Time-parameterised drill-tip path through a set of knots, in the pelvis frame.

    Constant orientation; each segment has its own duration (brisk transit, slow
    insertion so the arm integral nulls the droop).

    * **Geometric (G1).** Segments are cubic Béziers; each ends tangent to the
      next segment's chord, easing the transit into the insertion axis. The last
      segment stays an exact straight line.
    * **Temporal.** Arc length follows a quintic Hermite per segment with a
      shared non-zero speed at interior knots (continuous position, speed and
      tangential acceleration).

    Not G2: a small centripetal step (~1.8 mm/s^2) remains at the knot.
    Interior knot speed is the slower adjoining leg's mean speed, keeping each
    schedule monotone. ``u = 1`` lands exactly on the final knot.
    """

    def __init__(
        self,
        knots: list[np.ndarray],
        rotation: np.ndarray,
        segment_durations: list[float],
    ) -> None:
        self.knots = [np.asarray(k, dtype=np.float64) for k in knots]
        self.rotation = np.asarray(rotation, dtype=np.float64)
        if len(segment_durations) != len(self.knots) - 1:
            raise ValueError(
                f"expected {len(self.knots) - 1} segment durations for "
                f"{len(self.knots)} knots, got {len(segment_durations)}"
            )
        self.segment_durations = [float(d) for d in segment_durations]
        self.duration = float(sum(self.segment_durations))
        self._elapsed_at_knot = np.concatenate(
            [[0.0], np.cumsum(self.segment_durations)]
        )

        self.segments = self._build_segments()
        self.length = float(sum(seg.length for seg in self.segments))
        self.knot_speeds = self._build_knot_speeds()

    def _build_segments(self) -> list[CubicBezier]:
        """Bend each segment so it exits along the next segment's chord direction."""
        chords = [b - a for a, b in zip(self.knots[:-1], self.knots[1:])]
        directions = [unit(c) for c in chords]

        segments = []
        for i, chord in enumerate(chords):
            # Enter along own chord, leave along the next one; handles 1/3 chord.
            start_dir = directions[i]
            end_dir = directions[i + 1] if i + 1 < len(chords) else directions[i]
            handle = float(np.linalg.norm(chord)) / 3.0

            start = self.knots[i]
            end = self.knots[i + 1]
            segments.append(
                CubicBezier(
                    start,
                    start + handle * start_dir,
                    end - handle * end_dir,
                    end,
                )
            )
        return segments

    def _build_knot_speeds(self) -> list[float]:
        """Path speed [m/s] at each knot: zero at the ends, and the slower
        adjoining leg's mean speed at interior knots."""
        mean_speeds = [
            seg.length / duration if duration > 1e-9 else 0.0
            for seg, duration in zip(self.segments, self.segment_durations)
        ]
        speeds = [0.0]
        for before, after in zip(mean_speeds[:-1], mean_speeds[1:]):
            speeds.append(min(before, after))
        speeds.append(0.0)
        return speeds

    def pose_at(self, u: float) -> pin.SE3:
        """Tip pose at normalised time ``u`` in [0, 1] over the whole trajectory."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.duration <= 1e-9:
            return pin.SE3(self.rotation.copy(), self.knots[-1].copy())

        elapsed = u * self.duration
        # Segment containing ``elapsed`` (clamped to the last one at u == 1).
        i = min(
            int(np.searchsorted(self._elapsed_at_knot[1:], elapsed, side="left")),
            len(self.segments) - 1,
        )
        segment = self.segments[i]
        span = self.segment_durations[i]
        tau = 0.0 if span < 1e-9 else (elapsed - self._elapsed_at_knot[i]) / span
        tau = float(np.clip(tau, 0.0, 1.0))

        if segment.length <= 1e-12:
            return pin.SE3(self.rotation.copy(), segment.control_points[3].copy())

        # Hermite endpoint velocities are per unit tau, so scale m/s by the span.
        distance = quintic_hermite(
            tau,
            segment.length,
            self.knot_speeds[i] * span,
            self.knot_speeds[i + 1] * span,
        )
        position = segment.point_at_arclength_fraction(distance / segment.length)
        return pin.SE3(self.rotation.copy(), position)
