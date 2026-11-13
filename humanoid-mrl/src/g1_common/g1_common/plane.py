# MIT License
"""Plane fitting to point clouds and ray-plane intersection.

Used to range the wrist detector's monocular ray against the wall plane
measured by the head D435. Pure numpy; the ROS side is
``g1_sensors.depth.wall_plane_node`` and deprojection is :mod:`g1_common.depth`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    """A fitted plane together with its fit quality.

    ``normal`` is unit length; its sign is arbitrary until :meth:`facing`.
    """

    point: np.ndarray
    """(3,) A point on the plane: the centroid of the fitted points."""
    normal: np.ndarray
    """(3,) Unit normal."""
    inliers: int
    """Points within the fit tolerance of the returned plane."""
    sampled: int
    """Points the fit was offered; ``inliers / sampled`` is the plane's share of the
    view."""
    rms_m: float
    """RMS distance of the inliers from the plane [m]."""

    @property
    def offset(self) -> float:
        """``d`` in ``n . x = d``."""
        return float(self.normal @ self.point)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of ``points`` (..., 3) from the plane [m], positive along the
        normal."""
        points = np.asarray(points, dtype=float)
        return (points - self.point) @ self.normal

    def facing(self, view_direction: np.ndarray) -> "Plane":
        """The same plane with its normal pointing against ``view_direction``
        (outward)."""
        direction = np.asarray(view_direction, dtype=float).reshape(3)
        if float(self.normal @ direction) > 0.0:
            return Plane(
                point=self.point,
                normal=-self.normal,
                inliers=self.inliers,
                sampled=self.sampled,
                rms_m=self.rms_m,
            )
        return self

    def transformed(self, transform: np.ndarray) -> "Plane":
        """This plane in another frame, given the rigid 4x4 ``T(other <- this)``."""
        transform = np.asarray(transform, dtype=float).reshape(4, 4)
        point = (transform @ np.append(self.point, 1.0))[:3]
        normal = transform[:3, :3] @ self.normal
        norm = float(np.linalg.norm(normal))
        return Plane(
            point=point,
            normal=normal / norm if norm > 1e-12 else normal,
            inliers=self.inliers,
            sampled=self.sampled,
            rms_m=self.rms_m,
        )

    def ray_cos(self, ray: np.ndarray) -> float:
        """|cos| between ``ray`` and the normal (1 = head-on, 0 = parallel)."""
        ray = np.asarray(ray, dtype=float).reshape(3)
        length = float(np.linalg.norm(ray))
        if length < 1e-12:
            return 0.0
        return abs(float(self.normal @ ray)) / length

    def intersect_ray(
        self, ray: np.ndarray, origin: np.ndarray | None = None
    ) -> float | None:
        """Scale ``s`` such that ``origin + s * ray`` lies on the plane; None if
        parallel.

        ``origin`` defaults to zero (the optical centre in a camera frame).
        """
        ray = np.asarray(ray, dtype=float).reshape(3)
        start = (
            np.zeros(3)
            if origin is None
            else np.asarray(origin, dtype=float).reshape(3)
        )
        along = float(self.normal @ ray)
        if abs(along) < 1e-12:
            return None
        return float(self.normal @ (self.point - start)) / along


def fit_plane(points: np.ndarray) -> Plane | None:
    """Total-least-squares plane through ``points`` (N, 3).

    Returns None for fewer than 3 points or degenerate (collinear/coincident)
    input. ``inliers`` is every point offered.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centred = points - centroid
    # full_matrices=False avoids the N x N left basis.
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    # Degenerate (coincident/collinear) points: tiny second singular value.
    if singular[0] < 1e-12 or singular[1] < 1e-9 * singular[0]:
        return None
    normal = vt[-1]
    normal = normal / float(np.linalg.norm(normal))
    distances = centred @ normal
    return Plane(
        point=centroid,
        normal=normal,
        inliers=len(points),
        sampled=len(points),
        rms_m=float(np.sqrt(np.mean(distances**2))),
    )


def fit_plane_ransac(
    points: np.ndarray,
    *,
    tolerance_m: float = 0.01,
    iterations: int = 64,
    seed: int = 0,
    up_axis: np.ndarray | None = None,
    max_tilt_deg: float = 20.0,
    min_inliers: int = 200,
) -> Plane | None:
    """The largest near-vertical plane in ``points``, or None.

    RANSAC hypotheses followed by a least-squares refit on the consensus.

    Args:
        up_axis: frame up direction; rejects planes tilted more than
            ``max_tilt_deg`` from vertical (e.g. the floor). None accepts any.
        min_inliers: minimum consensus size, in (possibly decimated) points.
        seed: fixed sampling seed, for reproducible estimates.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    sampled = len(points)
    if sampled < max(3, min_inliers):
        return None

    up = None if up_axis is None else _unit(up_axis)
    # sin(tilt from vertical) = |normal . up|.
    max_up_component = float(np.sin(np.deg2rad(max_tilt_deg)))

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(int(iterations)):
        trio = points[rng.choice(sampled, size=3, replace=False)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:  # collinear sample: no plane
            continue
        normal = normal / norm
        if up is not None and abs(float(normal @ up)) > max_up_component:
            continue
        mask = np.abs((points - trio[0]) @ normal) <= tolerance_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < min_inliers:
        return None

    # Refit on the consensus.
    refined = fit_plane(points[best_mask])
    if refined is None:
        return None
    # Re-check the tilt prior on the refined normal.
    if up is not None and abs(float(refined.normal @ up)) > max_up_component:
        return None
    return Plane(
        point=refined.point,
        normal=refined.normal,
        inliers=refined.inliers,
        sampled=sampled,
        rms_m=refined.rms_m,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` scaled to unit length; raises on a zero vector."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("axis has no direction")
    return vector / norm
