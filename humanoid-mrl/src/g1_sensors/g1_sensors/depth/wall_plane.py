# MIT License
"""Wall plane estimation from one depth frame, ROS-free.

:mod:`.wall_plane_node` is only an adapter; depth band, decimation, verticality
prior and consensus thresholds all live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_common.camera import PinholeIntrinsics
from g1_common.depth import deproject, transform_points
from g1_common.plane import fit_plane_ransac, Plane


@dataclass(frozen=True)
class WallEstimatorConfig:
    """What the fit is allowed to consider, and what it must achieve.

    Defaults are sized for the head d435 at standing distance; each is a node
    ROS parameter.
    """

    stride: int = 4
    """Take every n-th row and column (640x480 -> ~19k points); read together
    with ``min_inliers``."""
    min_depth_m: float = 0.3
    """Ignore nearer readings (also drops zero 'no reading' pixels); ~D435 minimum
    range."""
    max_depth_m: float = 4.0
    """Ignore further readings: competing planes, and stereo noise grows with range."""
    tolerance_m: float = 0.01
    """RANSAC inlier distance, above D435 noise at this range."""
    iterations: int = 64
    """RANSAC hypotheses; a margin for cluttered scenes."""
    max_tilt_deg: float = 20.0
    """How far the wall's normal may sit out of horizontal (rejects the floor)."""
    min_inliers: int = 500
    """Points that must agree before a plane is published (~3% of the view at
    ``stride`` 4); rejects degenerate fits."""
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The reference frame's up direction, for the tilt prior. ``+z`` in the
    pelvis frame."""
    depth_bias_m: float = 0.0
    """Added to every valid depth reading before deprojection [m].

    Bench-measured sensor range error; positive moves points further from the
    camera. Zero (no-match) pixels are not biased."""

    def as_up_axis(self) -> np.ndarray | None:
        """The up axis as an array, or None when the prior is switched off.

        An all-zero axis disables the verticality prior.
        """
        axis = np.asarray(self.up_axis, dtype=float).reshape(3)
        if float(np.linalg.norm(axis)) < 1e-9:
            return None
        return axis


@dataclass(frozen=True)
class WallEstimate:
    """One frame's answer: the plane, and the cloud it was read from."""

    plane: Plane | None
    """None when no surface in view could be the wall; see ``reason``."""
    points: np.ndarray
    """(N, 3) The decimated, depth-banded cloud in the reference frame.

    Returned even on failure, to distinguish a missing stream (empty) from a
    scene with no wall.
    """
    reason: str = ""
    """Why :attr:`plane` is None. Empty when it is not."""


class WallPlaneEstimator:
    """Fits the wall from one depth frame, config held once.

    Stateless between frames: each plane is a measurement of its own frame;
    smoothing is left to the consumer.
    """

    def __init__(self, config: WallEstimatorConfig | None = None) -> None:
        self._cfg = config or WallEstimatorConfig()

    @property
    def config(self) -> WallEstimatorConfig:
        return self._cfg

    def estimate(
        self,
        depth_m: np.ndarray,
        intrinsics: PinholeIntrinsics,
        reference_from_camera: np.ndarray,
    ) -> WallEstimate:
        """Fit the wall in ``depth_m``, reported in the reference frame.

        ``reference_from_camera`` is the 4x4 ``T(reference <- camera_optical)``
        at this frame's stamp. The cloud is transformed before fitting so the
        verticality prior has a fixed up axis.
        """
        cfg = self._cfg
        depth_m = np.asarray(depth_m, dtype=float)
        if cfg.depth_bias_m:
            depth_m = np.where(depth_m > 0.0, depth_m + cfg.depth_bias_m, depth_m)
        points = deproject(
            depth_m,
            intrinsics,
            stride=cfg.stride,
            min_depth_m=cfg.min_depth_m,
            max_depth_m=cfg.max_depth_m,
        )
        if not len(points):
            return WallEstimate(
                None,
                points.reshape(0, 3),
                f"no depth readings between {cfg.min_depth_m:.2f} and "
                f"{cfg.max_depth_m:.2f} m",
            )
        # Optical +z carried into the reference frame; picks the normal facing the
        # sensor.
        view = reference_from_camera[:3, :3] @ np.array([0.0, 0.0, 1.0])
        points = transform_points(points, reference_from_camera)

        plane = fit_plane_ransac(
            points,
            tolerance_m=cfg.tolerance_m,
            iterations=cfg.iterations,
            up_axis=cfg.as_up_axis(),
            max_tilt_deg=cfg.max_tilt_deg,
            min_inliers=cfg.min_inliers,
        )
        if plane is None:
            return WallEstimate(
                None,
                points,
                f"no plane within {cfg.max_tilt_deg:.0f} deg of vertical had "
                f"{cfg.min_inliers} of the {len(points)} points behind it",
            )
        return WallEstimate(plane.facing(view), points)
