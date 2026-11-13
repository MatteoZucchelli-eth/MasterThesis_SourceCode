# MIT License
"""Rigid-transform fitting used by the head-camera calibration node."""

from dataclasses import dataclass, field, replace

import numpy as np

WALL_FROM_TAG = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def signed_plane_standoff(
    point: np.ndarray, plane_point: np.ndarray, outward_normal: np.ndarray
) -> float:
    """Signed perpendicular distance from ``point`` to a plane [m].

    ``outward_normal`` points out of the wall (towards the camera), so the result
    is positive in front of the wall. The normal is normalised here.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    plane_point = np.asarray(plane_point, dtype=float).reshape(3)
    normal = np.asarray(outward_normal, dtype=float).reshape(3)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("a wall plane cannot have a zero normal")
    return float((point - plane_point) @ (normal / length))


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest angle between two directed plane normals [deg]."""
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    a_length = float(np.linalg.norm(a))
    b_length = float(np.linalg.norm(b))
    if a_length < 1e-12 or b_length < 1e-12:
        raise ValueError("cannot compare a zero normal")
    cosine = float(np.clip((a @ b) / (a_length * b_length), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def orthonormalise(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix).reshape(3, 3))
    u[:, -1] *= np.sign(np.linalg.det(u @ vt))
    return u @ vt


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a).T @ np.asarray(b)
    skew = relative - relative.T
    sine = np.linalg.norm([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2
    cosine = (np.trace(relative) - 1) / 2
    return float(np.degrees(np.arctan2(sine, cosine)))


def measured_wall_rotation(
    camera_from_tag: np.ndarray, wall_from_tag: np.ndarray = WALL_FROM_TAG
) -> np.ndarray:
    return (
        np.asarray(camera_from_tag).reshape(3, 3)
        @ np.asarray(wall_from_tag).reshape(3, 3).T
    )


def corrected_optical_rotation(
    parent_from_optical: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(parent_from_optical).reshape(3, 3)
        @ np.asarray(correction).reshape(3, 3).T
    )


def corrected_optical_translation(
    parent_from_optical: np.ndarray,
    parent_to_optical: np.ndarray,
    correction_rotation: np.ndarray,
    correction_translation: np.ndarray,
) -> np.ndarray:
    corrected_rotation = corrected_optical_rotation(
        parent_from_optical, correction_rotation
    )
    return np.asarray(parent_to_optical).reshape(3) - corrected_rotation @ np.asarray(
        correction_translation
    ).reshape(3)


def corrected_mount_rotation(
    parent_from_mount: np.ndarray,
    mount_from_optical: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    optical = np.asarray(mount_from_optical).reshape(3, 3)
    return (
        np.asarray(parent_from_mount).reshape(3, 3)
        @ optical
        @ np.asarray(correction).reshape(3, 3).T
        @ optical.T
    )


def corrected_mount_translation(
    parent_from_mount: np.ndarray,
    parent_to_mount: np.ndarray,
    mount_from_optical: np.ndarray,
    mount_to_optical: np.ndarray,
    correction_rotation: np.ndarray,
    correction_translation: np.ndarray,
) -> np.ndarray:
    desired_rotation = np.asarray(parent_from_mount).reshape(
        3, 3
    ) @ corrected_optical_rotation(mount_from_optical, correction_rotation)
    desired_translation = np.asarray(parent_to_mount).reshape(3) + np.asarray(
        parent_from_mount
    ).reshape(3, 3) @ corrected_optical_translation(
        mount_from_optical,
        mount_to_optical,
        correction_rotation,
        correction_translation,
    )
    optical = np.asarray(mount_from_optical).reshape(3, 3)
    return desired_translation - desired_rotation @ optical.T @ np.asarray(
        mount_to_optical
    ).reshape(3)


@dataclass(frozen=True)
class Solution:
    rotation: np.ndarray
    samples: int
    mean_residual_deg: float
    max_residual_deg: float
    consistency: float
    translation: np.ndarray | None = None
    mean_translation_residual_m: float | None = None
    max_translation_residual_m: float | None = None

    @property
    def angle_deg(self) -> float:
        return rotation_angle_deg(self.rotation, np.eye(3))


class RotationAccumulator:
    def __init__(self) -> None:
        self._estimates: list[np.ndarray] = []
        self._h = np.zeros((3, 3))
        self.translation = TranslationAccumulator()

    def __len__(self) -> int:
        return len(self._estimates)

    @property
    def h(self) -> np.ndarray:
        return self._h.copy()

    def add(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        measured_translation: np.ndarray | None = None,
        predicted_translation: np.ndarray | None = None,
    ) -> np.ndarray:
        estimate = (
            np.asarray(measured).reshape(3, 3) @ np.asarray(predicted).reshape(3, 3).T
        )
        self._estimates.append(estimate)
        self._h += estimate
        if measured_translation is not None and predicted_translation is not None:
            self.translation.add(
                measured,
                measured_translation,
                predicted,
                predicted_translation,
            )
        return estimate

    def solve(self) -> Solution | None:
        if not self._estimates:
            return None
        rotation = orthonormalise(self._h)
        residuals = [rotation_angle_deg(value, rotation) for value in self._estimates]
        singular = np.linalg.svd(self._h, compute_uv=False)
        solution = Solution(
            rotation,
            len(self),
            float(np.mean(residuals)),
            float(np.max(residuals)),
            float(singular[-1] / singular[0]),
        )
        translation = self.translation.solve()
        if translation is None:
            return solution
        value, mean_residual, max_residual = translation
        return replace(
            solution,
            translation=value,
            mean_translation_residual_m=mean_residual,
            max_translation_residual_m=max_residual,
        )


class TranslationAccumulator:
    def __init__(self) -> None:
        self._a: list[np.ndarray] = []
        self._b: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._b)

    def add(
        self,
        measured_rotation: np.ndarray,
        measured_translation: np.ndarray,
        predicted_rotation: np.ndarray,
        predicted_translation: np.ndarray,
    ) -> None:
        measured_rotation = np.asarray(measured_rotation).reshape(3, 3)
        predicted_rotation = np.asarray(predicted_rotation).reshape(3, 3)
        measured_translation = np.asarray(measured_translation).reshape(3)
        predicted_translation = np.asarray(predicted_translation).reshape(3)

        # Both observations are head <- wall. Invert them to put the two head
        # origins in the wall frame, then stack
        #   R_wall_from_real_head t_real_head_from_urdf_head
        #       = t_wall_from_urdf_head - t_wall_from_real_head.
        wall_from_real_head = measured_rotation.T
        wall_to_urdf_head = -predicted_rotation.T @ predicted_translation
        wall_to_real_head = -measured_rotation.T @ measured_translation
        self._a.append(wall_from_real_head)
        self._b.append(wall_to_urdf_head - wall_to_real_head)

    def solve(self) -> tuple[np.ndarray, float, float] | None:
        if not self._b:
            return None
        a = np.vstack(self._a)
        b = np.concatenate(self._b)
        translation, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        residuals = np.linalg.norm(
            np.asarray(
                [
                    matrix @ translation - value
                    for matrix, value in zip(self._a, self._b)
                ]
            ),
            axis=1,
        )
        return translation, float(np.mean(residuals)), float(np.max(residuals))


@dataclass
class CalibrationSession:
    total: RotationAccumulator = field(default_factory=RotationAccumulator)
    placements: list[RotationAccumulator] = field(default_factory=list)
    _open: RotationAccumulator | None = field(default=None, init=False)

    @property
    def accumulating(self) -> bool:
        return self._open is not None

    def start(self) -> None:
        if self._open is None:
            self._open = RotationAccumulator()
            self.placements.append(self._open)

    def pause(self) -> None:
        if self._open is not None and not len(self._open):
            self.placements.remove(self._open)
        self._open = None

    def add(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        measured_translation: np.ndarray | None = None,
        predicted_translation: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self._open is None:
            return None
        self.total.add(measured, predicted, measured_translation, predicted_translation)
        return self._open.add(
            measured, predicted, measured_translation, predicted_translation
        )

    def solve(self) -> Solution | None:
        self.pause()
        return self.total.solve()

    def placement_solutions(self) -> list[Solution]:
        return [solution for p in self.placements if (solution := p.solve())]

    def spread_deg(self) -> float:
        rotations = [solution.rotation for solution in self.placement_solutions()]
        return max(
            (
                rotation_angle_deg(a, b)
                for index, a in enumerate(rotations)
                for b in rotations[index + 1 :]
            ),
            default=0.0,
        )
