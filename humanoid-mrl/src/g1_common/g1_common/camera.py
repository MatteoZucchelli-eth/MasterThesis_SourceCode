# MIT License
"""Camera frame preparation: sensor frame -> the frame a policy is trained on.

Sim and robot run the same code, built from each stream's CameraInfo:

* aspect: both render/stream 1280x720 and are centre-cropped to a square.
* lens: the See3CAM's barrel distortion (k1 = -0.348) is removed; zero
  coefficients select a plain crop. Only Brown-Conrady models are accepted.

:attr:`FramePreparer.intrinsics` always describes :meth:`FramePreparer.prepare`'s
output (crop and undistortion change the camera matrix).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Pinhole camera matrix of a distortion-free image, plus its size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_k(cls, k, width: int, height: int) -> "PinholeIntrinsics":
        """Build from a row-major 3x3 ``CameraInfo.k`` (9 floats)."""
        k = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            width=int(width),
            height=int(height),
        )

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 camera matrix, as OpenCV wants it."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def cropped(self, width: int, height: int) -> "PinholeIntrinsics":
        """Intrinsics of a centred ``width`` x ``height`` crop (only the principal point
        moves)."""
        return PinholeIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx - (self.width - width) / 2.0,
            cy=self.cy - (self.height - height) / 2.0,
            width=int(width),
            height=int(height),
        )


def centre_crop_origin(width: int, height: int, side: int) -> tuple[int, int]:
    """Top-left (x, y) of the largest centred ``side`` x ``side`` crop."""
    return (width - side) // 2, (height - side) // 2


class FramePreparer(abc.ABC):
    """Turns one raw camera frame into the frame the policy is trained on."""

    @property
    @abc.abstractmethod
    def intrinsics(self) -> PinholeIntrinsics:
        """Intrinsics of the frames :meth:`prepare` returns."""

    @abc.abstractmethod
    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Return the prepared frame. Input is a raw HxWxC sensor frame."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.prepare(image)


class CentreCrop(FramePreparer):
    """Square centre crop (zero-copy view) for an ideal pinhole source."""

    def __init__(self, source: PinholeIntrinsics, side: int):
        self._x0, self._y0 = centre_crop_origin(source.width, source.height, side)
        self._side = int(side)
        self._intrinsics = source.cropped(side, side)

    @property
    def intrinsics(self) -> PinholeIntrinsics:
        return self._intrinsics

    def prepare(self, image: np.ndarray) -> np.ndarray:
        y0, x0, side = self._y0, self._x0, self._side
        return image[y0 : y0 + side, x0 : x0 + side]


class UndistortedCentreCrop(FramePreparer):
    """Undistort and square centre crop in a single ``cv2.remap`` pass.

    Output is rectified, so :attr:`intrinsics` fully describes it.
    """

    def __init__(
        self, source: PinholeIntrinsics, distortion, side: int, interpolation=None
    ):
        # Local import: only this implementation needs OpenCV.
        import cv2

        self._cv2 = cv2
        self._interpolation = (
            cv2.INTER_LINEAR if interpolation is None else interpolation
        )
        self._intrinsics = source.cropped(side, side)

        # The cropped camera matrix makes the remap output the crop window directly.
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            source.matrix,
            np.asarray(distortion, dtype=np.float64).reshape(-1),
            None,
            self._intrinsics.matrix,
            (int(side), int(side)),
            cv2.CV_16SC2,
        )

    @property
    def intrinsics(self) -> PinholeIntrinsics:
        return self._intrinsics

    def prepare(self, image: np.ndarray) -> np.ndarray:
        return self._cv2.remap(
            image, self._map_x, self._map_y, interpolation=self._interpolation
        )


BROWN_CONRADY_MODELS: dict[str, frozenset] = {
    "plumb_bob": frozenset({4, 5}),
    "rational_polynomial": frozenset({4, 5, 8}),
}
"""``CameraInfo.distortion_model`` values :class:`UndistortedCentreCrop` supports,
with their allowed coefficient counts.

``cv2.initUndistortRectifyMap`` implements only Brown-Conrady; other models
(e.g. ``equidistant``) are rejected by :func:`_check_distortion_model`."""

DEFAULT_DISTORTION_MODEL = "plumb_bob"
"""Assumed when a caller supplies bare coefficients with no model name (ROS default)."""


def _has_distortion(distortion) -> bool:
    """Whether a CameraInfo ``d`` vector asks for any correction at all."""
    if distortion is None:
        return False
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    return bool(coefficients.size) and bool(np.any(coefficients != 0.0))


def _check_distortion_model(distortion_model: str, distortion) -> None:
    """Reject non-zero distortion under a model the remap cannot correct."""
    name = (distortion_model or "").strip().lower()
    allowed = BROWN_CONRADY_MODELS.get(name)
    if allowed is None:
        raise ValueError(
            f"CameraInfo declares distortion_model {distortion_model!r} with "
            "non-zero coefficients, which is not a Brown-Conrady model. "
            "Undistorting it with cv2.initUndistortRectifyMap would silently "
            "produce the wrong geometry, so no frame preparer is built. "
            f"Supported: {sorted(BROWN_CONRADY_MODELS)}."
        )
    count = np.asarray(distortion, dtype=float).reshape(-1).size
    if count not in allowed:
        raise ValueError(
            f"CameraInfo declares distortion_model {name!r} with {count} "
            f"coefficients; that model carries {sorted(allowed)}. The stream is "
            "describing itself inconsistently, so its geometry cannot be trusted."
        )


def build_frame_preparer(
    intrinsics: PinholeIntrinsics,
    distortion,
    side: int,
    distortion_model: str = DEFAULT_DISTORTION_MODEL,
) -> FramePreparer:
    """Pick the cheapest preparer that makes this source model-ready.

    Zero distortion -> :class:`CentreCrop`; otherwise
    :class:`UndistortedCentreCrop` after validating ``distortion_model``.

    :raises ValueError: the source distorts by an unsupported model.
    """
    if _has_distortion(distortion):
        _check_distortion_model(distortion_model, distortion)
        return UndistortedCentreCrop(intrinsics, distortion, side)
    return CentreCrop(intrinsics, side)


def preparer_from_camera_info(msg, side: int) -> FramePreparer:
    """Build a preparer from a ``sensor_msgs/CameraInfo``."""
    return build_frame_preparer(
        PinholeIntrinsics.from_k(msg.k, msg.width, msg.height),
        msg.d,
        side,
        msg.distortion_model,
    )
