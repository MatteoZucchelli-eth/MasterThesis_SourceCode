"""Camera geometry used by the FK Calibration state."""

from dataclasses import dataclass

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics and distortion belonging to one image stream."""

    matrix: np.ndarray
    distortion: np.ndarray
    frame: str

    @classmethod
    def from_msg(cls, msg: CameraInfo, fallback_frame: str) -> "CameraModel":
        """Construct a checked camera model from CameraInfo."""
        matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("CameraInfo has non-positive focal length")
        distortion = np.asarray(msg.d, dtype=np.float64).reshape(-1)
        frame = msg.header.frame_id or fallback_frame
        return cls(matrix, distortion, frame)

    @property
    def detector_params(self) -> tuple[float, float, float, float]:
        """Return pupil_apriltags' ``fx, fy, cx, cy`` tuple."""
        return (
            float(self.matrix[0, 0]),
            float(self.matrix[1, 1]),
            float(self.matrix[0, 2]),
            float(self.matrix[1, 2]),
        )


def average_tag_poses(
    translations: list[np.ndarray], rotations: list[np.ndarray]
) -> pin.SE3:
    """Return the arithmetic position and SO(3) mean of matching samples."""
    if not translations or len(translations) != len(rotations):
        raise ValueError("translation and rotation samples must be non-empty and paired")
    translation = np.mean(np.stack(translations), axis=0)
    rotation = Rotation.from_matrix(np.stack(rotations)).mean().as_matrix()
    return pin.SE3(rotation, translation)


def project_camera_point(point_camera: np.ndarray, model: CameraModel):
    """Project a camera-frame 3D point with this stream's CameraInfo model."""
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)) or point[2] <= 1e-6:
        return None
    projected, _ = cv2.projectPoints(
        point.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        model.matrix,
        model.distortion if model.distortion.size else None,
    )
    uv = projected.reshape(2)
    if not np.all(np.isfinite(uv)) or np.any(np.abs(uv) > 1e6):
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def point_offset_from_tag(
    tag_pose: pin.SE3, below_m: float, normal_m: float = 0.0
) -> np.ndarray:
    """Return a point offset from the tag centre in the tag's own frame.

    ``below_m`` (>= 0) is along image-down +y. ``normal_m`` is along +z, which
    pupil_apriltags points into the surface; negative is towards the camera.
    """
    below_m = float(below_m)
    normal_m = float(normal_m)
    if not np.isfinite(below_m) or below_m < 0.0:
        raise ValueError("tag below-offset must be finite and non-negative")
    if not np.isfinite(normal_m):
        raise ValueError("tag normal-offset must be finite")
    return tag_pose.translation + tag_pose.rotation @ np.array(
        [0.0, below_m, normal_m]
    )


__all__ = [
    "CameraModel",
    "average_tag_poses",
    "point_offset_from_tag",
    "project_camera_point",
]
