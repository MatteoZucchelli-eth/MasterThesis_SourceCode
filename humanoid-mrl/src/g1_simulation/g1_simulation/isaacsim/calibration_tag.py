# MIT License
"""Geometry for the head-camera calibration tag."""

import numpy as np

TAG_FRAME = "head_calibration_tag"
TAG_ID = 0
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.12
TAG_CENTRE_YZ = (-0.10, 0.82)
QUIET_CELLS = 2
BACKING_INSET_M = 0.001
PATTERN_INSET_M = 0.002
PRIM_PATH = "/World/HeadCalibrationTag"


def _apriltag():
    from g1_common import apriltag

    return apriltag


def cell_pitch_m(family: str = TAG_FAMILY, size_m: float = TAG_SIZE_M) -> float:
    return size_m / _apriltag().cells_per_side(family)


def plaque_side_m(family: str = TAG_FAMILY, size_m: float = TAG_SIZE_M) -> float:
    return cell_pitch_m(family, size_m) * (
        _apriltag().cells_per_side(family) + 2 * QUIET_CELLS
    )


def backing_quad(
    centre_yz=TAG_CENTRE_YZ, family: str = TAG_FAMILY, size_m: float = TAG_SIZE_M
):
    half = plaque_side_m(family, size_m) / 2
    y, z = centre_yz
    return y - half, y + half, z - half, z + half


def black_cell_quads(
    tag_id: int = TAG_ID,
    centre_yz=TAG_CENTRE_YZ,
    family: str = TAG_FAMILY,
    size_m: float = TAG_SIZE_M,
):
    cells = _apriltag().tag_cells(tag_id, family)
    pitch = cell_pitch_m(family, size_m)
    n = len(cells)
    quads = []
    for row, column in zip(*cells.nonzero()):
        y = centre_yz[0] + (n / 2 - column - 0.5) * pitch
        z = centre_yz[1] + (n / 2 - row - 0.5) * pitch
        half = pitch / 2
        quads.append((y - half, y + half, z - half, z + half))
    return quads


def quaternion_from_rpy_deg(rpy_deg) -> np.ndarray:
    half = np.radians(rpy_deg) / 2
    (sr, sp, sy), (cr, cp, cy) = np.sin(half), np.cos(half)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def quaternion_multiply(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quaternion_rotate(quaternion_xyzw, vector) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    conjugate = np.array(
        [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
    )
    pure = np.array([*np.asarray(vector, dtype=float), 0.0])
    return quaternion_multiply(quaternion_multiply(quaternion, pure), conjugate)[:3]


def perturbed_optical_quaternion(base_xyzw, rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    return quaternion_multiply(base_xyzw, quaternion_from_rpy_deg(rpy_deg))
