# MIT License
"""Whether a freshly inferred action chunk is worth switching to (pure numpy)."""

from __future__ import annotations

import numpy as np

from .tip_estimate import rot6d_to_matrix


def chunk_within_deadband(
    new_9d: np.ndarray,
    published_9d: np.ndarray,
    pos_tol: float,
    rot_tol_deg: float,
) -> bool:
    """True when a fresh tip chunk is too close to the published one to switch to.

    Suppresses inter-chunk jitter (~1.9 mm while stationary). Compared against the
    last *published* chunk so sub-threshold noise cannot random-walk. Both the
    position and rotation tolerance must hold; set the tolerance above measured noise.

    Args:
        new_9d: Fresh chunk, ``(T, 9)`` xyz + rot6d.
        published_9d: The chunk currently being executed, same shape.
        pos_tol: Position tolerance in metres. ``<= 0`` disables the deadband.
        rot_tol_deg: Rotation tolerance in degrees, as a geodesic angle.

    Returns:
        True if every step of the chunk is inside both tolerances.
    """
    if pos_tol <= 0:
        return False
    if published_9d is None or new_9d.shape != published_9d.shape:
        return False

    new_9d = np.asarray(new_9d, dtype=float)
    published_9d = np.asarray(published_9d, dtype=float)

    if float(np.linalg.norm(new_9d[:, :3] - published_9d[:, :3], axis=-1).max()) > pos_tol:
        return False

    # Per-step geodesic angle; rot6d_to_matrix re-orthonormalises.
    r_new = np.stack([rot6d_to_matrix(row) for row in new_9d[:, 3:9]])
    r_pub = np.stack([rot6d_to_matrix(row) for row in published_9d[:, 3:9]])
    trace = np.trace(np.einsum("nji,njk->nik", r_new, r_pub), axis1=-2, axis2=-1)
    max_deg = float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))).max())
    return max_deg <= rot_tol_deg
