# MIT License
"""Per-frame projection of the logged 3D target into camera pixels.

* T_pelvis->camera via pinocchio FK over the logged arm joints through the
  drill URDF (same chain as robot_state_publisher); unrecorded joints stay neutral.
* T_pelvis->target from the state's drill-tip pose composed with the recorded
  ``next.tip_target_9d`` (target in the tip frame).
* Centre and outline pixels, optical depth, and wall normal in the optical
  frame. The outline is only used for the visibility check.

FK tip pose is compared with the state's tip pose per frame to surface
URDF/joint-mapping drift.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

_SRC = Path(__file__).resolve().parents[2] / "src"
for _pkg in ("g1_definitions",):
    p = str(_SRC / _pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from g1_definitions import dataset as ds
from g1_definitions.joints import G1_BASE_PR_JOINT_NAMES, JOINT_PR_SLICES

from . import camera

URDF_PATH = _SRC / "g1_state_publisher" / "urdf" / "g1_29_dof_with_hand_drill.urdf.xml"

TARGET_SIDE_M = 0.01
"""Nominal target size (g1_simulation.isaacsim.scene.TARGET_SIDE)."""

_SHAPE_SIDES = {"square": 4, "circle": 32, "triangle": 3}
_SHAPE_PHASE = {"square": math.pi / 4.0, "circle": 0.0, "triangle": math.pi / 2.0}


def target_outline(shape: str, size: float = TARGET_SIDE_M) -> np.ndarray:
    """Target-face outline in its local frame, matching Isaac Sim exactly.

    Area-matched regular polygons (area of a square with side ``size``);
    duplicated here to avoid importing Isaac/pxr.
    """
    try:
        sides = _SHAPE_SIDES[shape]
        phase = _SHAPE_PHASE[shape]
    except KeyError as exc:
        raise ValueError(
            f"unknown target shape {shape!r}; expected {tuple(_SHAPE_SIDES)}"
        ) from exc
    radius = math.sqrt(2.0 * size * size / (sides * math.sin(2.0 * math.pi / sides)))
    return np.array(
        [
            [
                0.0,
                radius * math.cos(phase + 2.0 * math.pi * i / sides),
                radius * math.sin(phase + 2.0 * math.pi * i / sides),
            ]
            for i in range(sides)
        ]
    )


def se3_from_9d(pose_9d: np.ndarray) -> pin.SE3:
    """xyz(3) + rot6d(6) -> SE3, Gram-Schmidt as in RobotModel.action_9d_to_se3."""
    xyz = np.asarray(pose_9d[:3], dtype=float)
    x_raw = np.asarray(pose_9d[3:6], dtype=float)
    y_raw = np.asarray(pose_9d[6:9], dtype=float)
    x = x_raw / np.linalg.norm(x_raw)
    y = y_raw - np.dot(x, y_raw) * x
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return pin.SE3(np.column_stack([x, y, z]), xyz)


@dataclass
class FrameProjection:
    """Label geometry of one dataset frame (pixels; metres; optical frame)."""

    uv: np.ndarray  # (2,) target centre pixel, NaN if behind the camera
    uv_outline: np.ndarray  # (N, 2) target-shape outline; visibility audit only
    depth_z: float  # optical-axis depth of the centre [m]
    wall_normal_cam: np.ndarray  # (3,) target x-axis in the optical frame
    tip_target_dist: float  # the recorded ||tip->target|| [m]
    in_front: bool  # centre has positive optical depth
    in_fov: bool  # centre inside the image (no margin)
    fk_tip_err_m: float  # |FK tip - state tip| consistency check [m]


class TargetProjector:
    """FK + pinhole projection for one dataset's frames.

    The :mod:`.camera` frame is already optical; no extra change of basis.
    """

    def __init__(self, urdf_path: str | Path = URDF_PATH) -> None:
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        self._data = self._model.createData()
        self._frame_name = camera.CAMERA_FRAME
        if not self._model.existFrame(self._frame_name):
            raise ValueError(
                f"URDF {urdf_path} lacks '{self._frame_name}' — need the drill "
                "URDF variant."
            )
        if not self._model.existFrame(ds.DRILL_TIP_FRAME):
            raise ValueError(
                f"URDF {urdf_path} lacks '{ds.DRILL_TIP_FRAME}' — need the drill "
                "URDF variant."
            )
        self._cam_fid = self._model.getFrameId(self._frame_name)
        self._tip_fid = self._model.getFrameId(ds.DRILL_TIP_FRAME)

        # State column -> pinocchio q index for recorded groups; others stay neutral.
        self._q_map: list[tuple[int, int]] = []
        for group in ("left_arm", "right_arm", "waist"):
            if group not in ds.STATE_SLICES:
                continue
            state_lo, _ = ds.STATE_SLICES[group]
            pr_lo, pr_hi = JOINT_PR_SLICES[group]
            for offset, name in enumerate(G1_BASE_PR_JOINT_NAMES[pr_lo:pr_hi]):
                jid = self._model.getJointId(name)
                if jid == 0 or jid >= self._model.njoints:
                    raise ValueError(f"joint '{name}' not in URDF")
                q_idx = self._model.joints[jid].idx_q
                self._q_map.append((state_lo + offset, q_idx))

    def _fk(self, state: np.ndarray) -> tuple[pin.SE3, pin.SE3]:
        """(T_pelvis_optical, T_pelvis_tip) at the frame's measured joints."""
        q = pin.neutral(self._model)
        for state_col, q_idx in self._q_map:
            q[q_idx] = state[state_col]
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return (
            self._data.oMf[self._cam_fid].copy(),
            self._data.oMf[self._tip_fid].copy(),
        )

    def project_frame(
        self, state: np.ndarray, tip_target_9d: np.ndarray, shape: str = "square"
    ) -> FrameProjection:
        """Project one frame's recorded target into wrist-camera pixels."""
        t_pelvis_cam, t_pelvis_tip_fk = self._fk(state)

        lo, hi = ds.STATE_SLICES[ds.RIGHT_EEF_KEY]
        t_pelvis_tip = se3_from_9d(state[lo:hi])
        t_tip_target = se3_from_9d(tip_target_9d)
        t_pelvis_target = t_pelvis_tip * t_tip_target

        t_cam_target = t_pelvis_cam.inverse() * t_pelvis_target
        center_cam = t_cam_target.translation
        outline_target = target_outline(shape)
        outline_cam = (t_cam_target.rotation @ outline_target.T).T + center_cam

        uv, in_front = camera.project(center_cam)
        uv_outline, _ = camera.project(outline_cam)
        return FrameProjection(
            uv=uv[0],
            uv_outline=uv_outline,
            depth_z=float(center_cam[2]),
            wall_normal_cam=t_cam_target.rotation[:, 0].copy(),
            tip_target_dist=float(np.linalg.norm(t_tip_target.translation)),
            in_front=bool(in_front[0]),
            in_fov=bool(camera.in_image(uv)[0]),
            fk_tip_err_m=float(
                np.linalg.norm(t_pelvis_tip_fk.translation - t_pelvis_tip.translation)
            ),
        )
