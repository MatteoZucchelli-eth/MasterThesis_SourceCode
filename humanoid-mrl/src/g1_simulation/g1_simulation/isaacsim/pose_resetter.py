# MIT License
"""Teleport the robot and resample the wall objects on a reset trigger."""
from __future__ import annotations

import logging

import numpy as np

from g1_simulation.isaacsim import scene
from g1_simulation.isaacsim.carb_keys import is_key_press


class RobotPoseResetter:
    """Teleport the robot to its base pose and resample the target squares on keypress.

    Default key: R.  Configurable via ``IsaacSimCfg.reset_key``.
    The simulation keeps running — only positions and velocities are overwritten.

    Base pose and wall objects are sampled uniformly (see ``IsaacSimCfg``) for
    even workspace coverage. The sampled pose is exposed as ``lock_position`` /
    ``lock_orientation`` for :class:`BaseLock`.
    """

    # Nine-object draws in a small window can be accepted <2% of the time.
    _MAX_SEPARATION_ATTEMPTS = 2000

    def __init__(
        self,
        articulation,
        position,
        yaw_deg: float,
        key: str = "R",
        *,
        target_y_range: tuple[float, float] = (-0.30, 0.02),
        target_z_range: tuple[float, float] = (0.80, 1.05),
        target_min_separation: float = 0.03,
        position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw_noise_deg: float = 0.0,
    ):
        self._articulation = articulation

        # Nominal base pose; noise is sampled around it.
        self._base_position = np.array(position, dtype=np.float32)  # (3,)
        self._base_yaw_deg = yaw_deg
        self._key_name = key.upper()

        self._target_y_range = tuple(target_y_range)
        self._target_z_range = tuple(target_z_range)
        self._target_min_separation = float(target_min_separation)
        self._position_noise = np.array(position_noise, dtype=np.float32)
        self._yaw_noise_deg = float(yaw_noise_deg)

        # Current (sampled) base pose.
        self._position = self._base_position.copy()
        self._orientation = self._yaw_to_quat(self._base_yaw_deg)

        # Set by _reset() (keyboard thread), drained by the ROS bridge (physics thread);
        # rclpy should not publish from a carb callback.
        self._reset_pending = False

        # TF frame name → USD prim path (constant; resets move positions only).
        self._target_frame_to_prim_path: dict[str, str] = {
            frame: path for (path, frame, _) in scene.WALL_OBJECTS.values()
        }

        logging.info(
            f"RobotPoseResetter ready — press '{self._key_name}' to reset robot "
            "and targets with noise."
        )

    @staticmethod
    def _yaw_to_quat(yaw_deg: float) -> np.ndarray:
        """World-Z yaw (degrees) as a (w, x, y, z) quaternion."""
        half = np.deg2rad(yaw_deg) / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    def pop_reset_event(self) -> bool:
        """Consume a pending reset notification. True at most once per reset-key press."""
        if not self._reset_pending:
            return False
        self._reset_pending = False
        return True

    @property
    def target_frame_to_prim_path(self) -> dict[str, str]:
        """TF frame name (e.g. 'red_triangle_target') → its USD prim path."""
        return self._target_frame_to_prim_path

    @property
    def lock_position(self) -> np.ndarray:
        """Base position the robot currently stands at (noise included)."""
        return self._position.copy()

    @property
    def lock_orientation(self) -> np.ndarray:
        """Base orientation quaternion (w, x, y, z) the robot currently stands at."""
        return self._orientation.copy()

    def on_keyboard_input(self, e) -> None:
        if is_key_press(e, self._key_name):
            self.reset()

    def reset(self) -> None:
        """Randomize the robot base pose and resample the target squares.

        Public entry point shared by the reset key and the ROS reset request."""
        self._reset()

    def _teleport(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Move the robot to a base pose and zero every joint position/velocity."""
        import warp as wp

        self._position = np.asarray(position, dtype=np.float32)
        self._orientation = np.asarray(orientation, dtype=np.float32)
        self._articulation.set_world_poses(
            positions=wp.array([self._position], dtype=wp.float32),
            orientations=wp.array([self._orientation], dtype=wp.float32),
        )
        n_dof = self._articulation.num_dofs
        zeros_dof = wp.zeros((1, n_dof), dtype=wp.float32)
        self._articulation.set_dof_positions(zeros_dof)
        self._articulation.set_dof_velocities(zeros_dof)

    def teleport_to(self, position, yaw_deg: float) -> None:
        """Teleport the robot to an explicit base pose (evaluation scenarios).

        Updates ``lock_position`` / ``lock_orientation`` like any reset, so
        BaseLock pins the pelvis at the commanded pose."""
        self._teleport(
            np.asarray(position, dtype=np.float32), self._yaw_to_quat(yaw_deg)
        )

    def _sample_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly perturb the nominal base pose within the configured noise box."""
        offset = np.random.uniform(-self._position_noise, self._position_noise)
        yaw = self._base_yaw_deg + np.random.uniform(
            -self._yaw_noise_deg, self._yaw_noise_deg
        )
        return (self._base_position + offset).astype(np.float32), self._yaw_to_quat(yaw)

    def _sample_target_positions(self, count: int = 3) -> list[tuple[float, float]]:
        """Uniformly sample ``count`` (y, z) wall spots, rejecting overlapping draws.

        All objects are drawn jointly so every pair is at least
        ``target_min_separation`` apart. Falls back to the last (overlapping)
        sample after ``_MAX_SEPARATION_ATTEMPTS``.
        """
        for _ in range(self._MAX_SEPARATION_ATTEMPTS):
            samples = [
                (
                    float(np.random.uniform(*self._target_y_range)),
                    float(np.random.uniform(*self._target_z_range)),
                )
                for _ in range(count)
            ]
            if all(
                np.hypot(a[0] - b[0], a[1] - b[1]) >= self._target_min_separation
                for i, a in enumerate(samples)
                for b in samples[i + 1 :]
            ):
                return samples
        logging.warning(
            "RobotPoseResetter: could not separate %d objects by %s m in %d "
            "attempts — using the last (overlapping) sample. Widen "
            "target_y_range/target_z_range, lower target_min_separation, or "
            "show fewer distractors.",
            count,
            self._target_min_separation,
            self._MAX_SEPARATION_ATTEMPTS,
        )
        return samples

    def _reset(self) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        logging.info(
            "RobotPoseResetter: teleporting robot to a randomized base pose and "
            "resampling targets."
        )

        # ==========================================
        # 1. Robot Pose — base pose + uniform noise
        # ==========================================
        position, orientation = self._sample_base_pose()
        self._teleport(position, orientation)

        # ==========================================
        # 2. Wall objects — uniform (y, z) samples
        # ==========================================
        stage = omni.usd.get_context().get_stage()

        # Every colour x shape cell is placed each episode.
        cells = list(scene.WALL_OBJECTS)
        positions = self._sample_target_positions(len(cells))

        for cell, (new_y, new_z) in zip(cells, positions):
            prim_path = scene.WALL_OBJECTS[cell][0]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            UsdGeom.Imageable(prim).MakeVisible()
            # X stays locked to the wall plane; only the in-wall (y, z) moves.
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(scene.TARGET_X, new_y, new_z))
                    break

        # Each prim is permanently its (colour, shape) cell.
        self._target_frame_to_prim_path = {
            frame: path for (path, frame, _) in scene.WALL_OBJECTS.values()
        }

        # Announce last: subscribers assume robot and objects have moved.
        self._reset_pending = True
