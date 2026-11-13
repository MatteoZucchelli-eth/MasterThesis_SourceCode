# MIT License
"""Pin the floating base (pelvis) and the waist joints in place."""
from __future__ import annotations

import logging

import numpy as np

from g1_simulation.isaacsim.carb_keys import is_key_press
from g1_simulation.isaacsim.pose_resetter import RobotPoseResetter


class BaseLock:
    """Pin the floating base (pelvis) and the waist joints in place.

    The pelvis is pinned by a world PhysX ``FixedJoint`` at the reset pose; a
    per-step kinematic root override would corrupt joint velocities and hence PD
    damping. The waist joints are held at zero each step.

    Engaged by any lock key (default V and R, shared with the evaluation sequencer
    and pose resetter); optional release key (default none). ``enabled=False``
    makes it a no-op (walk mode).
    """

    _WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    _JOINT_PATH = "/World/robot/base_lock_fixed_joint"
    _PELVIS_PATH = "/World/robot/pelvis"

    def __init__(
        self,
        articulation,
        pose_resetter: RobotPoseResetter,
        *,
        lock_keys: tuple[str, ...] = ("V", "R"),
        release_key: str | None = None,
        enabled: bool = True,
    ):
        self._articulation = articulation
        self._pose_resetter = pose_resetter
        self._lock_keys = tuple(k.upper() for k in lock_keys)
        self._release_key = release_key.upper() if release_key else None
        self._enabled = enabled

        self._locked = False
        self._position: np.ndarray | None = None
        self._orientation: np.ndarray | None = None
        self._waist_dof_indices: list[int] | None = None
        self._can_zero_root_vel = True

        # Constraint state. _joint_active tracks whether the FixedJoint exists;
        # _fallback_teleport is set only if the joint cannot be created, so the
        # base does not collapse on an unexpected USD/PhysX layout.
        # _rebuild_requested re-authors the joint when the lock pose has moved.
        self._joint_active = False
        self._fallback_teleport = False
        self._rebuild_requested = False

        if not self._enabled:
            logging.info(
                "BaseLock disabled (walk mode) — the pelvis stays free and the "
                f"{'/'.join(self._lock_keys)} keys will not pin it."
            )
            return

        release_msg = (
            f", '{self._release_key}' to release"
            if self._release_key
            else " (no release key — locked for the whole session)"
        )
        logging.info(
            f"BaseLock ready — press {'/'.join(self._lock_keys)} to lock "
            f"pelvis + waist{release_msg}."
        )

    def _resolve_waist_indices(self) -> None:
        if self._waist_dof_indices is not None:
            return
        dof_names = getattr(self._articulation, "dof_names", None)
        if dof_names is None:
            dof_names = self._articulation._dof_names
        dof_names = list(dof_names)
        self._waist_dof_indices = [
            dof_names.index(name) for name in self._WAIST_JOINTS if name in dof_names
        ]
        missing = [n for n in self._WAIST_JOINTS if n not in dof_names]
        if missing:
            logging.warning(f"BaseLock: waist joints not found: {missing}")

    def engage(self) -> None:
        if not self._enabled:
            logging.debug("BaseLock: lock request ignored (disabled).")
            return
        self._resolve_waist_indices()
        # Use the resetter's commanded pose, not sim state. The joint is authored
        # in step() (physics thread), and re-authored if the lock pose moved.
        position = self._pose_resetter.lock_position
        orientation = self._pose_resetter.lock_orientation
        pose_moved = self._position is None or not (
            np.allclose(position, self._position)
            and np.allclose(orientation, self._orientation)
        )
        self._position = position
        self._orientation = orientation
        self._locked = True
        if pose_moved and self._joint_active:
            self._rebuild_requested = True
        logging.info("BaseLock: lock requested — pinning pelvis via a fixed joint.")

    def release(self) -> None:
        if self._locked:
            logging.info("BaseLock: release requested — removing fixed joint.")
        self._locked = False

    def on_keyboard_input(self, e) -> None:
        if any(is_key_press(e, k) for k in self._lock_keys):
            self.engage()
        elif self._release_key and is_key_press(e, self._release_key):
            self.release()

    def _create_fixed_joint(self) -> bool:
        """Author a world→pelvis FixedJoint at the lock pose. Returns success.

        The pelvis is first snapped once to the lock pose so the constraint is
        created with the body already at its anchor (no snap-back impulse).
        """
        try:
            import omni.usd
            import warp as wp
            from pxr import Gf, Sdf, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath(self._PELVIS_PATH).IsValid():
                logging.warning(
                    f"BaseLock: {self._PELVIS_PATH} not found; using teleport fallback."
                )
                return False

            # One-time snap to the exact lock pose before constraining.
            self._articulation.set_world_poses(
                positions=wp.array([self._position], dtype=wp.float32),
                orientations=wp.array([self._orientation], dtype=wp.float32),
            )

            pos = self._position
            ori = self._orientation  # [w, x, y, z]
            joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(self._JOINT_PATH))
            joint.CreateBody1Rel().SetTargets([Sdf.Path(self._PELVIS_PATH)])
            # body0 left empty → anchored to the world frame. localPose0 is the
            # lock pose in world; localPose1 is the pelvis origin (identity).
            joint.CreateLocalPos0Attr().Set(
                Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))
            )
            joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3]))
            )
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            logging.info(
                "BaseLock: pelvis pinned with a fixed joint (no per-step teleport)."
            )
            return True
        except (
            Exception
        ) as exc:  # noqa: BLE001 — USD/PhysX layout varies across versions
            logging.warning(
                f"BaseLock: fixed-joint lock failed ({exc}); using teleport fallback."
            )
            return False

    def _remove_fixed_joint(self) -> None:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(self._JOINT_PATH).IsValid():
                stage.RemovePrim(self._JOINT_PATH)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"BaseLock: could not remove fixed joint ({exc}).")

    def _zero_root_velocity(self) -> None:
        if not self._can_zero_root_vel:
            return
        import warp as wp

        zeros3 = wp.zeros((1, 3), dtype=wp.float32)
        try:
            self._articulation.set_velocities(zeros3, zeros3)
        except Exception as exc:  # noqa: BLE001 — API name varies across versions
            self._can_zero_root_vel = False
            logging.warning(
                f"BaseLock: could not zero root velocity ({exc}); holding pose only."
            )

    def step(self) -> None:
        """Physics callback — maintain the lock state while engaged.

        Manages the constraint lifecycle and the waist hold; the root is not
        written unless the teleport fallback is active.
        """
        if not self._enabled:
            return
        # A moved lock pose invalidates the existing pin. Drop it and let the next
        # step re-author it, so PhysX observes the removal before the new joint.
        if self._rebuild_requested:
            self._rebuild_requested = False
            if self._joint_active:
                self._remove_fixed_joint()
                self._joint_active = False
                return

        if self._locked and not self._joint_active and not self._fallback_teleport:
            if self._create_fixed_joint():
                self._joint_active = True
            else:
                self._fallback_teleport = True
        elif not self._locked and self._joint_active:
            self._remove_fixed_joint()
            self._joint_active = False

        if not self._locked:
            return

        import warp as wp

        # Fallback only: if the FixedJoint could not be created, pin the base
        # kinematically every step so it does not collapse.
        if self._fallback_teleport:
            self._articulation.set_world_poses(
                positions=wp.array([self._position], dtype=wp.float32),
                orientations=wp.array([self._orientation], dtype=wp.float32),
            )
            self._zero_root_velocity()

        # Hold the waist joints rigid (waist DOFs only).
        if self._waist_dof_indices:
            n = len(self._waist_dof_indices)
            dof_idx = wp.array(self._waist_dof_indices, dtype=wp.int32)
            self._articulation.set_dof_positions(
                wp.zeros((1, n), dtype=wp.float32), dof_indices=dof_idx
            )
            self._articulation.set_dof_velocities(
                wp.zeros((1, n), dtype=wp.float32), dof_indices=dof_idx
            )
