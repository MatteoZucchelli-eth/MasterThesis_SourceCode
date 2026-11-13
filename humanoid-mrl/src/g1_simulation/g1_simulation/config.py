from dataclasses import dataclass

from pathlib import Path

PHYSICS_DT_S: float = 1 / 300
"""Physics step used by the Isaac world. The render period must be a multiple.

300 Hz divides the head D435's 6/15/30 Hz evenly, and is the discretisation
interval of the joint PD (run inside PhysX with the gear_wbc.yaml gains).
"""


def render_period_s(
    camera_publish_hz: float, physics_dt: float = PHYSICS_DT_S
) -> float:
    """Render period [s] for a camera rate, rejecting rates that split a step.

    Isaac advances sim time (image stamps) by ``floor(render_dt / physics_dt)``
    whole steps, so a non-integer ratio would permanently drift dataset time.
    """
    if camera_publish_hz <= 0:
        raise ValueError(f"camera_publish_hz must be > 0, got {camera_publish_hz}")
    render_dt = 1.0 / camera_publish_hz
    steps = render_dt / physics_dt
    if abs(steps - round(steps)) > 1e-6:
        whole = max(int(steps), 1)
        valid = ", ".join(f"{1.0 / (physics_dt * n):.4g}" for n in range(2, 11))
        raise ValueError(
            f"camera_publish_hz={camera_publish_hz} gives a render period of "
            f"{render_dt * 1000:.3f} ms = {steps:.3f} physics steps of "
            f"{physics_dt * 1000:.1f} ms, which is not a whole number. Simulation "
            f"time (image stamps) would advance {whole * physics_dt * 1000:.1f} ms "
            f"per render rather than the requested {render_dt * 1000:.3f} ms, "
            f"so dataset time would drift by "
            f"{1 - whole * physics_dt / render_dt:.1%} forever. Pick a "
            f"rate that divides {1 / physics_dt:.0f} Hz physics evenly: {valid}."
        )
    return render_dt


@dataclass
class IsaacSimCfg:
    """Run simulation using IsaacSim"""

    diagnostic_seed: int | None = None
    """Optional repeatable scene randomisation for controller comparisons."""

    asset_source_dir: Path = Path("/workspace/src/g1_simulation/assets/isaacsim/g1")
    """Root dir of robot assets"""

    robot_asset: str = "g1_29dof_rev_1_0_w_hilti_drill.usd"
    """Robot asset name"""

    urdf_path: Path = Path(
        "/workspace/src/g1_state_publisher/urdf/g1_29_dof_with_hand_drill.urdf.xml"
    )
    """URDF the camera prims in ``robot_asset`` are kept in sync with.

    Same file robot_state_publisher serves; read only by
    :func:`g1_simulation.isaacsim.camera_extrinsics.sync`."""

    sync_camera_extrinsics: bool = True
    """Rewrite the asset's camera prims from ``urdf_path`` at startup, if they differ.

    Edits the asset in place (logged). A TF/renderer mismatch is invisible in the
    images but biases targets. Turn off to render stale geometry;
    ``scripts/sync_camera_extrinsics_to_usd.py --check`` reports agreement."""

    gui: bool = True
    """Whether to launch simulation with gui or headless"""

    camera_publish_hz: float = 30.0
    """Target camera image publish frequency in Hz.

    Matches the robot's head D435 and RECORD_FPS. Must divide the physics rate
    evenly (see :func:`render_period_s`).
    """

    reset_position: tuple[float, float, float] = (2.80, 0.0, 0.7283520102500916)
    """World-frame XYZ the robot is teleported to on reset, near the wooden wall.

    At READY the drill tip is 0.4545 m ahead of the pelvis, so x = 2.80 leaves it
    54.5 mm off the wall (x = 3.309). A longer bit requires moving this back. With
    +/-0.06 m x noise the nearest draw can just touch the wall at READY."""

    reset_yaw_deg: float = 0.0
    """Yaw angle (degrees, around world Z) the robot faces after a pose reset."""

    reset_key: str = "R"
    """Keyboard key that triggers a pose reset (any single letter A-Z, or digits 0-9)."""

    demo_key: str = "Z"
    """Keyboard key that places the fixed demo wall layout (GUI runs only).

    See :mod:`g1_simulation.isaacsim.demo_scene`. Places geometry only: no
    teleport, base lock or ROS announcement.
    """

    target_y_range: tuple[float, float] = (-0.25, -0.02)
    """Inclusive world-frame Y range the target squares are uniformly sampled from.

    Bounded by right-arm reach with the pinned pelvis and by the head camera's
    field of view (-y maps to +u; beyond ~-0.25 squares leave the image)."""

    target_z_range: tuple[float, float] = (0.82, 0.93)
    """Inclusive world-frame Z (height) range the squares are uniformly sampled from.

    The ceiling keeps squares in the head camera view (visible wall z is
    ~[0.21, 0.95]; 0.93 leaves a >=27 px top margin). The floor is set by arm
    reach: below ~0.82 the arm-only IK residual frequently exceeds the 5 mm
    settle gate. Re-measure reach before lowering it.

    Measured visibility (all three squares in view, 40 px margin): 98.9%."""

    target_min_separation: float = 0.03
    """Minimum in-wall distance [m] between any two wall objects.

    0.03 m lets all nine colour x shape cells be placed (~11.5% acceptance per
    draw, so 200 attempts almost always succeed). Must stay above ~0.0175 m
    (twice the triangle circumradius) to avoid overlap."""

    robot_position_noise: tuple[float, float, float] = (0.06, 0.02, 0.0)
    """Half-width of the uniform XYZ noise [m] added to ``reset_position`` on reset.

    Wide X varies the pelvis-to-wall distance (0.45-0.57 m) so tip depth can't
    be inferred from proprioception alone; unreachable draws are discarded by
    the auto-collect settle gate. Z is 0 (base pinned)."""

    robot_yaw_noise_deg: float = 3.0
    """Half-width of the uniform yaw noise [deg] added to ``reset_yaw_deg`` on reset."""

    drill_bit_high_contrast: bool = False
    """Repaint the drill bit matte dark grey so it reads against the wall.

    Viewing aid only: recorded images then differ from the real drill. Keep off
    when collecting.
    """

    disable_drill_bit_collision: bool = True
    """Turn off the drill bit's collider at load time (applied at runtime, not baked).

    The tip collider ends exactly on ``drill_tip_link``, so contact
    (``DRILL_CONTACT=1`` / ``--cfg.no-disable-drill-bit-collision``) is stable
    and reachable. With contact off the tip overshoots ~1 mm through the wall
    and relaxes back. See docs/pages/simulation.rst."""

    disable_leg_collisions: bool | None = None
    """Turn off collision geometry on the leg links at load time. None = follow the mode.

    With the pelvis pinned (:class:`BaseLock`), leg/ground penetration
    destabilises the articulation (arm jitter); :attr:`walk_mode` needs them on.
    Applied to the USD, not the URDF. See :meth:`leg_collisions_disabled`.
    """

    walk_mode: bool = False
    """Stand the robot on its feet and drive it around, instead of pinning the pelvis.

    Enables leg colliders, removes the pin and spawns at
    :attr:`walk_reset_height`; WASD via the wireless_remote drives it. Press 9
    in the Isaac window to release the gantry. Not for AUTO_COLLECT or evals."""

    calibration: bool = False
    """Show the head-camera calibration tag instead of wall targets."""

    calibration_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Head-camera rotation error injected in calibration mode [deg]."""

    calibration_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Head-camera translation error injected in calibration mode [m]."""

    walk_reset_height: float = 0.80
    """Pelvis height [m] a walk-mode spawn/reset teleports to. Ignored otherwise.

    Straight legs put the soles 0.792 m below the pelvis, so 0.80 drops the
    robot onto its feet without interpenetration."""

    def leg_collisions_disabled(self) -> bool:
        """Whether to strip the leg colliders, resolving ``None`` against the mode."""
        if self.disable_leg_collisions is not None:
            return self.disable_leg_collisions
        return not self.walk_mode

    def spawn_position(self) -> tuple[float, float, float]:
        """``reset_position``, lifted to a standing pelvis height in walk mode."""
        if not self.walk_mode:
            return self.reset_position
        x, y, _ = self.reset_position
        return (x, y, self.walk_reset_height)
