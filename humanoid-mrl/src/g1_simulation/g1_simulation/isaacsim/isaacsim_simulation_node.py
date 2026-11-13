# MIT License
"""Isaac Sim entry point: boot the app, build the scene, wire the components.

Only sequences the boot-order-sensitive startup; logic lives in the sibling modules.
"""
import logging

import tyro

from g1_simulation import config
from g1_simulation.config import IsaacSimCfg


def _configure_carb(gui: bool) -> None:
    """Carb/PhysX settings; must run right after SimulationApp is constructed."""
    import carb
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.enable_fabric(True)

    carb_settings = carb.settings.get_settings()
    # Hydra scene-graph instancing (renders instanceable assets in the GUI).
    carb_settings.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)
    # Use the PhysX SDK dispatcher instead of carb tasking.
    carb_settings.set_bool("/physics/physxDispatcher", True)
    # The simulation exposes no contact-reporting channel.
    carb_settings.set_bool("/physics/disableContactProcessing", True)
    carb_settings.set_bool("/physics/autoPopupSimulationOutputWindow", False)

    carb_settings.set_bool("/physics/fabricUpdateEnabled", True)
    carb_settings.set_bool("/app/runLoops/main/disableRefScaling", True)

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")
    if gui:
        enable_extension("omni.physx.supportui")


def _sync_camera_extrinsics(asset_path, urdf_path) -> None:
    """Write the URDF camera poses into the asset before it is referenced. Never
    fatal."""
    from g1_simulation.isaacsim import camera_extrinsics

    if not urdf_path.exists():
        logging.warning(
            "Camera extrinsics not checked: no URDF at %s. That path is where "
            "docker-compose bind-mounts it; running the sim outside the container "
            "needs --cfg.urdf-path (or --cfg.no-sync-camera-extrinsics).",
            urdf_path,
        )
        return

    try:
        report = camera_extrinsics.sync(asset_path, urdf_path)
    except (RuntimeError, OSError) as exc:
        logging.error(
            "Camera extrinsics NOT synced (%s). The rendered cameras may not be "
            "where TF says they are, which is invisible in the images and shows "
            "up only as the robot missing the target. Run "
            "g1_simulation/scripts/sync_camera_extrinsics_to_usd.py to see why.",
            exc,
        )
        return

    if report.written:
        logging.warning(
            "Camera extrinsics were OUT OF SYNC and have been rewritten from the "
            "URDF — %s. %s is edited on disk; commit it. Data recorded before "
            "this run used the previous prim poses.",
            "; ".join(str(delta) for delta in report.written),
            asset_path,
        )
    if report.blocked:
        logging.error("Camera extrinsics NOT synced: %s", report.blocked)
    for delta in report.drifted:
        logging.warning(
            "URDF and USD disagree about %s, which this does not write: %s",
            delta.urdf_frame,
            delta,
        )
    if report.cameras_in_sync and not report.written:
        logging.info("Camera extrinsics match the URDF.")
    logging.debug("camera extrinsics report:\n%s", "\n".join(report.lines))


def main(cfg: IsaacSimCfg = IsaacSimCfg()):
    """Run Isaac Sim simulation with custom ros bridge."""
    logging.basicConfig(level=logging.INFO)
    if cfg.diagnostic_seed is not None:
        import numpy as np
        np.random.seed(cfg.diagnostic_seed)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not cfg.gui})
    _configure_carb(cfg.gui)
    simulation_app.update()

    import rclpy

    rclpy.init()

    from isaacsim.core.api import World
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.experimental.utils.stage import (
        add_reference_to_stage,
        get_stage_units,
    )
    from isaacsim.core.utils.viewports import set_camera_view

    from g1_simulation.isaacsim import scene
    from g1_simulation.isaacsim.base_lock import BaseLock
    from g1_simulation.isaacsim.demo_scene import DemoScene
    from g1_simulation.isaacsim.elastic_band import ElasticBand
    from g1_simulation.isaacsim.eval_sequencer import EvaluationSequencer
    from g1_simulation.isaacsim.pose_resetter import RobotPoseResetter
    from g1_simulation.isaacsim.ros_bridge import RosBridge

    # Raises unless the camera rate is a whole number of physics steps.
    physics_dt = config.PHYSICS_DT_S
    render_dt = config.render_period_s(cfg.camera_publish_hz, physics_dt)

    world = World(
        physics_dt=physics_dt,
        rendering_dt=render_dt,
        stage_units_in_meters=1.0,
    )
    # Walk mode: leg colliders, solid floor, no pelvis pin, standing spawn height.
    walk_mode = cfg.walk_mode
    scene.setup_scene(
        world,
        add_reference_to_stage,
        floor_collision=walk_mode,
        calibration=cfg.calibration,
    )
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    # Viewport camera: robot's front-right, looking at the target wall.
    set_camera_view(
        eye=[0.2, -2.2, 1.6],
        target=[0.75, -0.3, 1.0],
        camera_prim_path="/OmniverseKit_Persp",
    )

    asset_path = cfg.asset_source_dir / cfg.robot_asset
    if not asset_path.exists():
        raise ValueError(
            f"Robot asset: {cfg.robot_asset} does not exist in {cfg.asset_source_dir}"
        )

    if cfg.sync_camera_extrinsics:
        _sync_camera_extrinsics(asset_path, cfg.urdf_path)

    add_reference_to_stage(usd_path=asset_path.as_posix(), path="/World/robot")
    if cfg.leg_collisions_disabled():
        scene.disable_leg_collisions("/World/robot")
    elif walk_mode:
        logging.info(
            "Walk mode: leg collisions left ON — the robot stands on its feet."
        )
    if cfg.disable_drill_bit_collision:
        scene.disable_drill_bit_collision("/World/robot")
    scene.set_drill_bit_physics_material("/World/robot")
    if cfg.drill_bit_high_contrast:
        scene.set_drill_bit_high_contrast("/World/robot")
    robot = Articulation(paths="/World/robot")

    import warp as wp

    # Spawn at the lock X/Y; height 1.0 (band settles it) except in walk mode,
    # where it spawns at standing height.
    reset_position = cfg.spawn_position()
    spawn_x, spawn_y, spawn_z = reset_position
    if not walk_mode:
        spawn_z = 1.0
    robot.set_world_poses(
        positions=wp.array([[spawn_x, spawn_y, spawn_z]], dtype=wp.float32)
        / get_stage_units()[0]
    )

    ros_bridge = RosBridge(
        articulation=robot,
        calibration=cfg.calibration,
        calibration_rpy_deg=cfg.calibration_rpy_deg,
        calibration_xyz_m=cfg.calibration_xyz_m,
    )

    pose_resetter = RobotPoseResetter(
        articulation=robot,
        position=reset_position,
        yaw_deg=cfg.reset_yaw_deg,
        key=cfg.reset_key,
        target_y_range=cfg.target_y_range,
        target_z_range=cfg.target_z_range,
        target_min_separation=cfg.target_min_separation,
        position_noise=cfg.robot_position_noise,
        yaw_noise_deg=cfg.robot_yaw_noise_deg,
    )
    ros_bridge.set_pose_resetter(pose_resetter)

    elastic_band = ElasticBand(
        articulation=robot,
        link_name="torso_link",
        enable=True,
        # Anchor above the lock position; in walk mode press 9 to release it.
        anchor_xy=(reset_position[0], reset_position[1]),
    )

    eval_sequencer = EvaluationSequencer(pose_resetter=pose_resetter)
    ros_bridge.set_eval_sequencer(eval_sequencer)

    base_lock = BaseLock(
        articulation=robot,
        pose_resetter=pose_resetter,
        lock_keys=("V", cfg.reset_key),
        release_key="B",
        # Walk mode needs a free base.
        enabled=not walk_mode,
    )

    # ROS-triggered reset (sim/reset_request), equivalent to the reset key.
    def _handle_reset_request():
        pose_resetter.reset()
        base_lock.engage()

    ros_bridge.set_reset_handler(_handle_reset_request)

    # ROS-triggered scenario advance (sim/eval_advance_request), equivalent to V:
    # also re-locks the base at the new scenario pose.
    def _handle_eval_advance_request():
        eval_sequencer.advance()
        base_lock.engage()

    ros_bridge.set_eval_advance_handler(_handle_eval_advance_request)

    if cfg.gui:
        # Shared keyboard dispatcher (GUI only; headless uses ROS triggers).
        import carb.input
        import omni.appwindow

        # Demo layout is GUI only.
        demo_scene = DemoScene(key=cfg.demo_key, clear_keys=("V", cfg.reset_key))

        keyboard_handlers = [
            pose_resetter.on_keyboard_input,
            elastic_band.on_keyboard_input,
            eval_sequencer.on_keyboard_input,
            base_lock.on_keyboard_input,
            demo_scene.on_keyboard_input,
        ]

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()

        def _shared_keyboard_callback(e):
            for handler in keyboard_handlers:
                handler(e)

        input_iface.subscribe_to_keyboard_events(keyboard, _shared_keyboard_callback)

        if walk_mode:
            # Scene keys are carb events; robot keys go through wireless_remote
            # in LowState (g1_simulation.input_device, pynput).
            logging.info(
                "Walk mode keys — Isaac window (carb): '%s' reset pose + targets, "
                "9 toggle the gantry, 7/8 lower/raise it, V next eval scenario, "
                "'%s' place the demo wall layout. "
                "Robot (wireless_remote, needs the control stack up): q+UP = READY, "
                "o+1 = MOVE, then W/S forward/back, A/D strafe, J/L turn, I/K base "
                "height; q+2 = DAMPING. Release the gantry with 9 once MOVE holds "
                "the robot up, or it will tug the robot back toward the spawn.",
                cfg.reset_key,
                cfg.demo_key,
            )

    world.reset()

    if not cfg.gui:
        # Headless: perform the initial reset now. Must follow world.reset(),
        # which would clobber an earlier teleport.
        logging.info(
            "Headless run: %s at startup (auto 'R' press).",
            "teleporting to the reset pose" if walk_mode else "locking the base",
        )
        _handle_reset_request()

    world.add_physics_callback("compute elastic band", lambda x: elastic_band.step())
    world.add_physics_callback("base lock", lambda x: base_lock.step())
    world.add_physics_callback(
        "compute torques", lambda x: ros_bridge.compute_torques()
    )
    world.add_physics_callback(
        "publish lowstate", lambda _: ros_bridge.step(world.current_time)
    )

    def step_simulation():
        world.step(render=True, update_fabric=True)

    _ = ros_bridge.create_timer(render_dt, step_simulation)

    try:
        rclpy.spin(ros_bridge)
    finally:
        simulation_app.close()


def entrypoint():
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
