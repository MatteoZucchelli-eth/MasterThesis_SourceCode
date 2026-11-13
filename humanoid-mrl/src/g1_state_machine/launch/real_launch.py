"""The VLA_PLANNER drilling pipeline on the real robot: full stack minus the sim.

Hardware counterpart of full_launch.py. No mocap: the target comes from the
wrist detector.

Prerequisites (not started here, silent if missing):

  * The ``deployment`` docker profile for both cameras (``realsense-node``,
    ``g1-hand-cam-deploy``); head depth is configured in docker/head_camera.yaml.
  * Up-to-date hand-camera extrinsics
    (g1_cam_drill_interface calibrate_hand_camera_launch.py).

No bringup sequencer; publish the FSM transitions yourself::

    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: damping"
    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: ready"       # ~10 s
    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: vla_planner"

Check ``wall_plane``'s periodic report: with ``"stereo"`` and no plane the
planner accumulates nothing; a fitted rms in centimetres (not millimetres)
points at a depth/intrinsics mismatch.
"""

from g1_definitions import topics
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    debug_viz_arg = DeclareLaunchArgument(
        "enable_debug_viz",
        default_value="false",
        description="RViz debug visualization. Off on the robot by default, "
        "same as real_auto_collect_launch.py -- and unlike full_launch.py, "
        "which also starts an rviz2 of its own. Nothing here does.",
    )
    dataset_path_arg = DeclareLaunchArgument(
        "dataset_path",
        default_value="",
        description="LeRobot dataset root for state-only OOD scoring.",
    )
    ood_correction_arg = DeclareLaunchArgument(
        "ood_correction_enabled",
        default_value="false",
        description="Apply OOD gradient correction; false keeps scoring read-only.",
    )

    # On the robot, use the colour-aligned depth (align_depth.enable in
    # docker/head_camera.yaml) so depth pairs with the colour intrinsics.
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value=topics.TOPIC_CAMERA_DEPTH_ALIGNED,
        description="Rectified depth image to fit the wall from. The ALIGNED "
        "stream on the robot -- the unregistered one cannot be paired with the "
        "colour intrinsics.",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value=topics.TOPIC_CAMERA_DEPTH_ALIGNED_INFO,
        description="Intrinsics to deproject depth_topic with. Must describe "
        "the depth image, which for the aligned stream means the colour "
        "intrinsics.",
    )

    # Robot TF up to d435_link; realsense2_camera publishes the optical frames.
    state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_state_publisher"),
                    "launch",
                    "state_publisher.launch.py",
                ]
            )
        )
    )

    state_machine_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_state_machine"),
                    "launch",
                    "state_machine.launch.py",
                ]
            )
        )
    )

    # Image topics keep control_launch.py's '/compressed' defaults (required on
    # hardware).
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_control"), "launch", "control_launch.py"]
            )
        ),
        launch_arguments={
            "auto_collect_config": "auto_collect_real.yaml",
            "enable_debug_viz": LaunchConfiguration("enable_debug_viz"),
            "dataset_path": LaunchConfiguration("dataset_path"),
            "ood_correction_enabled": LaunchConfiguration("ood_correction_enabled"),
            # Defaults, stated explicitly: no /clock or sim reset on the robot.
            "auto_collect_loop": "false",
            "use_sim_time": "false",
        }.items(),
    )

    # Wall plane estimator, read by `vla_planner_depth_source: "stereo"`. Always
    # started.
    wall_plane_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_sensors"),
                    "launch",
                    "wall_plane.launch.py",
                ]
            )
        ),
        launch_arguments={
            "depth_topic": LaunchConfiguration("depth_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
            # Bench-measured D435 range error (zero in sim).
            "depth_bias_m": "0.03",
        }.items(),
    )

    return LaunchDescription(
        [
            debug_viz_arg,
            dataset_path_arg,
            ood_correction_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            state_publisher_launch,
            state_machine_launch,
            control_launch,
            wall_plane_launch,
        ]
    )
