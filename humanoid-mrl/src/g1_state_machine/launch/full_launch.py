from g1_definitions import definitions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Simulation entry point. Override control_launch.py's compressed-topic
    # defaults with the raw sim images: these nodes run on the sim host, the
    # '/compressed' mirror is for remote RViz.
    head_image_topic_arg = DeclareLaunchArgument(
        "head_image_topic",
        default_value=definitions.TOPIC_CAMERA,
        description=(
            "Head image fed to the VLA models, forwarded to control_launch.py. "
            "Defaults to the raw topic: these nodes run on the sim host, so "
            "they read the bridge's raw stream directly rather than the "
            "compressed mirror meant for RViz."
        ),
    )
    wrist_image_topic_arg = DeclareLaunchArgument(
        "wrist_image_topic",
        default_value=definitions.TOPIC_CAMERA_WRIST,
        description=(
            "Wrist image fed to the VLA models and drawn on by the perception "
            "overlay, forwarded to control_launch.py. Raw for the same reason "
            "as head_image_topic."
        ),
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Take time from the sim's /clock instead of the system "
        "clock. True because this launch is the simulation entry point — the "
        "hardware counterpart is real_launch.py. Set false only to run these "
        "nodes against something that publishes no /clock.",
    )
    wbc_eval_arg = DeclareLaunchArgument("wbc_eval", default_value="false")
    metrics_output_dir_arg = DeclareLaunchArgument(
        "metrics_output_dir", default_value="vla_metrics"
    )
    dataset_path_arg = DeclareLaunchArgument("dataset_path", default_value="")
    ood_correction_arg = DeclareLaunchArgument(
        "ood_correction_enabled", default_value="false"
    )
    enable_rviz_arg = DeclareLaunchArgument("enable_rviz", default_value="true")
    enable_wall_plane_arg = DeclareLaunchArgument(
        "enable_wall_plane", default_value="true"
    )
    enable_debug_viz_arg = DeclareLaunchArgument(
        "enable_debug_viz", default_value="true"
    )

    simulation_utils_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_simulation"), "launch", "simulation_utils.py"]
            )
        )
    )

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

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_control"),
                    "launch",
                    "control_launch.py",
                ]
            )
        ),
        launch_arguments={
            "head_image_topic": LaunchConfiguration("head_image_topic"),
            "wrist_image_topic": LaunchConfiguration("wrist_image_topic"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "wbc_eval": LaunchConfiguration("wbc_eval"),
            "metrics_output_dir": LaunchConfiguration("metrics_output_dir"),
            "dataset_path": LaunchConfiguration("dataset_path"),
            "ood_correction_enabled": LaunchConfiguration("ood_correction_enabled"),
            "enable_debug_viz": LaunchConfiguration("enable_debug_viz"),
        }.items(),
    )

    # Wall plane fitted from the head d435 depth stream. Disabled by the
    # scripted evaluation, which runs without depth and never enters VLA_PLANNER.
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
        condition=IfCondition(LaunchConfiguration("enable_wall_plane")),
    )

    # config.rviz lives at the workspace root (not installed), so use its source path.
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", "/workspace/config.rviz"],
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
    )

    return LaunchDescription(
        [
            head_image_topic_arg,
            wrist_image_topic_arg,
            use_sim_time_arg,
            wbc_eval_arg,
            metrics_output_dir_arg,
            dataset_path_arg,
            ood_correction_arg,
            enable_rviz_arg,
            enable_wall_plane_arg,
            enable_debug_viz_arg,
            # Applies to every node below (including included launches), so the
            # whole TF tree shares one clock. The sim itself is a separate
            # process on the system clock, since it produces /clock.
            SetParameter(
                name="use_sim_time", value=LaunchConfiguration("use_sim_time")
            ),
            simulation_utils_launch,
            state_publisher_launch,
            state_machine_launch,
            control_launch,
            wall_plane_launch,
            rviz_node,
        ]
    )
