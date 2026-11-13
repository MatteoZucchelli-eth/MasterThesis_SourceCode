"""Full simulated control stack plus the headless VLA evaluation sequencer."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    metrics = DeclareLaunchArgument("metrics_output_dir", default_value="/metrics")
    dataset = DeclareLaunchArgument("dataset_path", default_value="")
    ood_correction = DeclareLaunchArgument(
        "ood_correction_enabled", default_value="false"
    )
    rollout = DeclareLaunchArgument("rollout_s", default_value="60.0")
    # `vla` plays model chunks directly; `vla_planner` uses the model for
    # perception and plans the last leg. Budget rollout_s against
    # vla_planner_approach_timeout_s (default 90 s).
    rollout_state = DeclareLaunchArgument("rollout_state", default_value="vla")
    # Wall-plane estimator; only consumed by vla_planner with depth_source: stereo.
    # Off by default since the eval sim publishes no head depth.
    wall_plane = DeclareLaunchArgument("enable_wall_plane", default_value="false")
    # Seconds held in READY before each rollout (see ReadyController's ramp).
    ready_settle = DeclareLaunchArgument("ready_settle_s", default_value="20.0")
    # End the rollout early when VLA_PLANNER reports completion (no-op for vla).
    stop_on_complete = DeclareLaunchArgument("stop_on_complete", default_value="true")
    # Debug viz node (debug topics -> RViz TF frames/overlays). Off by default for
    # headless evaluation; evaluate_vla.sh enables it with SIM_GUI=1.
    debug_viz = DeclareLaunchArgument("enable_debug_viz", default_value="false")

    stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_state_machine"), "launch", "full_launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "wbc_eval": "true",
            "enable_rviz": "false",
            "enable_debug_viz": LaunchConfiguration("enable_debug_viz"),
            "enable_wall_plane": LaunchConfiguration("enable_wall_plane"),
            "metrics_output_dir": LaunchConfiguration("metrics_output_dir"),
            "dataset_path": LaunchConfiguration("dataset_path"),
            "ood_correction_enabled": LaunchConfiguration("ood_correction_enabled"),
        }.items(),
    )
    sequencer = Node(
        package="g1_control",
        executable="vla_eval_bringup_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "use_sim_time": True,
                "metrics_output_dir": LaunchConfiguration("metrics_output_dir"),
                "rollout_state": LaunchConfiguration("rollout_state"),
                "rollout_s": ParameterValue(
                    LaunchConfiguration("rollout_s"), value_type=float
                ),
                "ready_settle_s": ParameterValue(
                    LaunchConfiguration("ready_settle_s"), value_type=float
                ),
                "stop_on_complete": ParameterValue(
                    LaunchConfiguration("stop_on_complete"), value_type=bool
                ),
            }
        ],
    )
    return LaunchDescription(
        [
            metrics,
            dataset,
            ood_correction,
            rollout,
            rollout_state,
            wall_plane,
            ready_settle,
            stop_on_complete,
            debug_viz,
            stack,
            sequencer,
        ]
    )
