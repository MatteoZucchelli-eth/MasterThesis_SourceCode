import os

import yaml

from ament_index_python.packages import get_package_share_directory
from g1_definitions import definitions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("g1_control")
    g1_state_publisher_share = FindPackageShare("g1_state_publisher")

    rc_gains_config = PathJoinSubstitution([pkg_share, "config", "rc_gains.yaml"])
    gear_wbc_config = PathJoinSubstitution([pkg_share, "config", "gear_wbc.yaml"])

    balance_model_arg = DeclareLaunchArgument(
        "balance_model",
        default_value="GR00T-WholeBodyControl-Balance.onnx",
        description="Filename of the balance (standing) ONNX model under models/",
    )
    walk_model_arg = DeclareLaunchArgument(
        "walk_model",
        default_value="GR00T-WholeBodyControl-Walk.onnx",
        description="Filename of the walking ONNX model under models/",
    )
    debug_viz_arg = DeclareLaunchArgument(
        "enable_debug_viz",
        default_value="true",
        description="Launch the RViz debug visualization node (disable on robot)",
    )
    # Camera streams for the controller and the debug overlay. Compressed by
    # default: raw usb_cam frames stall on the robot (RELIABLE, 2.76 MB). Pass the
    # bare topics in sim, where publisher and subscriber share a host.
    head_image_topic_arg = DeclareLaunchArgument(
        "head_image_topic",
        default_value=definitions.TOPIC_CAMERA + "/compressed",
        description=(
            "Head (torso d435) image fed to the VLA models as ego_view. A "
            "'/compressed' suffix selects CompressedImage, anything else raw "
            "Image. Pass the bare "
            + definitions.TOPIC_CAMERA
            + " when publisher and subscriber share a host, as in simulation."
        ),
    )
    wrist_image_topic_arg = DeclareLaunchArgument(
        "wrist_image_topic",
        default_value=definitions.TOPIC_CAMERA_WRIST + "/compressed",
        description=(
            "Wrist (drill hand_cam) image fed to the VLA models as wrist_view "
            "and used as the debug overlay's canvas. A '/compressed' suffix "
            "selects CompressedImage, anything else raw Image. Pass the bare "
            + definitions.TOPIC_CAMERA_WRIST
            + " when publisher and subscriber share a host, as in simulation."
        ),
    )
    auto_collect_loop_arg = DeclareLaunchArgument(
        "auto_collect_loop",
        default_value="false",
        description=(
            "Chain AUTO_COLLECT episodes automatically (headless data collection); "
            "overrides vla_config.yaml's auto_collect_loop"
        ),
    )

    auto_collect_config_arg = DeclareLaunchArgument(
        "auto_collect_config",
        default_value="auto_collect.yaml",
        description=(
            "Filename under config/ holding the AUTO_COLLECT parameters. "
            "auto_collect.yaml is the simulation set; pass "
            "auto_collect_real.yaml on the robot, where there is no /clock, no "
            "sim reset to chain episodes with and one physical target instead "
            "of nine resampled squares."
        ),
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description=(
            "Take time from the sim's /clock instead of the system clock. "
            "False is correct on the real robot, where nothing publishes "
            "/clock and ROS time would stay pinned at 0 forever. Pass true "
            "when running this stack against Isaac — it is what makes an "
            "eval rollout's durations mean the same thing at any real-time "
            "factor. Note that VLA inference rate is wall-bound either way, "
            "so an eval below rtf 1.0 still gives the policy more inference "
            "per unit of robot motion than hardware will."
        ),
    )

    wbc_eval_arg = DeclareLaunchArgument(
        "wbc_eval",
        default_value="false",
        description=(
            "Pin the legs and waist to q0 instead of running the RL locomotion "
            "policy. Off by default, so the policy drives the lower body."
        ),
    )

    arm_integral_ki_override_arg = DeclareLaunchArgument(
        "arm_integral_ki_override",
        default_value="-1.0",
        description="Nonnegative diagnostic override for gear_wbc arm_integral_ki.",
    )
    metrics_output_dir_arg = DeclareLaunchArgument(
        "metrics_output_dir",
        default_value="vla_metrics",
        description="Folder for per-session VLA metrics and evaluation reports.",
    )
    dataset_path_arg = DeclareLaunchArgument(
        "dataset_path",
        default_value="",
        description=(
            "LeRobot dataset root used by the state-only OOD scorer. An empty "
            "value disables OOD scoring and correction."
        ),
    )
    ood_correction_arg = DeclareLaunchArgument(
        "ood_correction_enabled",
        default_value="false",
        description=(
            "Apply OOD gradient correction to VLA action chunks. State scoring "
            "and logging remain active when this is false."
        ),
    )

    pkg_share_dir = get_package_share_directory("g1_control")
    # One YAML per concern: VLA client core, then each state's own tuning.
    state_configs = [
        os.path.join(pkg_share_dir, "config", name)
        for name in (
            "vla_config.yaml",
            "vla_planner.yaml",
            "fk_calibration.yaml",
        )
    ]
    # Sim and robot AUTO_COLLECT configs are alternatives, chosen by launch arg.
    auto_collect_config = PathJoinSubstitution(
        [pkg_share, "config", LaunchConfiguration("auto_collect_config")]
    )

    node_params = {
        "balance_model_path": PathJoinSubstitution(
            [pkg_share, "models", LaunchConfiguration("balance_model")]
        ),
        "walk_model_path": PathJoinSubstitution(
            [pkg_share, "models", LaunchConfiguration("walk_model")]
        ),
        "gear_wbc_config_path": gear_wbc_config,
        # Drill variant (payload link for gravity FF); use
        # g1_29_dof_with_hand.urdf.xml without the drill.
        "urdf_path": PathJoinSubstitution(
            [g1_state_publisher_share, "urdf", "g1_29_dof_with_hand_drill.urdf.xml"]
        ),
        "asset_path": PathJoinSubstitution([g1_state_publisher_share, "urdf"]),
    }
    experiment_params = {}
    if os.environ.get("G1_CONTROLLER_EXPERIMENT"):
        with open(os.environ["G1_CONTROLLER_EXPERIMENT"]) as experiment_file:
            experiment_params = yaml.safe_load(experiment_file).get("collection", {})

    controller_node = Node(
        package="g1_control",
        executable="controller_manager_node",
        name="controller_manager_node",
        output="screen",
        emulate_tty=True,
        # Launch-arg overrides come after the yaml configs so they win.
        parameters=[
            rc_gains_config,
            node_params,
            *state_configs,
            auto_collect_config,
            {
                "auto_collect_loop": ParameterValue(
                    LaunchConfiguration("auto_collect_loop"), value_type=bool
                ),
                # Same topics as the debug overlay.
                "head_image_topic": LaunchConfiguration("head_image_topic"),
                "wrist_image_topic": LaunchConfiguration("wrist_image_topic"),
                "wbc_eval": ParameterValue(
                    LaunchConfiguration("wbc_eval"), value_type=bool
                ),
                "arm_integral_ki_override": ParameterValue(
                    LaunchConfiguration("arm_integral_ki_override"), value_type=float
                ),
                "progress_test_output_dir": LaunchConfiguration("metrics_output_dir"),
                "dataset_path": LaunchConfiguration("dataset_path"),
                "ood_correction_enabled": ParameterValue(
                    LaunchConfiguration("ood_correction_enabled"), value_type=bool
                ),
            },
            experiment_params,
        ],
    )

    debug_viz_node = Node(
        package="g1_control",
        executable="debug_viz_node",
        name="debug_viz_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("enable_debug_viz")),
        parameters=[
            {"wrist_image_topic": LaunchConfiguration("wrist_image_topic")},
        ],
    )

    return LaunchDescription(
        [
            balance_model_arg,
            walk_model_arg,
            debug_viz_arg,
            head_image_topic_arg,
            wrist_image_topic_arg,
            auto_collect_loop_arg,
            auto_collect_config_arg,
            wbc_eval_arg,
            arm_integral_ki_override_arg,
            metrics_output_dir_arg,
            dataset_path_arg,
            ood_correction_arg,
            use_sim_time_arg,
            SetParameter(
                name="use_sim_time", value=LaunchConfiguration("use_sim_time")
            ),
            controller_node,
            debug_viz_node,
        ]
    )
