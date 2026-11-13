"""Headless AUTO_COLLECT data-collection worker: everything except the sim.

Brings up the controller (episode loop on), the FSM, the dataset recorder and
the bringup node that enters AUTO_COLLECT once the sim publishes. Pair with a
headless Isaac Sim on the same ROS_DOMAIN_ID.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    dataset_path_arg = DeclareLaunchArgument(
        "dataset_path",
        default_value="/workspace/g1_dataset",
        description="Root folder the LeRobot dataset is written to. Must be "
        "unique per worker — the recorder is a single-writer design.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Take time from the sim's /clock instead of the system "
        "clock. True is correct here — this launch exists to pair with a "
        "headless Isaac Sim, and it is what makes an episode's durations mean "
        "the same thing at any real-time factor. Set false only when running "
        "this stack against real hardware, where nothing publishes /clock and "
        "ROS time would otherwise stay pinned at 0 forever.",
    )
    task_description_arg = DeclareLaunchArgument(
        "task_description",
        default_value="drill in the red square",
        description="Task string recorded with every episode.",
    )
    arm_integral_ki_override_arg = DeclareLaunchArgument(
        "arm_integral_ki_override",
        default_value="-1.0",
        description="Nonnegative diagnostic override for the arm integral gain.",
    )

    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_control"), "launch", "control_launch.py"]
            )
        ),
        launch_arguments={
            "enable_debug_viz": "false",
            "auto_collect_loop": "true",
            "wbc_eval": "true",
            "arm_integral_ki_override": LaunchConfiguration(
                "arm_integral_ki_override"
            ),
        }.items(),
    )

    state_machine = IncludeLaunchDescription(
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

    recorder = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_teleoperation"), "launch", "recorder.launch.py"]
            )
        ),
        launch_arguments={
            "dataset_path": LaunchConfiguration("dataset_path"),
            "task_description": LaunchConfiguration("task_description"),
        }.items(),
    )

    bringup = Node(
        package="g1_control",
        executable="auto_collect_bringup_node",
        name="auto_collect_bringup_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            dataset_path_arg,
            task_description_arg,
            arm_integral_ki_override_arg,
            use_sim_time_arg,
            # Applies to every node below; the sim (a separate process) produces /clock.
            SetParameter(
                name="use_sim_time", value=LaunchConfiguration("use_sim_time")
            ),
            control,
            state_machine,
            recorder,
            bringup,
        ]
    )
