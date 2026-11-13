"""AUTO_COLLECT on the real robot, driven by hand: full stack minus the sim.

Hardware counterpart of full_launch.py: the sim is replaced by Qualisys mocap
plus a TF bridge providing ``drill_chuck_link -> red_square_target``, with
``auto_collect_real.yaml`` parameters.

No sequencer or episode loop; publish transitions and phases by hand::

    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: damping"
    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: ready"       # ~10 s
    ros2 topic pub -1 /fsm_trigger std_msgs/String "data: auto_collect"

    ros2 topic pub -1 /auto_collect/phase std_msgs/String "data: going_to_default"
    ros2 topic pub -1 /auto_collect/phase std_msgs/String "data: trajectory_generation"
    ros2 topic pub -1 /auto_collect/phase std_msgs/String "data: approaching"

Before ``approaching``, check the planner's log line after
``trajectory_generation`` (target in pelvis frame, drill-axis misalignment) to
confirm the mocap chain resolved.
"""

from g1_definitions import topics
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parent_body_arg = DeclareLaunchArgument(
        "parent_body",
        default_value="drill_chuck_link",
        description="QTM 6DOF body tracking the drill. Must be named after a "
        "frame the URDF already publishes -- the bridge hangs the target off it, "
        "which is what connects the mocap measurement to the robot's TF tree "
        "without anyone having to track the pelvis. The body is defined at the "
        "drill's chuck face, so it must be the chuck frame and not "
        "right_drill_link: those are 128.7 mm apart, and naming the wrong one "
        "drags the target that far towards the robot with no error anywhere.",
    )
    child_body_arg = DeclareLaunchArgument(
        "child_body",
        default_value="red_square_target",
        description="QTM 6DOF body on the wall target. Must match "
        "ControlContext.drill_target_frame (red_square_target by default), and "
        "its own x-axis must point into the wall -- that is the normal the "
        "standoff point and the drill alignment are both built from.",
    )
    debug_viz_arg = DeclareLaunchArgument(
        "enable_debug_viz",
        default_value="false",
        description="RViz debug visualization. Off on the robot by default.",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value=topics.TOPIC_CAMERA_DEPTH_ALIGNED,
        description="Aligned D435 depth image used to measure the wall. The "
        "unregistered depth image cannot be paired with colour intrinsics.",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value=topics.TOPIC_CAMERA_DEPTH_ALIGNED_INFO,
        description="Intrinsics matching depth_topic; for aligned depth these "
        "are the D435 colour intrinsics.",
    )

    # Robot TF: pelvis -> ... -> drill_chuck_link, which the bridge attaches to.
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
                [FindPackageShare("g1_control"), "launch", "control_launch.py"]
            )
        ),
        launch_arguments={
            "auto_collect_config": "auto_collect_real.yaml",
            "enable_debug_viz": LaunchConfiguration("enable_debug_viz"),
            # Defaults, pinned explicitly: true would be wrong on hardware.
            "auto_collect_loop": "false",
            "use_sim_time": "false",
        }.items(),
    )

    # QTM host from qualisys_driver's params; self-configures/activates.
    qualisys_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("qualisys_driver"), "launch", "qualisys.launch.py"]
            )
        )
    )

    mocap_tf_bridge = Node(
        package="g1_sensors",
        executable="mocap_tf_bridge",
        name="mocap_tf_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "parent_body": LaunchConfiguration("parent_body"),
                "child_body": LaunchConfiguration("child_body"),
            }
        ],
    )

    # Independent head-camera check of the wall (/wall/plane); does not feed
    # AUTO_COLLECT's trajectory, which uses the mocap target frame.
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
            "depth_bias_m": "0.03",
        }.items(),
    )

    return LaunchDescription(
        [
            parent_body_arg,
            child_body_arg,
            debug_viz_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            state_publisher_launch,
            state_machine_launch,
            control_launch,
            qualisys_launch,
            mocap_tf_bridge,
            wall_plane_launch,
        ]
    )
