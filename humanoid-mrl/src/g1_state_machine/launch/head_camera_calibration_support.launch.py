"""Robot, camera-TF and mocap support for head-camera calibration.

Run ``scripts/head_camera_calibration.sh`` in a second terminal for the
calibration node. Camera drivers come from the deployment Docker profile; no
wall-plane estimator is started.
"""

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
        description="QTM rigid body attached to the robot. Its origin and axes "
        "must coincide with the identically named URDF frame.",
    )

    # QTM definition of the AprilTag rigid-body frame:
    #
    # * The origin is at the centre of the printed AprilTag.
    # * The AprilTag plane is offset 9 mm from the marker-ball centres.
    # * The origin is horizontally centred relative to the marker balls.
    # * The origin is 85 mm below the top marker balls.
    # * +x points into the AprilTag/plate, +z points up, and the right-handed
    #   convention therefore makes +y point left when viewing the printed face.
    # (Defined in the mocap software, not by this launch file.)
    child_body_arg = DeclareLaunchArgument(
        "child_body",
        default_value="head_calibration_tag",
        description="QTM rigid body defined at the centre of the printed tag: "
        "+x into the plate, +z up, and +y left when viewing its front.",
    )
    debug_viz_arg = DeclareLaunchArgument(
        "enable_debug_viz",
        default_value="true",
        description="Forwarded to the real robot control stack.",
    )
    # Robot TF publishes pelvis -> ... -> torso_link -> d435_link and the live
    # arm chain through drill_tip_link. The camera drivers publish the remaining
    # optical-frame edges.
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
            "auto_collect_loop": "false",
            "use_sim_time": "false",
        }.items(),
    )

    # Publishes QTM rigid-body poses on /rigid_bodies and performs its lifecycle
    # configure/activate transitions automatically.
    qualisys_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("qualisys_driver"), "launch", "qualisys.launch.py"]
            )
        )
    )

    # Publishes drill_chuck_link -> head_calibration_tag into the robot TF tree.
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

    return LaunchDescription(
        [
            parent_body_arg,
            child_body_arg,
            debug_viz_arg,
            state_publisher_launch,
            state_machine_launch,
            control_launch,
            qualisys_launch,
            mocap_tf_bridge,
        ]
    )
