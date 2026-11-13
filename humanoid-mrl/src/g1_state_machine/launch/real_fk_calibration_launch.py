"""FK Calibration on the real robot, using only the two RGB cameras.

This launch starts the live robot TF publishers, finite-state machine, and
whole-body controller.  It deliberately starts no depth/wall-plane estimator,
Qualisys driver, mocap bridge, camera driver, or simulation node.

Prerequisites
-------------
The deployment camera containers must already publish:

* ``/camera/hand/color/image_raw/compressed`` and its ``camera_info``;
* ``/camera/d435/color/image_raw/compressed`` and its ``camera_info``.

The hand camera is the only AprilTag detector input.  The head RGB stream is
used solely as a projection canvas for the hand-camera 3D measurement.

Run and enter the state
-----------------------
Bring the robot through its normal safe startup sequence, then press R1 + X::

    ros2 launch g1_state_machine real_fk_calibration_launch.py
    # RC: L2+B (damping), L2+DUP (ready), then R1+X (FK Calibration)
    ros2 topic pub --once /fk_calibration/start std_msgs/msg/Empty '{}'

Publish another Empty message after a round completes to collect another CSV
row. Moving the tag by itself never starts motion.

RViz is normally run on the operator workstation.  Either open the checked-in
configuration there, or pass ``launch_rviz:=true`` when this workspace has a
local display::

    rviz2 -d /workspace/config.rviz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the real-robot FK Calibration launch graph."""
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Start local RViz with /workspace/config.rviz. Keep false "
        "on a headless robot and run RViz on the operator workstation.",
    )

    # Publishes pelvis -> arm -> drill_tip_link and the camera mount links from
    # measured robot joints. Camera drivers publish the final optical edges.
    state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_state_publisher"),
                    "launch",
                    "state_publisher.launch.py",
                ]
            )
        ),
        launch_arguments={"use_sim_time": "false"}.items(),
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

    # control_launch loads fk_calibration.yaml and defaults the real camera
    # inputs to their compressed transports. Its generic VLA debug node is not
    # needed: FK Calibration publishes both overlays and its TF itself.
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_control"), "launch", "control_launch.py"]
            )
        ),
        launch_arguments={
            "auto_collect_config": "auto_collect_real.yaml",
            "auto_collect_loop": "false",
            "enable_debug_viz": "false",
            "use_sim_time": "false",
        }.items(),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="fk_calibration_rviz",
        output="screen",
        arguments=["-d", "/workspace/config.rviz"],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    return LaunchDescription(
        [
            launch_rviz_arg,
            state_publisher_launch,
            state_machine_launch,
            control_launch,
            rviz,
        ]
    )
