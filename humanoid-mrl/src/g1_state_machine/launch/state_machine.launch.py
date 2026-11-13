from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    rc_triggers = PathJoinSubstitution(
        [FindPackageShare("g1_state_machine"), "config", "transition_rc_mapping.yaml"]
    )

    fsm_node = Node(
        package="g1_state_machine",
        executable="state_machine",
        name="finite_state_machine",
        output="screen",
        emulate_tty=True,
        parameters=[rc_triggers],
    )

    return LaunchDescription([fsm_node])
