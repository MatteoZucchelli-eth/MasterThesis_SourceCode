from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dataset_path",
                default_value="/workspace/g1_dataset",
                description="Root folder where the LeRobot v2 dataset will be written.",
            ),
            DeclareLaunchArgument(
                "task_description",
                default_value="perform a task",
                description="Default task string used when no /task message has been received.",
            ),
            Node(
                package="g1_teleoperation",
                executable="recorder_node",
                name="recorder_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "dataset_path": LaunchConfiguration("dataset_path"),
                        "task_description": LaunchConfiguration("task_description"),
                    }
                ],
            ),
        ]
    )
