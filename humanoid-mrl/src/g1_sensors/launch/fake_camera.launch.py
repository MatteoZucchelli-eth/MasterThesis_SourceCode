from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare("g1_sensors"), "videos", "fake_camera.mp4"]
        ),
        description="Path to the video file used as camera source",
    )

    topic_name_arg = DeclareLaunchArgument(
        "topic_name",
        default_value="/camera/image_raw",
        description="Output image topic",
    )

    camera_node = Node(
        package="g1_sensors",
        executable="fake_camera_publisher",
        name="fake_camera_publisher",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "video_path": LaunchConfiguration("video_path"),
                "topic_name": LaunchConfiguration("topic_name"),
            }
        ],
    )

    return LaunchDescription(
        [
            video_path_arg,
            topic_name_arg,
            camera_node,
        ]
    )
