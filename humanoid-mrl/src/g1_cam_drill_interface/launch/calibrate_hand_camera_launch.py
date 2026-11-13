# Measure the wrist camera's pose relative to the drill body from the AprilTag
# on the drill, and print the numbers to paste into the URDF and the Isaac USD.
#
# Against a camera that is already running (robot or sim):
#     ros2 launch g1_cam_drill_interface calibrate_hand_camera_launch.py
# Against the sim, whose bridge publishes raw images and not compressed ones:
#     ros2 launch g1_cam_drill_interface calibrate_hand_camera_launch.py \
#         image_topic:=/camera/hand/color/image_raw
# Or bring the driver up with it in one go:
#     ros2 launch g1_cam_drill_interface hand_camera_launch.py calibrate:=true

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Defaults match the driver's /camera/hand/color namespace (TOPIC_CAMERA_WRIST).
ARGUMENTS = [
    DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/hand/color/image_raw/compressed",
        description="Wrist camera image. A '/compressed' suffix selects "
        "CompressedImage, anything else raw Image.",
    ),
    DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/hand/color/camera_info",
        description="Intrinsics + distortion for the same camera.",
    ),
    DeclareLaunchArgument(
        "tag_family", default_value="tag36h11", description="AprilTag family."
    ),
    DeclareLaunchArgument(
        "tag_size_m",
        default_value="0.011",
        description="Side of the tag's black border, in metres. A wrong value "
        "scales the whole result linearly, so measure it.",
    ),
    DeclareLaunchArgument(
        "tag_id", default_value="1", description="Id of the tag on the drill body."
    ),
    DeclareLaunchArgument(
        "num_samples",
        default_value="50",
        description="Detections to average before reporting.",
    ),
    DeclareLaunchArgument(
        "output_path",
        default_value="~/.ros/hand_camera_extrinsics.yaml",
        description="Where to write the report. Empty string to only log it.",
    ),
]


def generate_launch_description() -> LaunchDescription:
    calibration_node = Node(
        package="g1_cam_drill_interface",
        executable="calibrate_hand_camera_node",
        name="calibrate_hand_camera_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "tag_family": LaunchConfiguration("tag_family"),
                "tag_size_m": LaunchConfiguration("tag_size_m"),
                "tag_id": LaunchConfiguration("tag_id"),
                "num_samples": LaunchConfiguration("num_samples"),
                "output_path": LaunchConfiguration("output_path"),
            }
        ],
    )

    return LaunchDescription([*ARGUMENTS, calibration_node])
