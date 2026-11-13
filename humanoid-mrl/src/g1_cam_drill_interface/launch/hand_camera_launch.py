# Launch the SEE3CAM hand camera via the usb_cam driver from the corresponding ROS package.
#
# With calibrate:=true it also runs the AprilTag extrinsics measurement against
# the drill-body tag (calibrate_hand_camera_launch.py). Re-run after any change
# to the camera bracket.


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ARGUMENTS = [
    DeclareLaunchArgument(
        "calibrate",
        default_value="false",
        description="Also measure the camera's extrinsics against the AprilTag "
        "on the drill body, and report the URDF and USD numbers.",
    ),
    DeclareLaunchArgument(
        "tag_size_m",
        default_value="0.011",
        description="Side of the tag's black border, in metres (calibrate only).",
    ),
    DeclareLaunchArgument(
        "tag_id",
        default_value="1",
        description="Id of the tag on the drill body (calibrate only).",
    ),
    DeclareLaunchArgument(
        "num_samples",
        default_value="50",
        description="Detections to average before reporting (calibrate only).",
    ),
]


def generate_launch_description() -> LaunchDescription:
    params_file = PathJoinSubstitution(
        [FindPackageShare("g1_cam_drill_interface"), "config", "hand_camera.yaml"]
    )

    # usb_cam cannot resolve the by-id path (also inside containers): resolve it here.
    camera_by_id = (
        "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2321D509-video-index0"
    )
    # HAND_CAMERA_DEVICE optionally pins the camera to a specific /dev/videoN.
    video_device = os.environ.get("HAND_CAMERA_DEVICE") or (
        os.path.realpath(camera_by_id)
        if os.path.exists(camera_by_id)
        else "/dev/video0"
    )

    # Respawns on crash (e.g. a loose contact).
    hand_camera_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="hand_camera",
        # Topics on /camera/hand/color/{image_raw,camera_info} (TOPIC_CAMERA_WRIST).
        namespace="/camera/hand/color",
        parameters=[params_file, {"video_device": video_device}],
        output="screen",
        respawn=True,
        respawn_delay=2.0,
    )

    package_share = get_package_share_directory("g1_cam_drill_interface")
    install_base = os.path.join(
        package_share, "..", "..", "lib", "g1_cam_drill_interface"
    )
    script_path = os.path.join(install_base, "init_camera.sh")

    # Apply remaining camera controls via script on every node (re)start.
    def run_init_camera_script(context, *args, **kwargs):
        return [
            ExecuteProcess(
                name="init_exposure_parasms_bash",
                cmd=["bash", script_path, video_device],
                output="screen",
            )
        ]

    init_camera_controls = RegisterEventHandler(
        OnProcessStart(
            target_action=hand_camera_node,
            on_start=[OpaqueFunction(function=run_init_camera_script)],
        )
    )

    # Included here so the calibration uses the driver's topic names.
    calibration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("g1_cam_drill_interface"),
                    "launch",
                    "calibrate_hand_camera_launch.py",
                ]
            )
        ),
        condition=IfCondition(LaunchConfiguration("calibrate")),
        launch_arguments={
            "tag_size_m": LaunchConfiguration("tag_size_m"),
            "tag_id": LaunchConfiguration("tag_id"),
            "num_samples": LaunchConfiguration("num_samples"),
        }.items(),
    )

    return LaunchDescription(
        [*ARGUMENTS, hand_camera_node, init_camera_controls, calibration]
    )
