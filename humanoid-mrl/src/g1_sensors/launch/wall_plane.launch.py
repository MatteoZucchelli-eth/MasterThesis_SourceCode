# MIT License
"""Brings up the wall plane estimator on its own.

Included by the simulation's full_launch; usable standalone:

    ros2 launch g1_sensors wall_plane.launch.py

Only commonly retargeted arguments are declared; other defaults live in
``g1_sensors.depth.wall_plane.WallEstimatorConfig``. Topic defaults come from
``g1_definitions.topics`` since a forwarded empty argument would override the
node's default.
"""

from g1_definitions import topics
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_ARGUMENTS = {
    "depth_topic": (
        topics.TOPIC_CAMERA_DEPTH,
        "Rectified depth image to fit the wall from.",
    ),
    "camera_info_topic": (
        topics.TOPIC_CAMERA_DEPTH_INFO,
        "Intrinsics to deproject it with. Must describe the DEPTH image: in sim "
        "the colour camera_info does, because both streams come off one render "
        "product, but on the robot that holds only with align_depth enabled.",
    ),
    "reference_frame": (
        "pelvis",
        "Frame the plane and the cloud are published in. Must be rigid with the "
        "wall for the plane to be usable from an earlier snapshot, which is why "
        "it is not the camera's own frame.",
    ),
    "depth_bias_m": (
        "0.0",
        "Optical-axis depth correction in metres. Positive moves valid readings "
        "further from the camera; real-robot launches supply the measured value "
        "(0.03, bench-measured on the D435). Zero here because the only caller "
        "that does NOT set it is the simulation's full_launch.py, whose rendered "
        "depth has no range error to correct -- a hardware value left in this "
        "default pushes the fitted plane ~20 mm past the wall, and with "
        "vla_planner_depth_source: stereo that lands one-for-one on the target "
        "range the planner flies to.",
    ),
}


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(name, default_value=default, description=description)
            for name, (default, description) in _ARGUMENTS.items()
        ]
        + [
            Node(
                package="g1_sensors",
                executable="wall_plane",
                name="wall_plane",
                output="screen",
                parameters=[{name: LaunchConfiguration(name) for name in _ARGUMENTS}],
            )
        ]
    )
