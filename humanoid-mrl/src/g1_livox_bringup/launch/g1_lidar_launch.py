# MIT License
#
# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
#
# Portions of this file are derived from:
# Livox ROS2 Driver (https://github.com/tu-darmstadt-ros-pkg/livox_ros_driver2/tree/jazzy)
# Copyright (c) Livox. Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


livox_ros2_default_params = {
    "transfer_format": 0,  # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
    "multi_topic": 0,  # 0-All LiDARs share the same topic, 1-One LiDAR one topic
    "data_src": 0,  # 0-lidar, others-Invalid data src
    "publish_freq": 10.0,  # freqency of publish, 5.0, 10.0, 20.0, 50.0, etc.
    "user_config_path": "g1_mid360.json",  # Configuration file specifying lidar and host PC IPs.
    "output_data_type": 0,
    "frame_id": "livox_frame",  # NOTE(@naefjo): This param is unused :) pls change the one in the json file.
    "lvx_file_path": "/home/livox/livox_test.lvx",
    "cmdline_input_bd_code": "livox0000000001",
}


def generate_launch_description():

    # Declare the launch arguments that can be overwritten from outside
    declare_transfer_format_cmd = DeclareLaunchArgument(
        "transfer_format",
        default_value=str(livox_ros2_default_params["transfer_format"]),
        description="Transfer format: 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format",
    )

    declare_multi_topic_cmd = DeclareLaunchArgument(
        "multi_topic",
        default_value=str(livox_ros2_default_params["multi_topic"]),
        description="Multi topic: 0-All LiDARs share the same topic, 1-One LiDAR one topic",
    )

    declare_data_src_cmd = DeclareLaunchArgument(
        "data_src",
        default_value=str(livox_ros2_default_params["data_src"]),
        description="Data source: 0-lidar, others-Invalid data src",
    )

    declare_publish_freq_cmd = DeclareLaunchArgument(
        "publish_freq",
        default_value=str(livox_ros2_default_params["publish_freq"]),
        description="Publish frequency",
    )

    declare_output_type_cmd = DeclareLaunchArgument(
        "output_data_type",
        default_value=str(livox_ros2_default_params["output_data_type"]),
        description="Output data type",
    )

    declare_frame_id_cmd = DeclareLaunchArgument(
        "frame_id",
        default_value=livox_ros2_default_params["frame_id"],
        description="Frame ID",
    )

    declare_lvx_file_path_cmd = DeclareLaunchArgument(
        "lvx_file_path",
        default_value=livox_ros2_default_params["lvx_file_path"],
        description="LVX file path",
    )
    declare_user_config_path_cmd = DeclareLaunchArgument(
        "user_config_path",
        default_value=livox_ros2_default_params["user_config_path"],
        description="User config path",
    )

    declare_cmdline_bd_code_cmd = DeclareLaunchArgument(
        "cmdline_input_bd_code",
        default_value=livox_ros2_default_params["cmdline_input_bd_code"],
        description="Command line BD code",
    )

    livox_ros2_params = [
        {"xfer_format": LaunchConfiguration("transfer_format")},
        {"multi_topic": LaunchConfiguration("multi_topic")},
        {"data_src": LaunchConfiguration("data_src")},
        {"publish_freq": LaunchConfiguration("publish_freq")},
        {"output_data_type": LaunchConfiguration("output_data_type")},
        {"frame_id": LaunchConfiguration("frame_id")},
        {"lvx_file_path": LaunchConfiguration("lvx_file_path")},
        {
            "user_config_path": PathJoinSubstitution(
                [
                    FindPackageShare("g1_livox_bringup"),
                    "config",
                    LaunchConfiguration("user_config_path"),
                ]
            )
        },
        {"cmdline_input_bd_code": LaunchConfiguration("cmdline_input_bd_code")},
    ]

    livox_driver = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=livox_ros2_params,
    )

    return LaunchDescription(
        [
            declare_transfer_format_cmd,
            declare_multi_topic_cmd,
            declare_data_src_cmd,
            declare_publish_freq_cmd,
            declare_output_type_cmd,
            declare_frame_id_cmd,
            declare_lvx_file_path_cmd,
            declare_user_config_path_cmd,
            declare_cmdline_bd_code_cmd,
            livox_driver,
        ]
    )
