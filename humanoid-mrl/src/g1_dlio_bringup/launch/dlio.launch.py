# MIT License

# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
# Copyright (c) Kenny J. Chen, Ryan Nemiroff, and Brett T. Lopez
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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


def generate_launch_description():
    current_pkg = FindPackageShare("g1_dlio_bringup")

    # Set default arguments
    pointcloud_topic = LaunchConfiguration("pointcloud_topic", default="/livox/points")
    imu_topic = LaunchConfiguration("imu_topic", default="/livox/imu")

    # Define arguments
    declare_pointcloud_topic_arg = DeclareLaunchArgument(
        "pointcloud_topic",
        default_value=pointcloud_topic,
        description="Pointcloud topic name",
    )
    declare_imu_topic_arg = DeclareLaunchArgument(
        "imu_topic", default_value=imu_topic, description="IMU topic name"
    )

    # Load parameters
    dlio_yaml_path = PathJoinSubstitution([current_pkg, "config", "dlio.yaml"])
    dlio_params_yaml_path = PathJoinSubstitution([current_pkg, "config", "params.yaml"])

    # DLIO Odometry Node
    dlio_odom_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_odom_node",
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path],
        remappings=[
            ("pointcloud", pointcloud_topic),
            ("imu", imu_topic),
            ("odom", "dlio/odom_node/odom"),
            ("pose", "dlio/odom_node/pose"),
            ("path", "dlio/odom_node/path"),
            ("kf_pose", "dlio/odom_node/keyframes"),
            ("kf_cloud", "dlio/odom_node/pointcloud/keyframe"),
            ("deskewed", "dlio/odom_node/pointcloud/deskewed"),
        ],
    )

    # DLIO Mapping Node
    dlio_map_node = Node(
        package="direct_lidar_inertial_odometry",
        executable="dlio_map_node",
        output="screen",
        parameters=[dlio_yaml_path, dlio_params_yaml_path],
        remappings=[
            ("keyframes", "dlio/odom_node/pointcloud/keyframe"),
        ],
    )

    return LaunchDescription(
        [
            declare_pointcloud_topic_arg,
            declare_imu_topic_arg,
            dlio_odom_node,
            dlio_map_node,
        ]
    )
