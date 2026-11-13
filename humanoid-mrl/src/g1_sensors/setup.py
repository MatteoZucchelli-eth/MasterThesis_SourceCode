from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_sensors"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (path.join("share", package_name, "launch"), glob(path.join("launch", "*.py"))),
        (
            path.join("share", package_name, "videos"),
            glob(path.join("videos", "*.mp4"))
            + glob(path.join("videos", "*.avi"))
            + glob(path.join("videos", "*.mov")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="mrl@example.com",
    description="Simple sensor publishers for testing and development",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fake_camera_publisher = g1_sensors.camera.fake_camera_publisher:main",
            "head_camera_calibration = "
            "g1_sensors.camera.head_camera_calibration_node:main",
            "head_camera_calibration_tag = "
            "g1_sensors.camera.head_camera_calibration_tag:main",
            "mocap_tf_bridge = g1_sensors.mocap.mocap_tf_bridge:main",
            "wall_plane = g1_sensors.depth.wall_plane_node:main",
        ],
    },
)
