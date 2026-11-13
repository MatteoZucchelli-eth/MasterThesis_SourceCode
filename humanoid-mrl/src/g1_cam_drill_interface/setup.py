from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_cam_drill_interface"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (path.join("share", package_name, "launch"), glob(path.join("launch", "*.py"))),
        (path.join("share", package_name, "config"), glob("config/*.yaml")),
        # Files only: setuptools fails on directories such as scripts/__pycache__.
        (
            path.join("lib", package_name),
            [p for p in glob("scripts/*") if path.isfile(p)],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Simon Arnold",
    maintainer_email="simarnold@ethz.ch",
    description="Camera and Hilti drill interface for Unitree G1",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "calibrate_hand_camera_node = g1_cam_drill_interface.calibrate_hand_camera_node:main",
        ],
    },
)
