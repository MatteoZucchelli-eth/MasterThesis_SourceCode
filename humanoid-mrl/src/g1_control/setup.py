from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (path.join("share", package_name, "launch"), glob(path.join("launch", "*.py"))),
        (
            path.join("share", package_name, "models"),
            glob(path.join("models", "*.onnx")),
        ),
        (path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "onnx", "onnxruntime", "numpy"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="alexander.hansson@mrl.ethz.ch",
    description="RL policy controller for Unitree G1",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_manager_node = g1_control.nodes.controller_manager_node:main",
            "task_instruction_node = g1_control.nodes.task_instruction_node:main",
            "debug_viz_node = g1_control.nodes.debug_viz_node:main",
            "goal_pose_publisher_node = g1_control.nodes.goal_pose_publisher_node:main",
            "auto_collect_bringup_node = g1_control.nodes.auto_collect_bringup_node:main",
            "vla_eval_bringup_node = g1_control.nodes.vla_eval_bringup_node:main",
            "motor_health_node = g1_control.nodes.motor_health_node:main",
        ],
    },
)
