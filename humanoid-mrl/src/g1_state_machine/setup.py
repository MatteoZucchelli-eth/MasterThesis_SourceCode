from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_state_machine"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            path.join("share", package_name, "launch"),
            glob(path.join("launch", "*launch.[pxy][yma]*")),
        ),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "transitions"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="brandesa@ethz.ch",
    description="Implements state machine logic for switching between operating states",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": ["state_machine = g1_state_machine.state_machine_node:main"],
    },
)
