from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_simulation"

g1_xml_path = path.join("assets", "mujoco", "g1")
g1_mesh_path = path.join("assets", "mujoco", "g1", "meshes")
g1_usd_path = path.join("assets", "isaacsim", "g1")
g1_usd_configuration_path = path.join("assets", "isaacsim", "g1", "configuration")

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            path.join("share", package_name, g1_xml_path),
            glob(path.join(g1_xml_path, "*.xml")),
        ),
        (
            path.join("share", package_name, g1_mesh_path),
            glob(path.join(g1_mesh_path, "*.STL")),
        ),
        (
            path.join("share", package_name, g1_usd_path),
            glob(path.join(g1_usd_path, "*.usd")),
        ),
        (
            path.join("share", package_name, g1_usd_configuration_path),
            glob(path.join(g1_usd_configuration_path, "*.usd")),
        ),
        (
            path.join("share", package_name, "launch"),
            glob(path.join("launch", "*.py")),
        ),
    ],
    install_requires=["setuptools", "mujoco", "unitree-hg"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="joshua.naef@mrl.ethz.ch",
    description="Simulation for unitree G1",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_sim = g1_simulation.mujoco.mujoco_simulation_node:main",
            "isaacsim_sim = g1_simulation.isaacsim.isaacsim_simulation_node:entrypoint",
        ],
    },
)
