from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_state_publisher"


def _collect_urdf_data_files():
    """Return data_files entries that preserve urdf/ and urdf/meshes/ layout."""
    entries = []
    for src in glob("urdf/**", recursive=True):
        if not path.isfile(src):
            continue
        dest_dir = path.join("share", package_name, path.dirname(src))
        entries.append((dest_dir, [src]))
    return entries


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
        *_collect_urdf_data_files(),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="joshua.naef@mrl.ethz.ch",
    description="(Re)Publish G1 state in ROS specific formats",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "state_publisher = g1_state_publisher.state_publisher:main"
        ],
    },
)
