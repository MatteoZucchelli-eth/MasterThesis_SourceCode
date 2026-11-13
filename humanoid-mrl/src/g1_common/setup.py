from glob import glob
from os import path

from setuptools import find_packages, setup

package_name = "g1_common"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (path.join("share", package_name, "lib"), glob(path.join("lib", "*.so"))),
    ],
    install_requires=["setuptools", "ament-index-python", "unitree-hg"],
    zip_safe=True,
    maintainer="MRL",
    maintainer_email="brandesa@ethz.ch",
    description="Common toolbox to run the G1",
    license="Apache-2.0",
    tests_require=["pytest"],
)
