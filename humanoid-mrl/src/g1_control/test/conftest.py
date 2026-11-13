# MIT License
"""Shared pytest fixtures for g1_control unit tests."""

import os

import pytest


@pytest.fixture(scope="session")
def robot_model():
    """A G1 RobotModel built from the deployment URDF.

    Skips if the URDF (shipped by g1_state_publisher) or its dependencies are
    not available in the test environment.
    """
    pin = pytest.importorskip("pinocchio")  # noqa: F841
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("g1_state_publisher")
    except Exception as exc:  # package not built / not on path
        pytest.skip(f"g1_state_publisher share dir unavailable: {exc}")

    urdf = os.path.join(share, "urdf", "g1_29_dof_with_hand.urdf.xml")
    if not os.path.exists(urdf):
        pytest.skip(f"URDF not found at {urdf}")

    from g1_control.utils.g1_supplemental_info import G1SupplementalInfo
    from g1_control.utils.wbc_robot_model import RobotModel

    return RobotModel(
        urdf,
        os.path.join(share, "urdf"),
        set_floating_base=False,
        supplemental_info=G1SupplementalInfo(),
    )


@pytest.fixture(scope="session")
def drill_robot_model():
    """A G1 RobotModel built from the drill URDF (the one with drill_tip_link)."""
    pytest.importorskip("pinocchio")
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("g1_state_publisher")
    except Exception as exc:  # package not built / not on path
        pytest.skip(f"g1_state_publisher share dir unavailable: {exc}")

    urdf = os.path.join(share, "urdf", "g1_29_dof_with_hand_drill.urdf.xml")
    if not os.path.exists(urdf):
        pytest.skip(f"drill URDF not found at {urdf}")

    from g1_control.utils.g1_supplemental_info import G1SupplementalInfo
    from g1_control.utils.wbc_robot_model import RobotModel

    return RobotModel(
        urdf,
        os.path.join(share, "urdf"),
        set_floating_base=False,
        supplemental_info=G1SupplementalInfo(),
    )
