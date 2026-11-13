# MIT License
"""Unit tests for the coarse/fine source wiring (no ROS, no servers)."""

from g1_control.vla.sources import VlaRole, VlaSources


class _FakeClient:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeActions:
    """Stand-in for VLAActionManager; only its presence matters here."""


def test_only_the_coarse_role_drives_the_arm():
    """The fine role carries no action chunks and never drives the arm."""
    sources = VlaSources(
        coarse=VlaRole(_FakeClient("coarse"), _FakeActions()),
        fine=VlaRole(_FakeClient("detector")),
    )
    assert sources.coarse.drives_arm
    assert not sources.fine.drives_arm


def test_unique_keeps_both_clients_in_role_order():
    coarse, detector = _FakeClient("coarse"), _FakeClient("detector")
    sources = VlaSources(coarse=VlaRole(coarse, _FakeActions()), fine=VlaRole(detector))
    assert sources.unique == [coarse, detector]
