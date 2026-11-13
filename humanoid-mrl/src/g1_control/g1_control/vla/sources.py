# MIT License
"""Coarse and fine model roles of the drilling pipeline.

* **coarse**: action VLA on ``server_port``; drives the arm into wrist-camera range.
* **fine**: detector-only :class:`~g1_control.vla.perception_client.PerceptionClient`
  on ``fine_server_port``; its estimate is what the scripted approach is planned
  onto. It never drives the arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from g1_control.vla_action_manager import VLAActionManager


@dataclass(frozen=True)
class VlaRole:
    """One source: its client, plus its action buffer if it drives the arm."""

    client: object
    actions: "VLAActionManager | None" = None

    @property
    def drives_arm(self) -> bool:
        """Whether this role produces action chunks the state can play back."""
        return self.actions is not None


@dataclass(frozen=True)
class VlaSources:
    """The role assignment: a driving VLA and a separate detector."""

    coarse: VlaRole
    """Carries the approach and drives the arm."""
    fine: VlaRole
    """Owns the close range: the estimate the scripted approach is planned on."""

    @property
    def unique(self) -> list:
        """Both clients, so sensor updates fan out once to each."""
        return [self.coarse.client, self.fine.client]
