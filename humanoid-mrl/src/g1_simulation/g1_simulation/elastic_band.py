# MIT License
"""Simulator-independent elastic band ("gantry"): spring-damper toward an anchor.

Keeps the robot upright while it hangs unlocked. The Isaac and MuJoCo
simulations share this force law and only differ in how they read the link
state and apply the force (see their thin adapters).
"""
from __future__ import annotations

import numpy as np


class ElasticBandCore:
    def __init__(
        self,
        anchor: tuple[float, float, float],
        stiffness: float = 200.0,
        damping: float = 100.0,
        enable: bool = True,
    ) -> None:
        self.point = np.array(anchor, dtype=np.float64)
        self.enabled = enable
        self._stiffness = stiffness
        self._damping = damping

    def force(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Spring-damper pull toward the anchor for a link at ``pos``/``vel``."""
        delta_x = self.point - pos
        distance = np.linalg.norm(delta_x)
        direction = delta_x / distance
        v = np.dot(vel, direction)
        return (self._stiffness * distance - self._damping * v) * direction

    def lower(self, step: float = 0.1) -> None:
        self.point[2] -= step

    def raise_(self, step: float = 0.1) -> None:
        self.point[2] += step

    def toggle(self) -> None:
        self.enabled = not self.enabled
