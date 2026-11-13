# MIT License
"""Isaac adapter for the shared elastic band: RigidPrim I/O + gantry keys."""
from __future__ import annotations

import logging

import numpy as np

from g1_simulation.elastic_band import ElasticBandCore
from g1_simulation.isaacsim.carb_keys import is_key_press


class ElasticBand:
    """Simulate an external force that keeps the robot upright.

    Keys: 7 lower the anchor, 8 raise it, 9 toggle the band.
    """

    def __init__(self, articulation, link_name, enable, anchor_xy=(0.0, 0.0)):
        from isaacsim.core.experimental.prims import RigidPrim

        # Anchor above the base/lock XY so the pull is vertical; Z = gantry height.
        self._core = ElasticBandCore(
            anchor=(anchor_xy[0], anchor_xy[1], 2.4), enable=enable
        )
        self._link = RigidPrim(f"/World/robot/{link_name}")

    def on_keyboard_input(self, e):
        if is_key_press(e, "7"):
            logging.debug("lowering robot")
            self._core.lower()
        if is_key_press(e, "8"):
            logging.debug("raising robot")
            self._core.raise_()
        if is_key_press(e, "9"):
            logging.debug("toggling gantry")
            self._core.toggle()

    def step(self) -> None:
        """Apply the force to the robot"""
        if not self._core.enabled:
            return

        pos, _ = self._link.get_world_poses()
        vel, _ = self._link.get_velocities()
        force = self._core.force(np.squeeze(pos)[:3], np.squeeze(vel)[:3])
        self._link.apply_forces(forces=force)
