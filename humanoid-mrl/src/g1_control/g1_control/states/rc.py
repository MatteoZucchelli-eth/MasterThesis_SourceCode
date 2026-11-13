# MIT License
"""Default handler: map RC joystick input to a base twist command.

Active in every FSM state without a dedicated handler (MOVE, READY, DAMPING, ...).
"""
from __future__ import annotations

from g1_common import rc

from g1_control.states.base import ControlContext, StateHandler


class RcHandler(StateHandler):
    def __init__(self, node, ctx: ControlContext) -> None:
        self._ctx = ctx
        # Gains are read once; they do not change at runtime.
        self._gains = {
            name: param.value
            for name, param in node.get_parameters_by_prefix("gains").items()
        }

    def update(self) -> None:
        if self._ctx.latest_rc is None:
            return
        twist, _ = rc.get_velocity_commands(self._ctx.latest_rc, self._gains)
        self._ctx.robot_state.twist_command = twist
