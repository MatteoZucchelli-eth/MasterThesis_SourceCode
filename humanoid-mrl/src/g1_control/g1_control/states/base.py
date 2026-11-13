# MIT License
"""Shared context and interface for the per-FSM-state handlers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from g1_common.data import RobotState
from g1_definitions import definitions
from tf2_ros import Buffer

from g1_control.utils.wbc_robot_model import RobotModel


@dataclass
class EefTargets:
    """Commanded end-effector poses the recorder writes as the action.

    ``None`` means "no target commanded" — the recorder then falls back to the
    measured FK pose, so the action column never contains zeros.
    """

    left_9d: np.ndarray | None = None
    right_9d: np.ndarray | None = None

    def clear(self) -> None:
        self.left_9d = None
        self.right_9d = None


@dataclass
class ControlContext:
    """Control-loop state shared between the node and the state handlers.

    The node owns and updates it. Handlers write only ``targets`` and
    ``robot_state`` (AUTO_COLLECT also writes ``drill_target_frame``).
    """

    robot_model: RobotModel
    robot_state: RobotState
    tf_buffer: Buffer
    default_upper_body_pose: np.ndarray
    upper_body_indices: list[int]
    left_eef_frame: str
    right_eef_frame: str
    control_dt: float
    full_q: np.ndarray
    """Measured full pinocchio q; the node refreshes it in place per LowState."""
    targets: EefTargets = field(default_factory=EefTargets)
    current_left_eef_9d: np.ndarray | None = None
    current_right_eef_9d: np.ndarray | None = None
    latest_rc: list[int] | None = None
    """Latest wireless_remote buffer from LowState."""
    fsm_state: int = definitions.FSMState.DAMPING.value
    has_drill_tip: bool = False
    drill_target_frame: str | None = "red_square_target"
    """TF frame ``{colour}_{shape}_target`` of the current task's wall object.

    Resolved from each task instruction (or set by AUTO_COLLECT). None when the
    prompt names no unique colour+shape; target metrics then record NaN."""


class StateHandler(ABC):
    """Per-tick behaviour of one FSM state.

    The node is used only at construction (publishers, subscriptions,
    parameters, logger); at runtime handlers use the :class:`ControlContext`.
    """

    def on_enter(self, previous_state: int) -> None:
        """Called once when the FSM enters this handler's state."""

    def on_exit(self) -> None:
        """Called once when the FSM leaves this handler's state."""

    @abstractmethod
    def update(self) -> None:
        """Called every control tick while this handler's state is active.

        Writes ``robot_state`` (twist / upper-body targets) and ``ctx.targets``.
        """
