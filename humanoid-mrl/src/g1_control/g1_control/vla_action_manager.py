"""VLA action-chunk management helpers.

Separates the double-buffering / chunk-stepping logic from the main node so
that ``ControllerManagerNode`` only deals with high-level calls.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from g1_definitions import definitions
from g1_definitions.definitions import VLA_CHUNK_SIZE

from g1_control.utils.utils import create_action_buffer


class VLAActionManager:
    """Manages the double-buffered action chunk produced by the VLA client.

    The VLA client writes into ``action_buffer`` (shared, lock-protected).
    The control loop reads from ``local_action_buffer`` (private copy) and
    advances ``current_index`` every step until a new chunk arrives.
    """

    def __init__(
        self,
        chunk_size: int = VLA_CHUNK_SIZE,
        num_upper_body_joints: int = 28,
    ) -> None:
        self.chunk_size = chunk_size

        self.action_buffer: dict = create_action_buffer(
            chunk_size, num_upper_body_joints
        )
        self.local_action_buffer: dict = create_action_buffer(
            chunk_size, num_upper_body_joints
        )

        self.action_lock = threading.Lock()
        self.new_action_chunk = threading.Event()

        self.current_index: int = 0
        self.has_first_chunk: bool = False

    # ------------------------------------------------------------------
    # Called from the control loop
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop any chunk (and pending new-chunk flag) from a previous VLA session."""
        self.new_action_chunk.clear()
        self.current_index = 0
        self.has_first_chunk = False

    def step(self) -> None:
        """Check for a fresh chunk and advance the step index."""
        if self.new_action_chunk.is_set():
            self.new_action_chunk.clear()
            self.current_index = 0
            with self.action_lock:
                for key, arr in self.action_buffer.items():
                    np.copyto(self.local_action_buffer[key], arr)
            self.has_first_chunk = True

    def advance_index(self) -> None:
        """Move to the next action step, holding on the last one once exhausted."""
        if self.current_index < self.chunk_size - 1:
            self.current_index += 1

    def get(self, key: str) -> np.ndarray:
        """Return the slice for the current step from the local buffer."""
        return self.local_action_buffer[key][self.current_index, :]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_right_eef_9d(self) -> Optional[np.ndarray]:
        """Current-step right drill-tip 9D pose from the local buffer."""
        if not self.has_first_chunk:
            return None
        return self.get(definitions.RIGHT_EEF_KEY)
