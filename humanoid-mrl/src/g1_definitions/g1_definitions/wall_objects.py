# MIT License
"""Colour x shape taxonomy of the wall objects, shared by sim and control.

Also defines the TF frame names (:func:`object_frame`) both sides must agree on.
Rendering details live in ``g1_simulation.isaacsim.scene``.
"""
from __future__ import annotations

COLORS = ("red", "green", "blue")
SHAPES = ("square", "circle", "triangle")

ALL_CELLS = tuple((color, shape) for color in COLORS for shape in SHAPES)
"""Every ``(colour, shape)`` cell. All of them are built and shown each episode."""

HELD_OUT = (("red", "triangle"), ("green", "circle"), ("blue", "square"))
"""Cells shown as distractors but never commanded in training (Latin-square diagonal).

One per colour and per shape, so eval on them tests composition only.
"""

TARGETABLE_CELLS = tuple(cell for cell in ALL_CELLS if cell not in HELD_OUT)
"""Cells an episode may command: :data:`ALL_CELLS` minus :data:`HELD_OUT`."""


def object_frame(color: str, shape: str) -> str:
    """TF frame for one wall object, e.g. ``red_triangle_target``."""
    return f"{color}_{shape}_target"
