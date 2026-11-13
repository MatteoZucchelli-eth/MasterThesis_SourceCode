# MIT License
"""Per-episode drill-task sampling: which square to drill, and the sentence for it.

The target cell varies per episode; the instruction is always
"drill in the <colour> <shape>". The target is sampled first and the sentence
generated from it, so collection never parses language.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence, Tuple

from g1_definitions import wall_objects


@dataclass(frozen=True)
class DrillTask:
    """One episode's sampled task: where to drill and how the instruction reads."""

    color: str
    """Target colour, e.g. ``"green"``."""
    shape: str
    """Target shape, e.g. ``"triangle"``."""
    target_frame: str
    """TF frame of that cell as published by the sim, e.g. ``"green_triangle_target"``."""
    description: str
    """Language instruction recorded with the episode."""


class DrillTaskSampler:
    """Uniformly samples a target cell (colour x shape) for the fixed instruction.

    Both attributes are always named, since all nine cells are on the wall.
    ``CELLS`` excludes the held-out combinations (distractors only, used to test
    composition at eval). The taxonomy is shared with the sim via
    :mod:`g1_definitions.wall_objects`.
    """

    COLORS = wall_objects.COLORS
    SHAPES = wall_objects.SHAPES
    HELD_OUT = wall_objects.HELD_OUT
    CELLS = wall_objects.TARGETABLE_CELLS
    TEMPLATES = ("drill in the {color} {shape}",)
    """Instruction phrasings :meth:`sample` draws from."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def sample(self, cells: Sequence[Tuple[str, str]] | None = None) -> DrillTask:
        """Draw one task. ``cells`` overrides the default choice set (eval runs)."""
        color, shape = self._rng.choice(tuple(cells) if cells else self.CELLS)
        template = self._rng.choice(self.TEMPLATES)
        return DrillTask(
            color=color,
            shape=shape,
            target_frame=wall_objects.object_frame(color, shape),
            description=template.format(color=color, shape=shape),
        )


def resolve_target_frame(description: str) -> str | None:
    """Recover the TF frame a task sentence refers to, or None if it cannot.

    **Scoring only** (inverse of :meth:`DrillTaskSampler.sample`, for typed
    instructions). Strict: exactly one colour word and exactly one shape word
    must appear, otherwise None (target-relative metrics then record NaN).
    """
    found_colors = {
        color
        for color in DrillTaskSampler.COLORS
        if re.search(rf"\b{color}\b", description, re.IGNORECASE)
    }
    found_shapes = {
        shape
        for shape in DrillTaskSampler.SHAPES
        if re.search(rf"\b{shape}s?\b", description, re.IGNORECASE)
    }
    if len(found_colors) != 1 or len(found_shapes) != 1:
        return None
    return wall_objects.object_frame(found_colors.pop(), found_shapes.pop())
