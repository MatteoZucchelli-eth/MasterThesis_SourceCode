# MIT License
"""Tests for the per-episode drill-task sampler (colour x shape grounding)."""

import random

import pytest

from g1_control.utils.drill_task_sampler import DrillTaskSampler, resolve_target_frame


def test_description_names_the_sampled_cell_and_frame_matches():
    sampler = DrillTaskSampler(random.Random(0))
    for _ in range(200):
        task = sampler.sample()
        assert task.color in DrillTaskSampler.COLORS
        assert task.shape in DrillTaskSampler.SHAPES
        assert task.color in task.description
        assert task.shape in task.description
        assert task.target_frame == f"{task.color}_{task.shape}_target"


def test_held_out_cells_are_never_commanded():
    # Held-out cells appear only as distractors; they are commanded only at eval.
    sampler = DrillTaskSampler(random.Random(11))
    held = set(DrillTaskSampler.HELD_OUT)
    for _ in range(3000):
        task = sampler.sample()
        assert (task.color, task.shape) not in held


def test_every_cell_is_reachable_under_the_fixed_phrasing():
    sampler = DrillTaskSampler(random.Random(1))
    descriptions = {sampler.sample().description for _ in range(4000)}
    expected = {
        template.format(color=color, shape=shape)
        for template in DrillTaskSampler.TEMPLATES
        for color, shape in DrillTaskSampler.CELLS
    }
    assert descriptions == expected


def test_no_cell_is_starved():
    # Every targetable cell gets a roughly uniform share.
    sampler = DrillTaskSampler(random.Random(2))
    counts = {cell: 0 for cell in DrillTaskSampler.CELLS}
    n = 6000
    for _ in range(n):
        task = sampler.sample()
        counts[(task.color, task.shape)] += 1
    for count in counts.values():
        assert count > n / len(counts) * 0.8


def test_explicit_cells_override_the_default_choice_set():
    # How an eval run asks for a held-out combination.
    sampler = DrillTaskSampler(random.Random(5))
    for _ in range(50):
        task = sampler.sample(cells=[("red", "triangle")])
        assert (task.color, task.shape) == ("red", "triangle")
        assert task.target_frame == "red_triangle_target"


def test_seeded_sampler_is_reproducible():
    a = DrillTaskSampler(random.Random(7))
    b = DrillTaskSampler(random.Random(7))
    assert [a.sample() for _ in range(50)] == [b.sample() for _ in range(50)]


def test_resolver_inverts_every_sentence_the_sampler_can_produce():
    sampler = DrillTaskSampler(random.Random(3))
    for _ in range(500):
        task = sampler.sample()
        assert resolve_target_frame(task.description) == task.target_frame


@pytest.mark.parametrize(
    "description",
    [
        "DRILL IN THE BLUE SQUARE",
        "  drill in the Green triangle  ",
        "drill in the red circle.",
    ],
)
def test_resolver_is_case_and_punctuation_tolerant(description):
    assert resolve_target_frame(description) is not None


@pytest.mark.parametrize(
    "description",
    [
        "drill in the leftmost square",  # spatial reference, no colour
        "drill in the purple square",  # colour outside the task family
        "drill in the reddish square",  # substring, not the colour word
        "drill in the red one",  # colour without a shape: three candidates
        "drill in the triangle",  # shape without a colour: three candidates
        "",
    ],
)
def test_unresolvable_instructions_return_none(description):
    assert resolve_target_frame(description) is None


@pytest.mark.parametrize(
    "description",
    [
        "drill in the red square, not the blue triangle",  # two colours
        "drill in the red square, not the red circle",  # two shapes
    ],
)
def test_ambiguous_instruction_returns_none(description):
    assert resolve_target_frame(description) is None
