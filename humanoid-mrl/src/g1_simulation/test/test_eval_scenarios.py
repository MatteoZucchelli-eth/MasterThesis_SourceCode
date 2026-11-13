# MIT License
"""Tests for the generated evaluation scenario table (pure data, no Isaac import).

Checks properties a hand edit, stale table or taxonomy change would break;
generator-side arithmetic (margins, IK residuals) is not re-derived.
"""
from collections import Counter

import pytest

from g1_definitions import wall_objects
from g1_simulation.config import IsaacSimCfg
from g1_simulation.isaacsim.eval_scenarios import SCENARIOS


def test_every_targetable_cell_is_probed_once_per_family():
    """Six targetable cells x two probe families (colour, shape) = twelve scenarios."""
    seen = Counter((sc["probe"], tuple(sc["target_cell"])) for sc in SCENARIOS)
    expected = {
        (probe, cell)
        for probe in ("colour", "shape")
        for cell in wall_objects.TARGETABLE_CELLS
    }
    assert set(seen) == expected
    assert all(count == 1 for count in seen.values())


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"S{sc['id']}")
def test_the_scene_makes_the_other_attribute_useless(sc):
    """A probe's three objects must differ in exactly the attribute under test."""
    cells = [tuple(cell) for cell in sc["objects"]]
    assert len(cells) == 3
    colours = {colour for colour, _ in cells}
    shapes = {shape for _, shape in cells}
    if sc["probe"] == "colour":
        assert shapes == {sc["constant"]}
        assert colours == set(wall_objects.COLORS)
    else:
        assert colours == {sc["constant"]}
        assert shapes == set(wall_objects.SHAPES)


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"S{sc['id']}")
def test_the_commanded_cell_is_on_the_wall_and_was_trained(sc):
    cell = tuple(sc["target_cell"])
    assert cell in {tuple(c) for c in sc["objects"]}
    assert cell in wall_objects.TARGETABLE_CELLS
    assert cell not in wall_objects.HELD_OUT
    assert sc["target_frame"] == wall_objects.object_frame(*cell)


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"S{sc['id']}")
def test_the_instruction_names_the_target_and_nothing_else(sc):
    """The instruction names exactly one colour and one shape (g1_control's parser is
    strict)."""
    colour, shape = sc["target_cell"]
    words = sc["instruction"].split()
    assert [w for w in words if w in wall_objects.COLORS] == [colour]
    assert [w for w in words if w in wall_objects.SHAPES] == [shape]


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"S{sc['id']}")
def test_objects_are_separated_enough_to_be_told_apart(sc):
    """Objects are far enough apart that a near miss is not a wrong-object hit."""
    positions = [tuple(yz) for yz in sc["objects"].values()]
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            distance = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            assert distance >= 0.049, f"{a} and {b} are {distance * 1000:.0f} mm apart"


def test_targets_spread_across_the_wall_rather_than_clustering():
    """Precision is reported against wall position, which needs spread.

    The z requirement (0.07 m) is bounded by the height of ``target_z_range``.
    """
    targets = [sc["objects"][tuple(sc["target_cell"])] for sc in SCENARIOS]
    ys = sorted({round(y, 4) for y, _ in targets})
    zs = sorted({round(z, 4) for _, z in targets})
    assert len(ys) >= 4 and len(zs) >= 3
    assert ys[-1] - ys[0] >= 0.08, "targets span less than 8 cm of wall y"
    assert zs[-1] - zs[0] >= 0.07, "targets span less than 7 cm of wall z"


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda sc: f"S{sc['id']}")
def test_objects_sit_inside_the_band_the_policy_was_trained_on(sc):
    """Every object, target and distractor, inside ``IsaacSimCfg``'s sample band.

    The band is the training distribution over wall position; this catches a
    stale table after the band changes.
    """
    cfg = IsaacSimCfg()
    y_lo, y_hi = cfg.target_y_range
    z_lo, z_hi = cfg.target_z_range
    for cell, (y, z) in sc["objects"].items():
        assert y_lo <= y <= y_hi, f"{cell} y={y:+.3f} outside {cfg.target_y_range}"
        assert z_lo <= z <= z_hi, f"{cell} z={z:.3f} outside {cfg.target_z_range}"


def test_every_scenario_carries_the_verification_the_generator_ran():
    """Stored margins and residuals meet the generator's thresholds."""
    for sc in SCENARIOS:
        assert sc["min_margin_px"] >= 20.0
        assert sc["ik_residual_mm"] <= 1.0
        u, v = sc["target_ego_px"]
        assert 0.0 <= u <= 480.0 and 0.0 <= v <= 480.0
