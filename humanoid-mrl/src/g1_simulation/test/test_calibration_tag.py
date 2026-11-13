# MIT License
"""Tests for the wall calibration tag (pure layout, no Isaac import).

The tag is built here and read by g1_sensors; the frame convention is asserted
across both packages since a mismatch is undetectable at runtime.
"""

import numpy as np
import pytest
from g1_common.apriltag import cells_per_side, DEFAULT_FAMILY, tag_cells
from g1_simulation.config import IsaacSimCfg
from g1_simulation.isaacsim import calibration_tag as ct

# Skipped when g1_sensors is not in the workspace.
head_camera_calibration = pytest.importorskip(
    "g1_sensors.camera.head_camera_calibration"
)


def test_the_family_matches_the_one_the_generator_defaults_to():
    """calibration_tag's hardcoded family matches g1_common's default."""
    assert ct.TAG_FAMILY == DEFAULT_FAMILY


def test_calibration_mode_is_off_by_default():
    assert not IsaacSimCfg().calibration
    assert IsaacSimCfg().calibration_xyz_m == (0.0, 0.0, 0.0)


def test_translation_injection_is_composed_in_the_optical_frame():
    """XYZ is ``t real_head -> urdf_head``, expressed in the real head frame."""
    base = (-0.5, 0.5, -0.5, 0.5)
    injected = (0.012, -0.007, 0.021)
    assert ct.quaternion_rotate(base, injected) == pytest.approx([0.021, -0.012, 0.007])


def test_the_plaque_carries_a_quiet_zone_around_the_black_square():
    """The detector needs white margin; the black square is what ``tag_size_m`` is."""
    n = cells_per_side(ct.TAG_FAMILY)
    y0, y1, z0, z1 = ct.backing_quad()
    assert (y1 - y0) == pytest.approx(ct.plaque_side_m())
    assert (z1 - z0) == pytest.approx(ct.plaque_side_m())
    assert ct.plaque_side_m() == pytest.approx(
        ct.TAG_SIZE_M * (n + 2 * ct.QUIET_CELLS) / n
    )
    assert ct.cell_pitch_m() * n == pytest.approx(ct.TAG_SIZE_M)


def test_the_black_layer_sits_in_front_of_the_backing():
    """Coplanar faces z-fight, and the black must win or the tag is a white square."""
    assert ct.PATTERN_INSET_M > ct.BACKING_INSET_M


def test_every_black_cell_lands_where_the_calibration_expects_to_see_it():
    """Built black cells match g1_sensors' ``WALL_FROM_TAG`` convention exactly.

    Tag frame: +x image-right, +y image-down; the sim wall frame is world-aligned.
    """
    n = cells_per_side(ct.TAG_FAMILY)
    pitch = ct.cell_pitch_m()
    cells = tag_cells(ct.TAG_ID, ct.TAG_FAMILY)
    origin = np.array([0.0, *ct.TAG_CENTRE_YZ])

    expected = set()
    for row in range(n):
        for column in range(n):
            if not cells[row, column]:
                continue
            in_tag_frame = np.array(
                [
                    (column + 0.5 - n / 2.0) * pitch,  # +x is image-right
                    (row + 0.5 - n / 2.0) * pitch,  # +y is image-down
                    0.0,
                ]
            )
            centre = origin + head_camera_calibration.WALL_FROM_TAG @ in_tag_frame
            expected.add((round(centre[1], 9), round(centre[2], 9)))

    built = {
        (round((y0 + y1) / 2.0, 9), round((z0 + z1) / 2.0, 9))
        for y0, y1, z0, z1 in ct.black_cell_quads()
    }
    assert built == expected
    assert len(built) == int(cells.sum())


def test_the_pattern_stays_inside_the_plaque():
    """A black cell outside the white backing has no quiet zone behind it."""
    y0, y1, z0, z1 = ct.backing_quad()
    for cy0, cy1, cz0, cz1 in ct.black_cell_quads():
        assert y0 < cy0 and cy1 < y1
        assert z0 < cz0 and cz1 < z1


def test_the_tag_is_small_enough_to_stay_in_the_head_camera_frame():
    """At ~0.5 m the plaque fits the head D435 frame yet spans >100 px of black
    square."""
    stand_off_m, focal_px, width_px = 0.5, 606.370361328125, 640
    covered_m = width_px * stand_off_m / focal_px
    assert ct.plaque_side_m() < 0.6 * covered_m
    assert focal_px * ct.TAG_SIZE_M / stand_off_m > 100.0
