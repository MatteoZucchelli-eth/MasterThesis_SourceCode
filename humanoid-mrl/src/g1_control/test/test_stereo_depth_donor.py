# MIT License
"""Unit tests for StereoDepthDonor: re-ranging detections on the measured wall.

Rebuilt samples must keep their bearing (move along the ray only) as well as
change range.
"""

import numpy as np
import pytest

from g1_common.plane import Plane
from g1_control.vla.depth_donor import (
    DEPTH_SOURCE_STEREO,
    DEPTH_SOURCES,
    StereoDepthDonor,
    WallPlaneSlot,
)
from g1_control.vla.tip_estimate import mat_to_pose_9d

# Identity camera->tip extrinsic, so tip-frame == camera-frame positions.
# The extrinsic itself is covered by test_the_extrinsic_is_applied_to_the_result.
IDENTITY = np.eye(4)
# The pelvis-frame anchor: tip at the origin, unrotated, so pelvis == camera too.
FLAT_ANCHOR = mat_to_pose_9d(np.eye(4))


def wall(z_m: float = 1.0, tilt: float = 0.25) -> Plane:
    """An oblique wall ``z_m`` down the camera's optical axis, facing it.

    Oblique so ray intersection and plain z-matching give different answers.
    """
    normal = np.array([tilt, 0.0, -1.0])
    return Plane(
        point=np.array([0.0, 0.0, z_m]),
        normal=normal / np.linalg.norm(normal),
        inliers=5000,
        sampled=19200,
        rms_m=0.002,
    )


def facing_wall(z_m: float = 1.0) -> Plane:
    """A wall square-on to the camera at ``z_m`` down the optical axis."""
    return Plane(
        point=np.array([0.0, 0.0, z_m]),
        normal=np.array([0.0, 0.0, -1.0]),
        inliers=5000,
        sampled=19200,
        rms_m=0.002,
    )


def detection(position, *, seq: int = 1, anchor=None):
    """An anchored close-range prediction with the target at ``position``."""
    pose = np.eye(4)
    pose[:3, 3] = position
    return (
        mat_to_pose_9d(pose),
        FLAT_ANCHOR if anchor is None else anchor,
        0.0,
        seq,
    )


def donor(plane_source, *, max_age_s: float = 2.0, max_shift_m: float = 0.06):
    return StereoDepthDonor(
        lambda: IDENTITY,
        plane_source,
        max_age_s=max_age_s,
        max_shift_m=max_shift_m,
    )


def slot_with(plane: Plane | None) -> WallPlaneSlot:
    slot = WallPlaneSlot()
    if plane is not None:
        slot.publish(plane, frame_id="pelvis")
    return slot


# ── The selector ─────────────────────────────────────────────────────────────


def test_stereo_is_a_selectable_depth_source():
    assert DEPTH_SOURCE_STEREO in DEPTH_SOURCES


# ── Re-ranging a detection ───────────────────────────────────────────────────


def test_a_detection_short_of_the_wall_is_pushed_onto_it():
    """A detection at 0.97 m on a wall measured at 1.00 m is re-ranged to the wall."""
    surface = facing_wall(1.0)
    d = donor(slot_with(surface).latest)
    result = d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert result.anchored is not None, result.reason
    np.testing.assert_allclose(result.anchored[0][:3], [0.0, 0.0, 1.0], atol=1e-9)


def test_the_bearing_survives_and_the_point_lands_on_the_wall():
    """On an oblique wall and off-axis ray, both range and bearing are asserted."""
    surface = wall(1.0)
    d = donor(slot_with(surface).latest)
    # Detector placed 3% short along a known off-axis ray; expected answer
    # is the wall point on that bearing.
    direction = np.array([0.12, -0.05, 0.95])
    on_wall = surface.intersect_ray(direction) * direction
    ray = 0.97 * on_wall

    result = d.fuse(detection(ray), 0.0)
    assert result.anchored is not None, result.reason
    rebuilt = np.asarray(result.anchored[0][:3])
    np.testing.assert_allclose(rebuilt, on_wall, atol=1e-12)
    np.testing.assert_allclose(np.cross(rebuilt, ray), np.zeros(3), atol=1e-12)
    assert surface.signed_distance(rebuilt) == pytest.approx(0.0, abs=1e-12)


def test_a_detection_beyond_the_wall_is_pulled_back_onto_it():
    """Detections beyond the wall are re-ranged the same way."""
    d = donor(slot_with(facing_wall(1.0)).latest)
    result = d.fuse(detection([0.0, 0.0, 1.04]), 0.0)
    assert result.anchored is not None, result.reason
    assert result.anchored[0][2] == pytest.approx(1.0)
    assert result.fused_depth_m < result.fine_depth_m


def test_the_orientation_is_left_alone():
    """Only the position moves; rot6d (predicted wall normal) is kept."""
    d = donor(slot_with(facing_wall(1.0)).latest)
    original = detection([0.0, 0.0, 0.97])
    result = d.fuse(original, 0.0)
    np.testing.assert_allclose(result.anchored[0][3:9], original[0][3:9])


def test_the_sample_keeps_the_detection_s_own_anchor_stamp_and_sequence():
    """The rebuilt sample keeps the detection's identity fields."""
    d = donor(slot_with(facing_wall(1.0)).latest)
    result = d.fuse(detection([0.0, 0.0, 0.97], seq=17), 0.0)
    _, anchor, stamp, seq = result.anchored
    np.testing.assert_allclose(anchor, FLAT_ANCHOR)
    assert (stamp, seq) == (0.0, 17)


def test_the_reported_depths_describe_the_move_along_the_optical_axis():
    d = donor(slot_with(facing_wall(1.0)).latest)
    result = d.fuse(detection([0.0, 0.0, 0.96]), 0.0)
    assert result.anchored is not None, result.reason
    assert result.fine_depth_m == pytest.approx(0.96)
    assert result.fused_depth_m == pytest.approx(1.0)
    assert result.shift_mm == pytest.approx(40.0)


def test_the_extrinsic_is_applied_to_the_result():
    """The result is in the TIP frame, so a non-identity extrinsic must show up."""
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.0, 0.0, -0.2]  # tip 20 cm in front of the lens
    d = StereoDepthDonor(
        lambda: extrinsic, slot_with(facing_wall(1.0)).latest, 2.0, 0.06
    )
    # Camera at pelvis z = -0.2 sees the wall (pelvis z = 1.0) at optical
    # z = 1.2; a tip-frame detection at 0.97 must come back as 1.0.
    result = d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert result.anchored is not None, result.reason
    assert result.anchored[0][2] == pytest.approx(1.0)


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_nothing_is_built_before_a_wall_has_been_measured():
    """Without a range, the sample is skipped rather than using detector depth."""
    d = donor(slot_with(None).latest)
    result = d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert result.anchored is None
    assert "no wall plane" in result.reason
    assert d.skipped == 1


def test_a_stale_plane_is_refused():
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest, max_age_s=1.0)
    assert d.fuse(detection([0.0, 0.0, 0.97]), 0.0).anchored is not None
    result = d.fuse(detection([0.0, 0.0, 0.97], seq=2), 5.0)
    assert result.anchored is None
    assert "5.0 s old" in result.reason


def test_the_plane_ages_even_on_ticks_with_no_detection():
    """Otherwise a source looks fresher the more often the detector fires."""
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest, max_age_s=1.0)
    d.fuse(None, 0.0)  # clock starts here, with no detection
    d.fuse(None, 3.0)
    result = d.fuse(detection([0.0, 0.0, 0.97]), 3.0)
    assert result.anchored is None
    assert "3.0 s old" in result.reason


def test_a_fresh_plane_restarts_the_clock():
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest, max_age_s=1.0)
    d.fuse(None, 0.0)
    slot.publish(facing_wall(1.0), frame_id="pelvis")
    result = d.fuse(detection([0.0, 0.0, 0.97]), 5.0)
    assert result.anchored is not None, result.reason


def test_a_disagreement_past_the_shift_limit_builds_nothing():
    """A range disagreement beyond the tolerance is rejected as the far mode."""
    d = donor(slot_with(facing_wall(1.0)).latest, max_shift_m=0.06)
    result = d.fuse(detection([0.0, 0.0, 0.5]), 0.0)
    assert result.anchored is None
    assert "disagree by 500 mm" in result.reason


def test_a_target_essentially_at_the_lens_is_refused():
    """Sign check: a wall a centimetre from the lens is rejected."""
    d = donor(slot_with(facing_wall(0.01)).latest, max_shift_m=10.0)
    result = d.fuse(detection([0.0, 0.0, 0.5]), 0.0)
    assert result.anchored is None
    assert "from the lens" in result.reason


def test_a_ray_pointing_away_from_the_wall_is_refused():
    """An intersection behind the camera is reported distinctly from "no range"."""
    d = donor(slot_with(facing_wall(1.0)).latest, max_shift_m=10.0)
    result = d.fuse(detection([0.0, 0.0, -0.5]), 0.0)
    assert result.anchored is None
    assert "behind the detection's ray" in result.reason


def test_a_ray_parallel_to_the_wall_is_refused_rather_than_matched_on_depth():
    """An edge-on ray cannot be ranged; no optical-axis fallback."""
    d = donor(slot_with(facing_wall(1.0)).latest, max_shift_m=10.0)
    # Almost parallel to a wall whose normal is the optical axis.
    result = d.fuse(detection([1.0, 0.0, 0.05]), 0.0)
    assert result.anchored is None
    assert "edge-on" in result.reason


def test_no_extrinsic_yet_builds_nothing():
    d = StereoDepthDonor(lambda: None, slot_with(facing_wall(1.0)).latest, 2.0, 0.6)
    result = d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert result.anchored is None
    assert "extrinsic" in result.reason


def test_a_detection_at_the_optical_centre_is_refused():
    """It is not a ray: there is no direction to re-range along."""
    d = donor(slot_with(facing_wall(1.0)).latest)
    result = d.fuse(detection([0.0, 0.0, 0.0]), 0.0)
    assert result.anchored is None
    assert "optical centre" in result.reason


def test_no_detection_is_not_a_skip():
    """A tick with nothing to range is not a failure to range something."""
    d = donor(slot_with(facing_wall(1.0)).latest)
    result = d.fuse(None, 0.0)
    assert result.anchored is None and result.reason == ""
    assert d.skipped == 0


# ── Tally and reporting ──────────────────────────────────────────────────────


def test_a_re_offered_detection_is_counted_once():
    """Counts per unique detection, not per control tick."""
    d = donor(slot_with(None).latest)
    for _ in range(5):
        d.fuse(detection([0.0, 0.0, 0.97], seq=3), 0.0)
    assert d.skipped == 1
    d_ok = donor(slot_with(facing_wall(1.0)).latest)
    for _ in range(5):
        d_ok.fuse(detection([0.0, 0.0, 0.97], seq=3), 0.0)
    assert d_ok.fused == 1


def test_donor_count_reports_how_many_distinct_planes_a_batch_rests_on():
    """Donor count is reported separately from the fused count."""
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest)
    for seq in range(1, 5):
        d.fuse(detection([0.0, 0.0, 0.97], seq=seq), 0.0)
    assert d.fused == 4 and d.donor_count == 1
    slot.publish(facing_wall(1.01), frame_id="pelvis")
    d.fuse(detection([0.0, 0.0, 0.97], seq=5), 0.0)
    assert d.fused == 5 and d.donor_count == 2


def test_the_summary_names_the_source_and_the_move_it_made():
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest)
    for seq in range(1, 4):
        d.fuse(detection([0.0, 0.0, 0.98], seq=seq), 0.0)  # +20 mm each
    summary = d.summary()
    assert "measured wall" in summary
    assert "+20.0 mm" in summary


def test_the_summary_says_so_when_nothing_was_built():
    d = donor(slot_with(None).latest)
    d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert "no sample took the measured wall's range" in d.summary()
    assert "1 skipped" in d.summary()


def test_reset_clears_the_batch_but_not_the_plane():
    """A new round resets the tally but keeps the wall."""
    slot = slot_with(facing_wall(1.0))
    d = donor(slot.latest)
    d.fuse(detection([0.0, 0.0, 0.97]), 0.0)
    assert d.fused == 1
    d.reset()
    assert d.fused == 0 and d.donor_count == 0
    rebuilt = d.fuse(detection([0.0, 0.0, 0.97], seq=2), 0.0)
    assert rebuilt.anchored is not None, rebuilt.reason


# ── The slot ─────────────────────────────────────────────────────────────────


def test_the_slot_is_empty_until_a_plane_arrives():
    assert WallPlaneSlot().latest() is None


def test_the_slot_sequence_counts_planes_and_never_resets():
    """So a node restart cannot make a fresh plane look like a repeat."""
    slot = WallPlaneSlot()
    slot.publish(wall(1.0))
    assert slot.latest()[1] == 1
    slot.publish(wall(1.1))
    assert slot.latest()[1] == 2
    slot.reset()
    assert slot.latest() is None
    slot.publish(wall(1.2))
    assert slot.latest()[1] == 3


def test_the_slot_carries_the_frame_for_the_caller_s_log():
    slot = WallPlaneSlot()
    slot.publish(wall(1.0), frame_id="pelvis")
    assert slot.frame_id == "pelvis"
