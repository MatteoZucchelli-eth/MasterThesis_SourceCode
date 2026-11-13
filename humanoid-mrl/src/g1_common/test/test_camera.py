# MIT License
"""Unit tests for g1_common.camera — the sim/robot frame preparation."""

import numpy as np
import pytest

from g1_common.camera import (
    build_frame_preparer,
    centre_crop_origin,
    CentreCrop,
    PinholeIntrinsics,
    preparer_from_camera_info,
    UndistortedCentreCrop,
)

# The calibrated See3CAM_24CUG on the drill mount
# (g1_cam_drill_interface/config/hand_camera_info.yaml).
WRIST_K = [
    690.1127251872208, 0.0, 639.8322628786954,
    0.0, 689.9782285763009, 369.7137298773904,
    0.0, 0.0, 1.0,
]  # fmt: skip
WRIST_D = [
    -0.348378440666311,
    0.1540764129132001,
    0.0002730650407718539,
    0.0002612332722158002,
    -0.03625109242875615,
]
SIDE = 720


def wrist_source() -> PinholeIntrinsics:
    return PinholeIntrinsics.from_k(WRIST_K, 1280, 720)


def test_crop_moves_principal_point_but_not_focal_length():
    """A crop removes pixels; it does not rescale them."""
    cropped = wrist_source().cropped(SIDE, SIDE)
    assert cropped.fx == pytest.approx(690.1127251872208)
    assert cropped.fy == pytest.approx(689.9782285763009)
    # 280 columns come off each side; no rows come off at all, so cy is untouched.
    assert cropped.cx == pytest.approx(639.8322628786954 - 280.0)
    assert cropped.cy == pytest.approx(369.7137298773904)
    assert (cropped.width, cropped.height) == (SIDE, SIDE)


def test_centre_crop_origin_is_centred():
    assert centre_crop_origin(1280, 720, 720) == (280, 0)
    assert centre_crop_origin(640, 480, 480) == (80, 0)


def test_zero_distortion_selects_the_non_resampling_implementation():
    """Isaac renders an ideal pinhole, so it must not pay for a remap."""
    preparer = build_frame_preparer(wrist_source(), [0.0] * 5, SIDE)
    assert isinstance(preparer, CentreCrop)


def test_real_distortion_selects_the_remap():
    preparer = build_frame_preparer(wrist_source(), WRIST_D, SIDE)
    assert isinstance(preparer, UndistortedCentreCrop)


def test_crop_only_path_is_exactly_a_slice():
    image = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    preparer = build_frame_preparer(wrist_source(), [0.0] * 5, SIDE)
    assert np.array_equal(preparer(image), image[:, 280:1000])


def test_single_pass_remap_equals_undistort_then_crop():
    """The fused map must be the composition it claims to be, not an approximation."""
    cv2 = pytest.importorskip("cv2")
    source = wrist_source()
    image = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    map_x, map_y = cv2.initUndistortRectifyMap(
        source.matrix, np.asarray(WRIST_D), None, source.matrix, (1280, 720),
        cv2.CV_16SC2,
    )  # fmt: skip
    reference = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    reference = reference[:, 280:1000]

    assert np.array_equal(build_frame_preparer(source, WRIST_D, SIDE)(image), reference)


def test_preparer_reports_the_intrinsics_of_its_own_output():
    """Prepared pixels and reported intrinsics agree."""
    for distortion in ([0.0] * 5, WRIST_D):
        preparer = build_frame_preparer(wrist_source(), distortion, SIDE)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = preparer(image)
        assert out.shape[:2] == (preparer.intrinsics.height, preparer.intrinsics.width)
        assert preparer.intrinsics.cx == pytest.approx(359.8322628786954)


class FakeCameraInfo:
    """The four CameraInfo fields the preparer reads, and nothing else."""

    def __init__(self, k, d, width, height, distortion_model):
        self.k, self.d = k, d
        self.width, self.height = width, height
        self.distortion_model = distortion_model


def test_declared_model_travels_from_camera_info_to_the_preparer():
    """The stream's own word on its lens is what decides, not a default."""
    pytest.importorskip("cv2")
    msg = FakeCameraInfo(WRIST_K, WRIST_D, 1280, 720, "plumb_bob")
    assert isinstance(preparer_from_camera_info(msg, SIDE), UndistortedCentreCrop)


def test_unsupported_model_is_refused_rather_than_mis_rectified():
    """Fisheye coefficients through the Brown-Conrady remap look fine and are wrong."""
    msg = FakeCameraInfo(WRIST_K, [-0.02, 0.003, -0.001, 0.0], 1280, 720, "equidistant")
    with pytest.raises(ValueError, match="equidistant"):
        preparer_from_camera_info(msg, SIDE)


def test_unsupported_model_without_distortion_is_still_a_plain_crop():
    """An ideal pinhole is one whatever model name the driver stamped on it."""
    msg = FakeCameraInfo(WRIST_K, [0.0] * 4, 1280, 720, "equidistant")
    assert isinstance(preparer_from_camera_info(msg, SIDE), CentreCrop)


def test_coefficient_count_must_match_the_declared_model():
    """A stream contradicting itself has no geometry worth trusting."""
    with pytest.raises(ValueError, match="coefficients"):
        build_frame_preparer(wrist_source(), WRIST_D + [0.1], SIDE, "plumb_bob")


def test_rational_polynomial_is_accepted_at_both_lengths():
    pytest.importorskip("cv2")
    for distortion in (WRIST_D, WRIST_D + [0.01, 0.002, 0.0003]):
        preparer = build_frame_preparer(
            wrist_source(), distortion, SIDE, "rational_polynomial"
        )
        assert isinstance(preparer, UndistortedCentreCrop)


def test_sim_and_robot_agree_once_isaac_renders_the_sensor_shape():
    """Isaac's 1280x720 render with the See3CAM focal length matches the real prepared
    geometry."""
    real = build_frame_preparer(wrist_source(), WRIST_D, SIDE).intrinsics
    sim_source = PinholeIntrinsics.from_k(
        [690.1127, 0.0, 640.0, 0.0, 689.9782, 360.0, 0.0, 0.0, 1.0], 1280, 720
    )
    sim = build_frame_preparer(sim_source, [0.0] * 5, SIDE).intrinsics

    assert sim.fx == pytest.approx(real.fx, rel=1e-6)
    assert sim.fy == pytest.approx(real.fy, rel=1e-6)
    # Principal points differ only by the real lens's calibrated off-centring.
    assert abs(sim.cx - real.cx) < 1.0
    assert abs(sim.cy - real.cy) < 10.0
