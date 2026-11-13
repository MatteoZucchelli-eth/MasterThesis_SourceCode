# MIT License
"""Unit tests for the head camera calibration — the estimate, without ROS.

Each test injects a known ``R hh'`` through the node's algebra and demands it back.
"""

import numpy as np
import pytest

from g1_sensors.camera.head_camera_calibration import (
    CalibrationSession,
    corrected_mount_rotation,
    corrected_mount_translation,
    corrected_optical_rotation,
    corrected_optical_translation,
    measured_wall_rotation,
    normal_angle_deg,
    orthonormalise,
    rotation_angle_deg,
    RotationAccumulator,
    signed_plane_standoff,
    WALL_FROM_TAG,
)
from scipy.spatial.transform import Rotation

#: The canonical body -> optical change of basis, as realsense2_camera publishes
#: it for d435_link -> d435_color_optical_frame and as the sim bridge mirrors it.
OPTICAL_FROM_BODY_QUAT = (-0.5, 0.5, -0.5, 0.5)

#: The head mount as the URDF has it: torso_link -> d435_link, pitched down.
MOUNT_PITCH_RAD = 0.8307767239493009


def rotation(*rpy_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()


def torso_orientations() -> list[np.ndarray]:
    """``R hw`` transposed for a few plausible torso poses in front of the wall.

    Uses the real mount pitch so a transposition is not hidden near identity.
    """
    body_from_wall = [rotation(0.0, 0.0, yaw) for yaw in (0.0, -18.0, 25.0, 7.0)]
    optical_from_body = Rotation.from_quat(OPTICAL_FROM_BODY_QUAT).as_matrix()
    pitch = rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0)
    return [optical_from_body.T @ pitch.T @ r for r in body_from_wall]


def test_wall_from_tag_is_a_rotation_and_maps_the_documented_axes():
    """The tag-to-body hop is the only hand-written matrix; checked axis by axis."""
    assert np.linalg.det(WALL_FROM_TAG) == pytest.approx(1.0)
    assert WALL_FROM_TAG.T @ WALL_FROM_TAG == pytest.approx(np.eye(3))
    # tag +z (out of the face, away from the camera) -> wall +x (into the wall)
    assert WALL_FROM_TAG @ [0, 0, 1] == pytest.approx([1, 0, 0])
    # tag +y (image-down) -> wall -z (down)
    assert WALL_FROM_TAG @ [0, 1, 0] == pytest.approx([0, 0, -1])
    # tag +x (image-right) -> wall -y
    assert WALL_FROM_TAG @ [1, 0, 0] == pytest.approx([0, -1, 0])


def test_standoff_is_perpendicular_signed_and_ignores_lateral_motion():
    plane_point = np.array([1.0, 4.0, -2.0])
    outward = np.array([-2.0, 0.0, 0.0])  # Deliberately not unit length.

    assert signed_plane_standoff(
        [0.8, 100.0, -50.0], plane_point, outward
    ) == pytest.approx(0.2)
    assert signed_plane_standoff(
        [1.005, -100.0, 50.0], plane_point, outward
    ) == pytest.approx(-0.005)


def test_normal_angle_uses_directed_outward_normals():
    assert normal_angle_deg([1, 0, 0], [2, 0, 0]) == pytest.approx(0.0)
    assert normal_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert normal_angle_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)


def test_zero_plane_normals_are_rejected():
    with pytest.raises(ValueError, match="zero normal"):
        signed_plane_standoff([0, 0, 0], [1, 0, 0], [0, 0, 0])


def test_orthonormalise_refuses_to_return_a_reflection():
    """``U Vt`` alone can project onto a reflection; the determinant term prevents
    it."""
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(orthonormalise(reflection)) == pytest.approx(1.0)


def test_one_placement_recovers_the_injected_rotation_exactly():
    """A full-rotation observation determines all three axes from one pose."""
    injected = rotation(1.5, -3.0, 2.0)
    accumulator = RotationAccumulator()
    predicted = torso_orientations()[0]
    for _ in range(5):
        accumulator.add(injected @ predicted, predicted)
    solution = accumulator.solve()
    assert rotation_angle_deg(solution.rotation, injected) < 1e-9
    assert solution.mean_residual_deg < 1e-9
    assert solution.consistency == pytest.approx(1.0)


def test_the_full_chain_recovers_the_injected_rotation_from_tag_poses():
    """End to end through the node's algebra, tag frame included.

    Synthesises detector output so a wrong ``WALL_FROM_TAG`` fails here.
    """
    injected = rotation(-0.8, 2.4, 1.1)
    session = CalibrationSession()
    for predicted in torso_orientations():
        session.start()
        # What the camera would report: the wall body seen from the *real*
        # camera, expressed back in the tag's own frame.
        camera_from_tag = injected @ predicted @ WALL_FROM_TAG
        for _ in range(10):
            session.add(measured_wall_rotation(camera_from_tag), predicted)
        session.pause()

    solution = session.solve()
    assert solution.samples == 40
    assert len(session.placement_solutions()) == 4
    assert rotation_angle_deg(solution.rotation, injected) < 1e-9
    assert session.spread_deg() < 1e-9


def test_the_stacked_system_recovers_the_injected_translation():
    """Recover ``t hh'`` from the same real/URDF head-to-wall observations."""
    correction_rotation = rotation(-0.8, 2.4, 1.1)
    correction_translation = np.array([0.012, -0.007, 0.021])
    session = CalibrationSession()
    session.start()

    for index, predicted_rotation in enumerate(torso_orientations()):
        predicted_translation = np.array([0.4 + index, -0.2, 1.3])
        measured_rotation = correction_rotation @ predicted_rotation
        measured_translation = (
            correction_rotation @ predicted_translation + correction_translation
        )
        session.add(
            measured_rotation,
            predicted_rotation,
            measured_translation,
            predicted_translation,
        )

    solution = session.solve()
    assert solution.translation == pytest.approx(correction_translation)
    assert rotation_angle_deg(solution.rotation, correction_rotation) < 1e-9
    assert solution.mean_translation_residual_m < 1e-12
    assert solution.max_translation_residual_m < 1e-12


def test_detector_noise_averages_down_and_is_reported():
    """Per-sample noise averages down and shows up as a non-zero residual."""
    rng = np.random.default_rng(7)
    injected = rotation(1.0, -2.0, 0.5)
    session = CalibrationSession()
    for predicted in torso_orientations():
        session.start()
        for _ in range(200):
            noise = Rotation.from_rotvec(np.deg2rad(rng.normal(0.0, 1.0, 3)))
            session.add(noise.as_matrix() @ injected @ predicted, predicted)
        session.pause()

    solution = session.solve()
    assert rotation_angle_deg(solution.rotation, injected) < 0.1
    assert 0.5 < solution.mean_residual_deg < 3.0
    assert solution.max_residual_deg > solution.mean_residual_deg


def test_a_biased_placement_shows_up_as_spread_and_not_as_residual():
    """A per-placement bias shows up in placement spread, not in the residual."""
    injected = rotation(1.0, 0.0, 0.0)
    bias = rotation(0.0, 0.0, 2.0)
    session = CalibrationSession()
    for index, predicted in enumerate(torso_orientations()):
        session.start()
        offset = bias if index == 0 else np.eye(3)
        for _ in range(50):
            session.add(offset @ injected @ predicted, predicted)
        session.pause()

    solution = session.solve()
    assert session.spread_deg() > 1.9
    assert solution.mean_residual_deg < session.spread_deg()


def test_samples_outside_a_placement_are_dropped():
    """Samples between placements (robot moving) are not recorded."""
    session = CalibrationSession()
    predicted = torso_orientations()[0]
    assert session.add(predicted, predicted) is None
    session.start()
    assert session.add(predicted, predicted) is not None
    session.pause()
    assert session.add(predicted, predicted) is None
    assert len(session.total) == 1


def test_an_empty_placement_is_not_recorded():
    """A segment in which the tag was never seen is not a placement."""
    session = CalibrationSession()
    session.start()
    session.pause()
    assert session.placements == []
    assert session.solve() is None


def test_the_optical_correction_undoes_a_deliberate_perturbation():
    """Perturb the optical TF edge by ``P``; the recovered correction must undo it."""
    truth = Rotation.from_quat(OPTICAL_FROM_BODY_QUAT).as_matrix()
    perturbation = rotation(1.5, -3.0, 2.0)
    published = truth @ perturbation

    # What the calibration recovers from a TF tree carrying `published`.
    predicted = published.T @ rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0).T
    measured = truth.T @ rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0).T
    accumulator = RotationAccumulator()
    accumulator.add(measured, predicted)
    recovered = accumulator.solve().rotation
    assert rotation_angle_deg(recovered, perturbation) < 1e-9

    assert corrected_optical_rotation(published, recovered) == pytest.approx(truth)


def test_the_mount_correction_lands_the_camera_in_the_same_place():
    """The optical and mount edits must be the same correction.

    The mount edit is the optical correction conjugated by the optical hop.
    """
    optical = Rotation.from_quat(OPTICAL_FROM_BODY_QUAT).as_matrix()
    mount = rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0)
    correction = rotation(1.5, -3.0, 2.0)

    via_optical = mount @ corrected_optical_rotation(optical, correction)
    via_mount = corrected_mount_rotation(mount, optical, correction) @ optical
    assert via_mount == pytest.approx(via_optical)


def test_translation_corrections_land_the_camera_in_the_same_place():
    """Both reported edits undo a full six-dimensional camera perturbation."""
    optical_rotation = Rotation.from_quat(OPTICAL_FROM_BODY_QUAT).as_matrix()
    optical_translation = np.array([0.004, -0.002, 0.006])
    mount_rotation = rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0)
    mount_translation = np.array([0.0576235, 0.01753, 0.42987])
    correction_rotation = rotation(1.5, -3.0, 2.0)
    correction_translation = np.array([0.012, -0.007, 0.021])

    corrected_optical_rotation_value = corrected_optical_rotation(
        optical_rotation, correction_rotation
    )
    corrected_optical_translation_value = corrected_optical_translation(
        optical_rotation,
        optical_translation,
        correction_rotation,
        correction_translation,
    )
    corrected_mount_rotation_value = corrected_mount_rotation(
        mount_rotation, optical_rotation, correction_rotation
    )
    corrected_mount_translation_value = corrected_mount_translation(
        mount_rotation,
        mount_translation,
        optical_rotation,
        optical_translation,
        correction_rotation,
        correction_translation,
    )

    via_optical = (
        mount_translation + mount_rotation @ corrected_optical_translation_value
    )
    via_mount = (
        corrected_mount_translation_value
        + corrected_mount_rotation_value @ optical_translation
    )
    assert via_mount == pytest.approx(via_optical)
    assert corrected_mount_rotation_value @ optical_rotation == pytest.approx(
        mount_rotation @ corrected_optical_rotation_value
    )
