# MIT License
"""End-to-end test of the head camera calibration: rendered tag, real TF, real fit.

A known ``R hh'`` is injected into TF as the sim bridge does, the tag is rendered
from the true camera pose, and the node must recover the rotation. Covers the
``pupil_apriltags`` tag frame convention, TF lookup direction and composition signs.
"""

from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("pupil_apriltags")

import cv2  # noqa: E402
from g1_common.apriltag import cells_per_side, printable_tag, QUIET_CELLS  # noqa: E402
from g1_sensors.camera.head_camera_calibration import rotation_angle_deg  # noqa: E402
from g1_sensors.camera.head_camera_calibration_node import (  # noqa: E402
    HeadCameraCalibrationNode,
)
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402

# Head d435 intrinsics (camera_specs) and mount (g1_state_publisher urdf).
HEAD_K = [
    606.370361328125, 0.0, 308.32464599609375,
    0.0, 606.370361328125, 246.02682495117188,
    0.0, 0.0, 1.0,
]  # fmt: skip
WIDTH, HEIGHT = 640, 480
MOUNT_XYZ = (0.0576235, 0.01753, 0.42987)
MOUNT_PITCH_RAD = 0.8307767239493009
#: Canonical body -> optical basis published by realsense2_camera.
OPTICAL_QUAT = (-0.5, 0.5, -0.5, 0.5)

CAMERA_FRAME = "d435_color_optical_frame"
MOUNT_FRAME = "d435_link"
MOUNT_PARENT = "torso_link"
WALL_FRAME = "head_calibration_tag"

TAG_ID = 0
TAG_SIZE_M = 0.12
TAG_CENTRE = np.array([3.307, -0.10, 0.82])
STAND_OFF_M = 0.5

#: Torso (yaw, pitch) offsets in degrees. None is head-on: a fronto-parallel
#: tag has an ill-conditioned out-of-plane tilt.
PLACEMENTS = ((-18.0, 3.0), (25.0, -4.0), (7.0, -6.0), (-9.0, 5.0))

SUPERSAMPLE = 2


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def plaque() -> np.ndarray:
    """The printed sheet, at ~4 px per tag cell before the perspective warp."""
    pitch_m = TAG_SIZE_M / cells_per_side()
    image, _ = printable_tag(
        TAG_ID, size_m=TAG_SIZE_M, dpi=int(round(32 * 0.0254 / pitch_m))
    )
    return image


def rotation(*rpy_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()


def optical_basis() -> np.ndarray:
    return Rotation.from_quat(OPTICAL_QUAT).as_matrix()


def torso_pose(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R(world <- torso), p_torso)`` placing the tag at the image centre."""
    r_torso = rotation(0.0, pitch_deg, yaw_deg)
    r_mount = rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0)
    boresight = r_torso @ r_mount @ optical_basis() @ np.array([0.0, 0.0, 1.0])
    p_camera = TAG_CENTRE - STAND_OFF_M * boresight
    return r_torso, p_camera - r_torso @ np.asarray(MOUNT_XYZ)


def true_camera_pose(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the camera *really* is: FK with the unperturbed optical basis."""
    r_torso, p_torso = torso_pose(yaw_deg, pitch_deg)
    r_mount = rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0)
    return (
        r_torso @ r_mount @ optical_basis(),
        p_torso + r_torso @ np.asarray(MOUNT_XYZ),
    )


def render(plaque: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """A mono8 frame of the tag as seen by the true camera.

    Homography warp, supersampled and area-downsampled to keep corner aliasing
    below ~0.1 deg.
    """
    r_camera, p_camera = true_camera_pose(yaw_deg, pitch_deg)
    half = TAG_SIZE_M * (cells_per_side() + 2 * QUIET_CELLS) / cells_per_side() / 2.0
    # Printed-right is -y in world and printed-down is -z, so the sheet's
    # top-left corner is at the largest y and the largest z.
    corners = np.array(
        [
            [TAG_CENTRE[0], TAG_CENTRE[1] + half, TAG_CENTRE[2] + half],
            [TAG_CENTRE[0], TAG_CENTRE[1] - half, TAG_CENTRE[2] + half],
            [TAG_CENTRE[0], TAG_CENTRE[1] - half, TAG_CENTRE[2] - half],
            [TAG_CENTRE[0], TAG_CENTRE[1] + half, TAG_CENTRE[2] - half],
        ]
    )
    in_camera = (corners - p_camera) @ r_camera
    assert np.all(in_camera[:, 2] > 0.0), "the tag is behind the camera"
    pixels = (
        in_camera[:, :2] / in_camera[:, 2:3] * [HEAD_K[0], HEAD_K[4]]
        + [HEAD_K[2], HEAD_K[5]]
    ) * SUPERSAMPLE

    height, width = plaque.shape
    homography = cv2.getPerspectiveTransform(
        np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            np.float32,
        ),
        pixels.astype(np.float32),
    )
    # Mid-grey background so the white quiet zone stays detectable.
    canvas = np.full((HEIGHT * SUPERSAMPLE, WIDTH * SUPERSAMPLE), 130, np.uint8)
    cv2.warpPerspective(
        plaque,
        homography,
        (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE),
        dst=canvas,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    return cv2.resize(canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _transform(parent, child, rot, translation, *, sec=0) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = parent
    tf.header.stamp.sec = sec
    tf.child_frame_id = child
    (
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z,
    ) = (float(v) for v in translation)
    quat = Rotation.from_matrix(rot).as_quat()
    (
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    ) = (float(v) for v in quat)
    return tf


def seed_static_tf(node, perturbation: np.ndarray, translation=None) -> None:
    """Static frames the node resolves; the optical edge carries ``R0 @ P``.

    Matches ros_bridge under G1_HEAD_OPTICAL_PERTURBATION_RPY_DEG. The renderer
    uses ``R0``, so TF is off by exactly ``P``.
    """
    node._tf_buffer.set_transform_static(
        _transform(
            MOUNT_PARENT,
            MOUNT_FRAME,
            rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0),
            MOUNT_XYZ,
        ),
        "test",
    )
    node._tf_buffer.set_transform_static(
        _transform(
            MOUNT_FRAME,
            CAMERA_FRAME,
            optical_basis() @ perturbation,
            optical_basis()
            @ np.asarray(np.zeros(3) if translation is None else translation),
        ),
        "test",
    )
    node._tf_buffer.set_transform_static(
        _transform("world", WALL_FRAME, np.eye(3), TAG_CENTRE), "test"
    )


def camera_info() -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = CAMERA_FRAME
    msg.width, msg.height = WIDTH, HEIGHT
    msg.k = HEAD_K
    msg.d = [0.0] * 5
    return msg


def image_message(sec: int = 0) -> Image:
    msg = Image()
    msg.header.frame_id = CAMERA_FRAME
    msg.header.stamp.sec = sec
    return msg


@pytest.fixture
def node():
    """The node with its rate limiter and settle window disabled."""
    n = HeadCameraCalibrationNode(
        parameter_overrides=[
            Parameter("wall_frame", value=WALL_FRAME),
            Parameter("tag_id", value=TAG_ID),
            Parameter("tag_size_m", value=TAG_SIZE_M),
            Parameter("accumulate_rate_hz", value=1.0e6),
            Parameter("settle_s", value=0.0),
            Parameter("output_path", value=""),
        ]
    )
    assert n._hand_image_topic == "/camera/hand/color/image_raw/compressed"
    n._camera_info_callback(camera_info())
    yield n
    n.destroy_node()


def collect(node, plaque, torso_sec: int | None = None) -> None:
    """Run every placement past the node the way an operator would.

    ``torso_sec`` stamps ``world -> torso_link``; None uses each image's stamp,
    a fixed value simulates a clock-domain mismatch.
    """
    for index, (yaw, pitch) in enumerate(PLACEMENTS):
        stamp = index + 1
        r_torso, p_torso = torso_pose(yaw, pitch)
        node._tf_buffer.set_transform(
            _transform(
                "world",
                MOUNT_PARENT,
                r_torso,
                p_torso,
                sec=stamp if torso_sec is None else torso_sec,
            ),
            "test",
        )
        node._command("s")
        for _ in range(3):
            node._image_callback(render(plaque, yaw, pitch), image_message(stamp))
        node._command("t")


# ── The whole path ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "injected_rpy_deg",
    [(0.0, 0.0, 0.0), (1.5, -3.0, 2.0), (-4.0, 0.0, 0.0), (0.0, 0.0, 6.0)],
)
def test_an_arbitrary_injected_rotation_comes_back_out(node, plaque, injected_rpy_deg):
    """Set ``R hh'``, retrieve ``R hh'``.

    Identity catches a spurious constant rotation; per-axis cases catch sign errors.
    """
    injected = rotation(*injected_rpy_deg)
    seed_static_tf(node, injected)
    collect(node, plaque)

    solution = node._session.solve()
    assert solution.samples == 3 * len(PLACEMENTS)
    # 0.3 deg is the renderer's corner accuracy at this tag size and range.
    assert rotation_angle_deg(solution.rotation, injected) < 0.3
    assert solution.mean_residual_deg < 0.3
    # Every sample used the exact-stamp TF lookup, not the latest-TF fallback.
    assert node._latest_tf_samples == 0
    assert node._warned_latest_tf is False


def test_an_injected_translation_comes_back_from_the_same_images(node, plaque):
    injected_rotation = rotation(1.5, -3.0, 2.0)
    injected_translation = np.array([0.012, -0.007, 0.021])
    seed_static_tf(node, injected_rotation, injected_translation)
    collect(node, plaque)

    solution = node._session.solve()
    assert solution.translation == pytest.approx(injected_translation, abs=0.005)
    assert solution.mean_translation_residual_m < 0.005


def test_the_report_names_the_edits_and_scores_the_injection(node, plaque, tmp_path):
    """The report's URDF and sim-bridge corrections undo the injection."""
    injected = rotation(1.5, -3.0, 2.0)
    node._expected_rpy_deg = [1.5, -3.0, 2.0]
    node._output_path = str(tmp_path / "report.yaml")
    seed_static_tf(node, injected)
    collect(node, plaque)

    node._command("e")
    assert node.finished.done() and node.finished.result() is True

    import yaml

    report = yaml.safe_load((tmp_path / "report.yaml").read_text())
    assert "depth" not in yaml.safe_dump(report).lower()
    assert report["vs_expected"]["error_deg"] < 0.3
    assert len(report["placements"]) == len(PLACEMENTS)
    assert report["t_hh_prime_m"]["xyz"] == pytest.approx([0.0, 0.0, 0.0], abs=0.005)

    # Sim edit lands back on the unperturbed optical basis.
    corrected = Rotation.from_quat(
        report["corrections"]["sim_optical_frame_quat_xyzw"]
    ).as_matrix()
    assert rotation_angle_deg(corrected, optical_basis()) < 0.3
    assert report["corrections"]["sim_optical_frame_xyz"] == pytest.approx(
        [0.0, 0.0, 0.0], abs=0.005
    )
    # URDF edit yields the same camera pose (correction conjugated into mount axes).
    joint = Rotation.from_euler(
        "xyz", report["corrections"]["urdf_mount_joint"]["rpy"]
    ).as_matrix()
    assert report["corrections"]["urdf_mount_joint"]["xyz"] == pytest.approx(
        MOUNT_XYZ, abs=0.005
    )
    assert (
        rotation_angle_deg(
            joint @ optical_basis() @ injected,
            rotation(0.0, np.degrees(MOUNT_PITCH_RAD), 0.0) @ optical_basis(),
        )
        < 0.3
    )


def test_a_clock_mismatch_still_calibrates(node, plaque):
    """Falls back to the latest transform, and warns, when image and TF clocks
    differ."""
    injected = rotation(1.5, -3.0, 2.0)
    seed_static_tf(node, injected)
    collect(node, plaque, torso_sec=1000)  # while images are stamped t = 1..4

    solution = node._session.solve()
    assert solution.samples == 3 * len(PLACEMENTS)
    assert node._warned_latest_tf is True
    assert node._latest_tf_samples == solution.samples
    assert rotation_angle_deg(solution.rotation, injected) < 0.3


def test_a_frame_with_no_tag_is_counted_and_not_folded_in(node, plaque):
    """A blank frame must be a rejection, not a sample of nothing."""
    seed_static_tf(node, np.eye(3))
    r_torso, p_torso = torso_pose(*PLACEMENTS[0])
    node._tf_buffer.set_transform(
        _transform("world", MOUNT_PARENT, r_torso, p_torso), "test"
    )
    node._command("s")
    node._image_callback(np.full((HEIGHT, WIDTH), 130, np.uint8), image_message())
    assert len(node._session.total) == 0
    assert node._no_tag == 1


def test_standoff_samples_compare_the_same_normal_axis(node):
    """Head, hand and mocap use point-to-plane distance, never centre L2."""
    hand_frame = node._hand_camera_frame
    node._tf_buffer.set_transform_static(
        _transform(
            CAMERA_FRAME,
            hand_frame,
            np.eye(3),
            [0.0, 0.0, 0.0],
        ),
        "test",
    )
    node._tf_buffer.set_transform_static(
        _transform(hand_frame, "drill_tip_link", np.eye(3), [0.8, 0.2, -0.1]),
        "test",
    )
    # camera <- wall is identity; wall at x=1.0 with outward normal -x.
    detection = SimpleNamespace(
        pose_R=node._wall_from_tag.copy(), pose_t=np.array([1.0, 0.0, 0.0])
    )
    node._session.start()
    assert node._record_hand_standoff(detection, image_message().header.stamp)
    node._record_standoff_sample(
        detection, np.eye(3), np.array([1.0, 0.0, 0.0]), image_message().header.stamp
    )

    sample = node._standoff_samples[-1]
    assert not any("depth" in key for key in sample)
    assert sample["tag_standoff_m"] == pytest.approx(0.2)
    assert sample["mocap_standoff_m"] == pytest.approx(0.2)
    assert sample["hand_tag_standoff_m"] == pytest.approx(0.2)
    assert sample["hand_minus_head_tag_m"] == pytest.approx(0.0)
    assert sample["hand_minus_mocap_m"] == pytest.approx(0.0)
    assert node._hand_standoff_samples[-1]["hand_tag_standoff_m"] == pytest.approx(0.2)


def test_ending_with_nothing_collected_reports_failure(node):
    """'e' with no samples reports failure without crashing."""
    node._command("e")
    assert node.finished.done() and node.finished.result() is False


def test_commands_after_the_report_are_ignored(node, plaque):
    """'e' may arrive from both terminal and topic; the second must not re-set the
    Future."""
    seed_static_tf(node, np.eye(3))
    collect(node, plaque)
    node._command("e")
    assert node.finished.result() is True

    node._command("e")
    node._command("s")
    assert not node._session.accumulating
