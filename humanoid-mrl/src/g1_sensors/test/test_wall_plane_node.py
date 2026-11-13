# MIT License
"""End-to-end test of the wall plane node with real messages, TF and fit.

Covers depth encodings, row padding, TF lookup and output frames. Skips when
rclpy or the WallPlane message is unavailable.
"""

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("g1_interfaces.msg")

from g1_sensors.depth.wall_plane_node import WallPlaneNode  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from sensor_msgs_py import point_cloud2  # noqa: E402

WALL_X_M = 1.2
PITCH_DEG = 47.6  # the head camera's pitch in the sim scene
CAMERA_FRAME = "d435_color_optical_frame"
HEAD_K = [
    606.370361328125, 0.0, 308.32464599609375,
    0.0, 606.370361328125, 246.02682495117188,
    0.0, 0.0, 1.0,
]  # fmt: skip


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def head_rotation() -> np.ndarray:
    """R(pelvis <- d435_color_optical_frame) for the pitched-down head mount."""
    pitch = np.deg2rad(PITCH_DEG)
    body = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pitch_down = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    return pitch_down @ body


@pytest.fixture
def node():
    """A node whose camera transform is static, so any stamp resolves.

    Static lookups ignore the stamp; use ``dynamic_node`` for stamping tests.
    """
    n = WallPlaneNode()
    n._tf_buffer.set_transform_static(
        _transform("pelvis", CAMERA_FRAME, head_rotation(), [0.0, 0.0, 1.202]),
        "test",
    )
    yield n
    n.destroy_node()


@pytest.fixture
def dynamic_node():
    """A node whose camera transform is dynamic and stamped at t = 1000.

    Mirrors the real stack: TF on the system clock, images on another clock.
    """
    n = WallPlaneNode()
    tf = _transform("pelvis", CAMERA_FRAME, head_rotation(), [0.0, 0.0, 1.202])
    tf.header.stamp.sec = 1000
    n._tf_buffer.set_transform(tf, "test")
    yield n
    n.destroy_node()


def _transform(parent, child, rot, translation) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(translation[0])
    tf.transform.translation.y = float(translation[1])
    tf.transform.translation.z = float(translation[2])
    w = float(np.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0)
    tf.transform.rotation.w = w
    tf.transform.rotation.x = float((rot[2, 1] - rot[1, 2]) / (4.0 * w))
    tf.transform.rotation.y = float((rot[0, 2] - rot[2, 0]) / (4.0 * w))
    tf.transform.rotation.z = float((rot[1, 0] - rot[0, 1]) / (4.0 * w))
    return tf


def camera_info() -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = CAMERA_FRAME
    msg.width, msg.height = 640, 480
    msg.k = HEAD_K
    return msg


def wall_depth() -> np.ndarray:
    """Depth image (metres) of a vertical wall at pelvis x = WALL_X_M."""
    rot = head_rotation()
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    rays = np.column_stack(
        (
            ((u - HEAD_K[2]) / HEAD_K[0]).ravel(),
            ((v - HEAD_K[5]) / HEAD_K[4]).ravel(),
            np.ones(u.size),
        )
    )
    in_pelvis = rays @ rot.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (WALL_X_M - 0.0) / in_pelvis[:, 0]
    return np.where((t > 0) & np.isfinite(t), t, 0.0).reshape(480, 640)


def depth_message(depth_m: np.ndarray, *, encoding="32FC1", pad=0) -> Image:
    msg = Image()
    msg.header.frame_id = CAMERA_FRAME
    msg.height, msg.width = depth_m.shape
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "32FC1":
        raw = depth_m.astype(np.float32)
    else:
        raw = (depth_m * 1000.0).astype(np.uint16)
    if pad:
        raw = np.hstack((raw, np.zeros((msg.height, pad), dtype=raw.dtype)))
    msg.step = raw.shape[1] * raw.dtype.itemsize
    msg.data = raw.tobytes()
    return msg


def capture(node):
    """Swap the node's publishers for recorders and return what they saw."""
    seen = {}
    for attr, key in (
        ("_plane_pub", "plane"),
        ("_cloud_pub", "cloud"),
        ("_marker_pub", "marker"),
    ):
        setattr(
            node,
            attr,
            type("P", (), {"publish": lambda _s, m, k=key: seen.__setitem__(k, m)})(),
        )
    return seen


# ── The whole path ───────────────────────────────────────────────────────────


def test_output_carries_the_transforms_stamp_not_the_images(dynamic_node):
    """The plane carries the transform's stamp, not the image's.

    Sim images are stamped from the Isaac timeline (t = 60) and TF from the
    system clock (t = 1000); only the TF stamp is resolvable in RViz.
    """
    seen = capture(dynamic_node)
    dynamic_node._camera_info_callback(camera_info())
    msg = depth_message(wall_depth())
    msg.header.stamp.sec = 60
    dynamic_node._depth_callback(msg)

    assert seen["plane"].point.x == pytest.approx(WALL_X_M, abs=1e-3)
    for key in ("plane", "cloud", "marker"):
        assert seen[key].header.stamp.sec == 1000, key


def test_a_clock_mismatch_still_produces_a_plane(dynamic_node):
    """An unresolvable image stamp falls back to the latest transform."""
    seen = capture(dynamic_node)
    dynamic_node._camera_info_callback(camera_info())
    msg = depth_message(wall_depth())
    msg.header.stamp.sec = 60  # unresolvable against a cache holding t = 1000
    dynamic_node._depth_callback(msg)

    assert "plane" in seen
    assert dynamic_node._failures == 0
    assert dynamic_node._warned_latest_tf is True


def test_all_three_publications_share_one_stamp_and_frame(node):
    """All outputs of one measurement share one stamp."""
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth()))

    headers = [seen[k].header for k in ("plane", "cloud", "marker")]
    assert len({(h.stamp.sec, h.stamp.nanosec) for h in headers}) == 1
    assert {h.frame_id for h in headers} == {"pelvis"}


def test_a_depth_frame_becomes_a_plane_in_the_pelvis_frame(node):
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth()))

    plane = seen["plane"]
    assert plane.header.frame_id == "pelvis"
    assert plane.point.x == pytest.approx(WALL_X_M, abs=1e-3)
    # Outward normal: the camera is at x < WALL_X_M, so out of the wall is -x.
    assert plane.normal.x == pytest.approx(-1.0, abs=1e-3)
    assert plane.inliers > 1000
    assert plane.sampled >= plane.inliers
    assert plane.rms_m < 1e-3


def test_the_cloud_comes_out_in_the_same_frame_and_lands_on_the_wall(node):
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth()))

    cloud = seen["cloud"]
    assert cloud.header.frame_id == "pelvis"
    points = (
        np.array(
            list(
                point_cloud2.read_points(cloud, ["x", "y", "z"], skip_nans=True)
            ).copy()
        )
        .view(np.float32)
        .reshape(-1, 3)
    )
    assert len(points) > 1000
    np.testing.assert_allclose(points[:, 0], WALL_X_M, atol=1e-3)


def test_millimetre_depth_gives_the_same_wall_as_metre_depth(node):
    """Both encodings: realsense2_camera 16UC1 millimetres, Isaac 32FC1 metres."""
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth(), encoding="16UC1"))
    assert seen["plane"].point.x == pytest.approx(WALL_X_M, abs=2e-3)


def test_a_padded_row_stride_does_not_shear_the_image(node):
    """Row step is honoured; padded rows read unpadded would shear the frame."""
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth(), pad=7))
    assert seen["plane"].point.x == pytest.approx(WALL_X_M, abs=1e-3)


def test_the_marker_pose_lies_on_the_measured_plane(node):
    """The marker pose agrees with the fitted plane."""
    seen = capture(node)
    node._camera_info_callback(camera_info())
    node._depth_callback(depth_message(wall_depth()))

    marker, plane = seen["marker"], seen["plane"]
    assert marker.header.frame_id == "pelvis"
    assert (marker.pose.position.x, marker.pose.position.y) == (
        pytest.approx(plane.point.x),
        pytest.approx(plane.point.y),
    )
    # A CYLINDER's axis is local +z; rotation must map +z onto the normal.
    q = marker.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    axis = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    normal = np.array([plane.normal.x, plane.normal.y, plane.normal.z])
    np.testing.assert_allclose(axis, normal, atol=1e-6)


# ── Refusals, through the real callbacks ─────────────────────────────────────


def test_nothing_is_published_before_camera_info_arrives(node):
    """Intrinsics must come from the image's own CameraInfo."""
    seen = capture(node)
    node._depth_callback(depth_message(wall_depth()))
    assert seen == {}
    assert node._failures == 1
    assert "CameraInfo" in node._last_reason


def test_nothing_is_published_without_a_transform(node):
    seen = capture(node)
    node._camera_info_callback(camera_info())
    msg = depth_message(wall_depth())
    msg.header.frame_id = "some_other_camera"
    node._depth_callback(msg)
    assert "plane" not in seen
    assert "TF" in node._last_reason


def test_an_unknown_encoding_is_refused_rather_than_misread(node):
    seen = capture(node)
    node._camera_info_callback(camera_info())
    msg = depth_message(wall_depth())
    msg.encoding = "mono16"
    node._depth_callback(msg)
    assert "plane" not in seen
    assert "unsupported depth encoding" in node._last_reason


def test_a_big_endian_frame_is_refused(node):
    seen = capture(node)
    node._camera_info_callback(camera_info())
    msg = depth_message(wall_depth())
    msg.is_bigendian = 1
    node._depth_callback(msg)
    assert "plane" not in seen
    assert "big-endian" in node._last_reason


def test_intrinsics_describing_a_different_size_are_refused(node):
    """The pair only describes one image if they came from the same place."""
    seen = capture(node)
    info = camera_info()
    info.width, info.height = 320, 240
    node._camera_info_callback(info)
    node._depth_callback(depth_message(wall_depth()))
    assert "plane" not in seen
    assert "does not describe one image" in node._last_reason


def test_the_cloud_is_still_published_when_the_fit_fails(node):
    """Empty cloud and no-plane-found are reported distinctly."""
    seen = capture(node)
    node._camera_info_callback(camera_info())
    # A wall the verticality prior must reject: a floor filling the frame.
    node._depth_callback(depth_message(np.full((480, 640), 2.0)))
    assert "plane" not in seen
    assert seen["cloud"].width > 1000
    assert node._failures == 1
