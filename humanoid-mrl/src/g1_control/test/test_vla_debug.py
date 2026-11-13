# MIT License
"""Unit tests for the VLA debug publisher's detection routing (no ROS node)."""

import threading

import numpy as np
import pytest

from g1_control import topics

from g1_control.states.vla_debug import VlaDebugPublisher
from g1_control.vla.perception import PerceptionDetection


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _FakeNode:
    """Records one publisher per topic, so a test can read what was sent."""

    def __init__(self) -> None:
        self.pubs: dict[str, _FakePublisher] = {}

    def create_publisher(self, msg_type, topic, depth):  # noqa: ARG002
        pub = _FakePublisher()
        self.pubs[topic] = pub
        return pub


class _FakeClient:
    def __init__(self, detection=None, tip_target=None) -> None:
        self._detection = detection
        self._tip_target = tip_target

    def get_last_detection(self):
        return self._detection

    def get_tip_target(self):
        """The 9D estimate behind the detection; None on a model with no head."""
        return self._tip_target


class _FakeActions:
    """No chunk yet, so ``publish`` returns right after the detection."""

    def __init__(self) -> None:
        self.action_lock = threading.Lock()
        self.action_buffer: dict = {}


def _detection(u: float, v: float) -> PerceptionDetection:
    return PerceptionDetection(
        uv_px=np.array([u, v], dtype=float),
        visibility=0.9,
        visible=True,
        depth_m=0.25,
        wall_normal_cam=np.array([0.0, 0.0, 1.0]),
    )


@pytest.fixture
def publisher():
    node = _FakeNode()
    return VlaDebugPublisher(node), node


def _detection_msgs(node) -> list:
    return node.pubs[topics.TOPIC_PERCEPTION_DETECTION].published


def test_detection_client_overrides_the_driving_client(publisher):
    """The coarse model drives; the detection comes from the detector client."""
    debug, node = publisher
    coarse = _FakeClient(None)  # action-only model: no perception head
    fine = _FakeClient(_detection(360.0, 370.0))

    debug.publish(_FakeActions(), coarse, None, detection_client=fine)

    assert _detection_msgs(node)[-1].data[:2] == pytest.approx([360.0, 370.0])


def test_absent_detection_clears_the_overlay(publisher):
    """An empty message is what stops RViz freezing a stale crosshair."""
    debug, node = publisher

    debug.publish(_FakeActions(), _FakeClient(None), None, _FakeClient(None))

    assert list(_detection_msgs(node)[-1].data) == []


def _tip_target_msgs(node) -> list:
    return node.pubs[topics.TOPIC_VLA_TIP_TARGET].published


def _driver_msgs(node) -> list:
    return node.pubs[topics.TOPIC_VLA_TIP_TARGET_DRIVER].published


def test_split_roles_publish_one_tip_target_frame_each(publisher):
    """Each model's estimate is published on its own frame."""
    debug, node = publisher
    coarse = _FakeClient(None, tip_target=np.arange(9, dtype=float))
    fine = _FakeClient(_detection(360.0, 370.0), tip_target=np.ones(9))

    debug.publish(_FakeActions(), coarse, None, detection_client=fine)

    assert list(_tip_target_msgs(node)[-1].data) == pytest.approx(list(np.ones(9)))
    assert list(_driver_msgs(node)[-1].data) == pytest.approx(list(np.arange(9.0)))


def test_a_driver_with_no_estimate_leaves_its_frame_empty(publisher):
    """No stale frame is left behind."""
    debug, node = publisher
    coarse = _FakeClient(None, tip_target=None)  # vanilla driver: no head
    fine = _FakeClient(_detection(360.0, 370.0), tip_target=np.ones(9))

    debug.publish(_FakeActions(), coarse, None, detection_client=fine)

    assert list(_driver_msgs(node)[-1].data) == []


def test_alignment_pixels_are_appended_to_the_overlay_message(publisher):
    debug, node = publisher

    debug.publish_perception(
        None,
        None,
        _detection(100.0, 200.0),
        setpoint_uv_px=np.array([360.0, 361.0]),
        servo_target_uv_px=np.array([370.0, 350.0]),
    )

    assert _detection_msgs(node)[-1].data[5:] == pytest.approx(
        [360.0, 361.0, 370.0, 350.0]
    )


def test_alignment_pixels_can_render_without_a_live_detection(publisher):
    debug, node = publisher

    debug.publish_perception(None, None, None, setpoint_uv_px=np.array([360.0, 361.0]))

    data = _detection_msgs(node)[-1].data
    assert np.all(np.isnan(data[:5]))
    assert data[5:7] == pytest.approx([360.0, 361.0])
    assert np.all(np.isnan(data[7:9]))
