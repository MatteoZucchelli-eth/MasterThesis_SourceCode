# MIT License
"""Unit tests for g1_common.image_transport — topic name -> transport."""

import threading

import numpy as np
import pytest

from g1_common.image_transport import (
    CompressedTransport,
    ImageTransport,
    RawTransport,
    subscribe_image,
    subscribe_image_latest,
    transport_class_for_topic,
    transport_name,
)

WRIST = "/camera/hand/color/image_raw"
HEAD = "/camera/d435/color/image_raw"


# ── Selection ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("base", [WRIST, HEAD])
def test_bare_topic_selects_raw(base):
    assert transport_class_for_topic(base) is RawTransport
    assert transport_name(base) == "raw"


@pytest.mark.parametrize("base", [WRIST, HEAD])
def test_compressed_suffix_selects_compressed(base):
    topic = base + "/compressed"
    assert transport_class_for_topic(topic) is CompressedTransport
    assert transport_name(topic) == "compressed"


def test_compressed_is_matched_as_a_suffix_not_a_substring():
    """Only a trailing ``/compressed`` selects the compressed transport."""
    assert transport_class_for_topic("/camera/compressed/image_raw") is RawTransport
    assert transport_class_for_topic("/camera/hand/compressed_view") is RawTransport


def test_name_needs_no_bridge_and_no_ros():
    """transport_name is pure string work."""
    assert transport_name(WRIST + "/compressed") == "compressed"


# ── Substitutability ──────────────────────────────────────────────────────────


class FakeTransport(ImageTransport):
    """A transport that decodes nothing, to prove callers depend on the interface."""

    NAME = "fake"
    MESSAGE_TYPE = object()

    def __init__(self):
        # Skips super().__init__: no CvBridge needed.
        self.decoded = []

    @property
    def message_type(self):
        return self.MESSAGE_TYPE

    def decode(self, msg, encoding):
        self.decoded.append((msg, encoding))
        return np.full((2, 2, 3), 7, dtype=np.uint8)


class ExplodingTransport(FakeTransport):
    """Fails every decode, the way a truncated JPEG would."""

    def decode(self, msg, encoding):
        raise ValueError("not a JPEG")


class FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message, **_kwargs):
        self.errors.append(message)

    def warning(self, message, **_kwargs):
        self.errors.append(message)


class FakeNode:
    """The two node methods subscribe_image uses."""

    def __init__(self):
        self.logger = FakeLogger()
        self.subscriptions = []

    def get_logger(self):
        return self.logger

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((message_type, topic, callback, qos))
        return f"sub:{topic}"


def test_subscribe_image_uses_the_transport_message_type():
    node, transport = FakeNode(), FakeTransport()
    handle = subscribe_image(node, WRIST, lambda *_: None, "qos", transport=transport)

    ((message_type, topic, _callback, qos),) = node.subscriptions
    assert message_type is FakeTransport.MESSAGE_TYPE
    assert topic == WRIST
    assert qos == "qos"
    assert handle == f"sub:{WRIST}"


def test_callback_receives_decoded_pixels_and_the_original_message():
    node, transport = FakeNode(), FakeTransport()
    seen = []
    subscribe_image(
        node,
        WRIST,
        lambda img, msg: seen.append((img, msg)),
        "qos",
        encoding="mono8",
        transport=transport,
    )
    _, _, on_message, _ = node.subscriptions[0]

    sentinel = object()
    on_message(sentinel)

    ((image, msg),) = seen
    assert msg is sentinel, "the message has to survive: consumers need its header"
    assert image.shape == (2, 2, 3)
    assert transport.decoded == [(sentinel, "mono8")]


def test_a_bad_frame_is_dropped_rather_than_raised():
    """One undecodable frame must not unwind spin and kill the node."""
    node = FakeNode()
    seen = []
    subscribe_image(
        node,
        WRIST,
        lambda *args: seen.append(args),
        "qos",
        transport=ExplodingTransport(),
    )
    _, _, on_message, _ = node.subscriptions[0]

    on_message(object())  # must not raise

    assert seen == [], "a frame that failed to decode must not reach the consumer"
    assert len(node.logger.errors) == 1
    assert WRIST in node.logger.errors[0]


# ── Latest-only asynchronous decode ──────────────────────────────────────────


class BlockingTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def decode(self, msg, encoding):
        self.decoded.append((msg, encoding))
        if len(self.decoded) == 1:
            self.started.set()
            assert self.release.wait(1.0)
        return np.full((2, 2, 3), int(msg), dtype=np.uint8)


def test_latest_subscription_decodes_off_the_ros_callback_and_drops_backlog():
    node, transport = FakeNode(), BlockingTransport()
    seen = []
    worker = subscribe_image_latest(
        node,
        WRIST,
        lambda image, msg: seen.append((int(image[0, 0, 0]), msg)),
        "qos",
        transport=transport,
    )
    _, _, on_message, _ = node.subscriptions[0]

    try:
        on_message(1)
        assert transport.started.wait(1.0), "decode runs on the worker"
        # While decode 1 blocks, message 2 is replaced by 3.
        on_message(2)
        on_message(3)
        transport.release.set()
        for _ in range(100):
            if len(seen) == 2:
                break
            threading.Event().wait(0.01)
    finally:
        worker.close()

    assert [msg for _, msg in seen] == [1, 3]
    assert [msg for msg, _ in transport.decoded] == [1, 3]


def test_closing_latest_subscription_ignores_later_ros_callbacks():
    node, transport = FakeNode(), FakeTransport()
    worker = subscribe_image_latest(
        node, WRIST, lambda *_: None, "qos", transport=transport
    )
    _, _, on_message, _ = node.subscriptions[0]
    worker.close()

    on_message(object())
    threading.Event().wait(0.02)
    assert transport.decoded == []
