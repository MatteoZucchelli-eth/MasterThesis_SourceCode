# MIT License
"""Subscribe to a camera stream on either transport, chosen by the topic name.

``<base>`` carries ``sensor_msgs/Image``; ``<base>/compressed`` carries a JPEG
``sensor_msgs/CompressedImage``. The message type is inferred from the topic name,
so a node exposes a single topic parameter.

Compressed is the default on the robot: the raw 1280x720 RELIABLE usb_cam stream
stalls over the network (measured 24 Hz with 319 ms gaps vs 30.3 Hz compressed).
Raw is appropriate when publisher and subscriber share a host (simulation).
"""
from __future__ import annotations

import abc
import threading
from typing import Callable

import numpy as np

COMPRESSED_SUFFIX = "/compressed"
"""``image_transport``'s topic suffix for the JPEG/PNG transport."""

DEFAULT_ENCODING = "rgb8"
"""What every consumer in this stack asks for, sim and robot alike."""


class ImageTransport(abc.ABC):
    """How one camera stream's messages are carried, and turned into pixels.

    Provides only the message type and the decoder; QoS etc. belong to the caller.
    """

    NAME: str
    """Human-readable transport name, for log lines."""

    def __init__(self, bridge=None):
        """:param bridge: a :class:`cv_bridge.CvBridge` to use; built if omitted."""
        from cv_bridge import CvBridge

        self._bridge = CvBridge() if bridge is None else bridge

    @property
    @abc.abstractmethod
    def message_type(self):
        """The ``sensor_msgs`` type this transport's topics carry."""

    @abc.abstractmethod
    def decode(self, msg, encoding: str) -> np.ndarray:
        """Return the ``HxWxC`` array ``msg`` carries, in ``encoding``."""


class RawTransport(ImageTransport):
    """Uncompressed ``sensor_msgs/Image``; no decode cost, heavy on the network."""

    NAME = "raw"

    @property
    def message_type(self):
        from sensor_msgs.msg import Image

        return Image

    def decode(self, msg, encoding: str) -> np.ndarray:
        return self._bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


class CompressedTransport(ImageTransport):
    """JPEG ``sensor_msgs/CompressedImage``: a few ms of decode per frame."""

    NAME = "compressed"

    @property
    def message_type(self):
        from sensor_msgs.msg import CompressedImage

        return CompressedImage

    def decode(self, msg, encoding: str) -> np.ndarray:
        return self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding=encoding)


_TRANSPORT_BY_SUFFIX: tuple[tuple[str, type], ...] = (
    (COMPRESSED_SUFFIX, CompressedTransport),
)
"""Topic suffix -> transport, longest-suffix-first. Raw is the no-suffix fallback."""


def transport_class_for_topic(topic: str) -> type[ImageTransport]:
    """The :class:`ImageTransport` subclass ``topic``'s suffix names (default raw)."""
    for suffix, transport_class in _TRANSPORT_BY_SUFFIX:
        if str(topic).endswith(suffix):
            return transport_class
    return RawTransport


def transport_name(topic: str) -> str:
    """``"compressed"`` or ``"raw"``, for a log line. Needs no ROS and no bridge."""
    return transport_class_for_topic(topic).NAME


def transport_for_topic(topic: str, bridge=None) -> ImageTransport:
    """Build the transport ``topic`` names. See :class:`ImageTransport`."""
    return transport_class_for_topic(topic)(bridge)


def subscribe_image(
    node,
    topic: str,
    callback: Callable[[np.ndarray, object], None],
    qos,
    *,
    encoding: str = DEFAULT_ENCODING,
    bridge=None,
    transport: ImageTransport | None = None,
):
    """Subscribe to ``topic`` and hand ``callback`` decoded pixels.

    ``callback(image, msg)`` receives the decoded ``HxWxC`` array and the
    original message (for header/frame id). Frames that fail to decode are
    dropped with a throttled error.

    :param bridge: forwarded to the transport this builds; ignored when
        ``transport`` is given.
    :param transport: use this :class:`ImageTransport` instead of the one
        ``topic``'s suffix implies (e.g. a test fake).
    :returns: the ``rclpy`` subscription, so the caller can hold or destroy it.
    """
    from cv_bridge import CvBridgeError

    transport = transport_for_topic(topic, bridge) if transport is None else transport

    def _on_message(msg) -> None:
        try:
            image = transport.decode(msg, encoding)
        except (CvBridgeError, ValueError) as exc:
            node.get_logger().error(
                f"Dropped a frame on '{topic}': {exc}", throttle_duration_sec=5.0
            )
            return
        callback(image, msg)

    return node.create_subscription(transport.message_type, topic, _on_message, qos)


class LatestImageSubscription:
    """Latest-only image subscription decoded on a background thread.

    The ROS callback only replaces the pending message; decode and the consumer
    callback run on a daemon worker. Only the newest pending frame is kept.
    """

    def __init__(
        self,
        node,
        topic: str,
        callback: Callable[[np.ndarray, object], None],
        qos,
        *,
        encoding: str = DEFAULT_ENCODING,
        bridge=None,
        transport: ImageTransport | None = None,
    ) -> None:
        self._node = node
        self._topic = str(topic)
        self._callback = callback
        self._encoding = encoding
        self._transport = (
            transport_for_topic(topic, bridge) if transport is None else transport
        )
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self.subscription = node.create_subscription(
            self._transport.message_type, topic, self._on_message, qos
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"image:{self._topic}",
            daemon=True,
        )
        self._thread.start()

    def _on_message(self, msg) -> None:
        """Latch one message without decoding it on the ROS executor."""
        with self._condition:
            if self._closed:
                return
            self._pending = msg
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending is not None
                )
                if self._closed:
                    return
                msg = self._pending
                self._pending = None

            try:
                image = self._transport.decode(msg, self._encoding)
            except Exception as exc:
                # Catch all: decoders raise transport-specific errors (e.g. cv2.error).
                self._node.get_logger().error(
                    f"Dropped a frame on '{self._topic}': {exc}",
                    throttle_duration_sec=5.0,
                )
                continue

            try:
                self._callback(image, msg)
            except Exception as exc:
                # Log consumer errors without killing the worker.
                self._node.get_logger().error(
                    f"Image consumer failed on '{self._topic}': {exc}",
                    throttle_duration_sec=5.0,
                )

    def close(self) -> None:
        """Stop the worker, dropping any frame that has not started decoding."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self._node.get_logger().warning(
                f"Image worker for '{self._topic}' did not stop within 2 s"
            )


def subscribe_image_latest(
    node,
    topic: str,
    callback: Callable[[np.ndarray, object], None],
    qos,
    *,
    encoding: str = DEFAULT_ENCODING,
    bridge=None,
    transport: ImageTransport | None = None,
) -> LatestImageSubscription:
    """Subscribe with latest-only asynchronous decode.

    The returned owner must be retained and closed before its ROS node is
    destroyed.  Its :attr:`LatestImageSubscription.subscription` attribute is
    the underlying rclpy subscription when a caller needs it.
    """
    return LatestImageSubscription(
        node,
        topic,
        callback,
        qos,
        encoding=encoding,
        bridge=bridge,
        transport=transport,
    )
