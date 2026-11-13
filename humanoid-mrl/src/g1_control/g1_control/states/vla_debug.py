# MIT License
"""RViz debug overlays for VLA and VLA_PLANNER, consumed by debug_viz_node.

Publishes predicted palm chunks (latest and pre-OOD-correction), tip->target
predictions, fused target estimates and the raw 2D detection as flat arrays;
RViz formatting lives in the debug node.
"""
from __future__ import annotations

import numpy as np
from g1_definitions import definitions
from std_msgs.msg import Float32MultiArray

from g1_control import topics


class VlaDebugPublisher:
    """Publishes VLA chunks and target estimates for the debug viz node."""

    def __init__(self, node) -> None:
        self._right_chunk_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_CHUNK_RIGHT, 1
        )
        self._right_chunk_prev_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_CHUNK_RIGHT_PREV, 1
        )
        # Predicted target pose in the drill-tip frame (flat 9D), one publisher
        # per source: fine/perception role and the driving model's tip_target head.
        self._tip_target_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET, 1
        )
        self._tip_target_driver_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET_DRIVER, 1
        )
        # Fused pelvis-frame target estimate: the servo's EMA in VLA, the
        # accumulator mean in VLA_PLANNER. Empty until one exists.
        self._tip_target_fused_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET_FUSED, 1
        )
        # VLA_PLANNER only: live finalize target (fused + correction) and its
        # never-applied fine-only shadow. Empty in VLA.
        self._tip_target_live_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET_LIVE, 1
        )
        self._tip_target_live_fine_only_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_VLA_TIP_TARGET_LIVE_FINE_ONLY, 1
        )
        # Raw 2D detection of the close-range perception model (VLA_PLANNER),
        # drawn onto the wrist image by the debug node.
        self._perception_detection_pub = node.create_publisher(
            Float32MultiArray, topics.TOPIC_PERCEPTION_DETECTION, 1
        )

    def publish(self, vla_actions, vla_client, fused_9d, detection_client) -> None:
        """Publish the current chunks, the raw prediction and ``fused_9d``.

        Empty messages clear previous overlays. Chunk overlays wait for the first
        chunk; the 2D detection is published regardless. ``detection_client``
        supplies the detection and fine tip-target frame, ``vla_client`` the
        driver frame (empty when it reports none).
        """
        # A non-detection model returns None, which clears the overlay.
        self._publish_detection(detection_client.get_last_detection())
        self._tip_target_pub.publish(self._to_msg(detection_client.get_tip_target()))
        self._tip_target_driver_pub.publish(self._to_msg(vla_client.get_tip_target()))

        with vla_actions.action_lock:
            buffer = vla_actions.action_buffer
            right_chunk = buffer.get(definitions.RIGHT_EEF_KEY)

        if right_chunk is None:
            return

        self._right_chunk_pub.publish(self._to_msg(right_chunk))
        self._right_chunk_prev_pub.publish(
            self._to_msg(vla_client.get_prev_tip_chunk())
        )

        self._tip_target_fused_pub.publish(self._to_msg(fused_9d))

    def publish_perception(
        self,
        tip_target_9d,
        fused_9d,
        detection,
        driver_tip_target_9d=None,
        live_9d=None,
        live_fine_only_9d=None,
        setpoint_uv_px=None,
        servo_target_uv_px=None,
    ) -> None:
        """Publish the perception state's overlays (no VLA action chunks).

        ``tip_target_9d`` is the raw target pose in the drill-tip frame;
        ``fused_9d``, ``live_9d`` and ``live_fine_only_9d`` are pelvis-frame 9D
        (accumulator mean, finalize target, and its detector-depth shadow).
        ``detection`` is a
        :class:`~g1_control.vla.perception.PerceptionDetection` or None.
        ``setpoint_uv_px`` is the drill-axis entry point in the image;
        ``servo_target_uv_px`` the projected frozen 3D target during servoing.
        """
        self._tip_target_pub.publish(self._to_msg(tip_target_9d))
        self._tip_target_driver_pub.publish(self._to_msg(driver_tip_target_9d))
        self._tip_target_fused_pub.publish(self._to_msg(fused_9d))
        self._tip_target_live_pub.publish(self._to_msg(live_9d))
        self._tip_target_live_fine_only_pub.publish(self._to_msg(live_fine_only_9d))
        self._publish_detection(
            detection,
            setpoint_uv_px=setpoint_uv_px,
            servo_target_uv_px=servo_target_uv_px,
        )

    def _publish_detection(
        self, detection, *, setpoint_uv_px=None, servo_target_uv_px=None
    ) -> None:
        """Publish the raw 2D detection (None clears the overlay)."""
        msg = Float32MultiArray()
        if detection is not None:
            msg.data = [
                float(detection.uv_px[0]),
                float(detection.uv_px[1]),
                float(detection.visibility),
                float(detection.visible),
                float(detection.depth_m),
            ]
        elif setpoint_uv_px is not None or servo_target_uv_px is not None:
            msg.data = [float("nan")] * 5

        if msg.data:
            for point in (setpoint_uv_px, servo_target_uv_px):
                if point is None:
                    msg.data.extend([float("nan"), float("nan")])
                else:
                    uv = np.asarray(point, dtype=float).reshape(2)
                    msg.data.extend([float(uv[0]), float(uv[1])])
        self._perception_detection_pub.publish(msg)

    @staticmethod
    def _to_msg(chunk) -> Float32MultiArray:
        """Flatten a (N, 9) chunk / 9D pose into a Float32MultiArray (empty if None)."""
        msg = Float32MultiArray()
        if chunk is not None:
            msg.data = np.asarray(chunk, dtype=np.float32).reshape(-1).tolist()
        return msg
