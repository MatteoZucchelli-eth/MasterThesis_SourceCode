# MIT License
"""Scripted drill-tip sweep for measuring a model's estimate against ground truth.

In test mode the VLA state keeps running inference but drives the arm through
IK waypoints at known offsets from the target TF, recording the tip→target
estimate error per waypoint (``pred_rel_*`` / ``rel_err_*``) via MetricRecorder.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import rclpy
from visualization_msgs.msg import Marker, MarkerArray

from g1_control import topics
from g1_control.states.base import ControlContext
from g1_control.utils.ros_helpers import LATCHED_QOS, lookup_position, sphere_marker
from g1_control.utils.task_clock import TaskClock


class ProgressTestHarness:
    """Sweeps the drill tip through scripted waypoints, logging prediction error.

    Not a StateHandler itself: it runs *inside* the VLA state, dispatched by
    :class:`~g1_control.states.vla.VlaHandler` when test mode is enabled.
    """

    def __init__(self, node, ctx: ControlContext, metrics) -> None:
        self._ctx = ctx
        self._logger = node.get_logger()
        self._clock = TaskClock(node)
        self._metrics = metrics

        self.enabled = bool(
            node.get_parameter("progress_test_mode").value
            if node.has_parameter("progress_test_mode")
            else False
        )
        self._dwell_s = float(
            node.get_parameter("progress_test_dwell_s").value
            if node.has_parameter("progress_test_dwell_s")
            else 6.0
        )

        # Drill-tip offsets [m] from the target, pelvis frame (+x into the wall,
        # +y left, +z up), ordered far -> on-target.
        self._offsets = [
            np.array([-0.04, -0.05, 0.05]),
            np.array([-0.04, -0.05, -0.05]),
            np.array([-0.04, 0.05, -0.05]),
            np.array([-0.04, 0.0, -0.05]),
            np.array([-0.01, -0.05, 0.05]),
            np.array([-0.01, -0.05, -0.05]),
            np.array([-0.01, 0.05, -0.05]),
            np.array([-0.01, 0.0, -0.05]),
            # Final straight-in approach onto the target.
            np.array([-0.0, 0.0, 0.0]),
        ]

        # Lazily-built runtime state (populated on the first VLA test tick).
        self._waypoints: list[pin.SE3] | None = None
        self._index = 0
        self._waypoint_target: np.ndarray | None = None
        self._wp_start: float | None = None
        self._done = False

        self._marker_pub = node.create_publisher(
            MarkerArray, topics.TOPIC_PROGRESS_TEST_WAYPOINTS, LATCHED_QOS
        )

        if self.enabled:
            self._logger.info(
                "Progress-test mode ENABLED: VLA state runs a scripted drill-tip "
                "sweep; VLA actions will NOT drive the arm."
            )

    def reset(self) -> None:
        """Clear the sweep state so a VLA re-entry rebuilds it from the target."""
        self._waypoints = None
        self._index = 0
        self._waypoint_target = None
        self._wp_start = None
        self._done = False

    def update(self) -> None:
        """Per-tick sweep step (called while in the VLA state, test mode on).

        The VLA action chunk is ignored: ``upper_body_target_chunk`` stays None
        so the controller follows the IK targets (Path B).
        """
        rs = self._ctx.robot_state
        # Hold the base still; never populate the chunk so Path B is used.
        rs.twist_command = np.zeros(3)
        rs.upper_body_target_chunk = None

        if self._done:
            return

        if self._waypoints is None:
            if not self._build_waypoints():
                return  # TF/target not ready yet
            self._advance_waypoint(0)

        now = self._clock()

        # Keep commanding the current waypoint's cached joint target.
        if self._waypoint_target is not None:
            rs.upper_body_target = self._waypoint_target

        self._metrics.record(now, waypoint_idx=self._index)

        if now - self._wp_start >= self._dwell_s:
            next_index = self._index + 1
            if next_index >= len(self._waypoints):
                self._finalize()
            else:
                self._advance_waypoint(next_index)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_waypoints(self) -> bool:
        """Build absolute drill-tip goal SE3s from the live target-square TF.

        The tip orientation is captured once from current FK (drill pointing into
        the wall) and reused for every waypoint. Returns ``True`` once built.
        """
        ctx = self._ctx
        target_pos = lookup_position(ctx.tf_buffer, ctx.drill_target_frame, "pelvis")
        if target_pos is None:
            self._logger.warn(
                (
                    f"Progress test: waiting for {ctx.drill_target_frame} TF..."
                    if ctx.drill_target_frame is not None
                    else "Progress test: no target square resolved from the task prompt..."
                ),
                throttle_duration_sec=2.0,
            )
            return False

        ctx.robot_model.cache_forward_kinematics(ctx.full_q, auto_clip=False)
        tip_rot = ctx.robot_model.frame_placement("drill_tip_link").rotation.copy()

        self._waypoints = [pin.SE3(tip_rot, target_pos + off) for off in self._offsets]
        self._logger.info(
            f"Progress test: built {len(self._waypoints)} tip waypoints around "
            f"red_target at {target_pos.round(3)} (pelvis frame)."
        )
        self._publish_markers()
        return True

    def _advance_waypoint(self, index: int) -> None:
        """Solve IK for waypoint ``index`` and cache the upper-body joint target."""
        ctx = self._ctx
        self._index = index
        target_se3 = self._waypoints[index]
        right_arm_q = ctx.robot_model.solve_ik("drill_tip_link", target_se3, ctx.full_q)
        self._waypoint_target = ctx.robot_model.assemble_upper_body_target(
            ctx.default_upper_body_pose, ctx.full_q, right_arm_q
        )
        self._wp_start = self._clock()
        self._logger.info(
            f"Progress test: waypoint {index + 1}/{len(self._waypoints)}, "
            f"tip target (pelvis) = {target_se3.translation.round(3)}"
        )
        self._publish_markers()

    def _finalize(self) -> None:
        """Complete the scripted sweep: flush the CSV, stop, refresh markers."""
        self._done = True
        self._metrics.finalize()
        self._publish_markers()

    def _publish_markers(self) -> None:
        """Publish the drill-tip waypoints as an RViz MarkerArray (pelvis frame).

        Active waypoint green, completed gray, upcoming blue.
        """
        if self._waypoints is None:
            return

        stamp = rclpy.time.Time().to_msg()
        markers = MarkerArray()
        for i, wp in enumerate(self._waypoints):
            if self._done or i < self._index:
                color = (0.5, 0.5, 0.5, 0.6)  # completed: gray
            elif i == self._index:
                color = (0.1, 0.9, 0.1, 1.0)  # active: green
            else:
                color = (0.1, 0.4, 0.9, 0.8)  # upcoming: blue

            markers.markers.append(
                sphere_marker("progress_test_waypoints", i, wp.translation, color, 0.03)
            )

            px, py, pz = (float(v) for v in wp.translation)
            label = Marker()
            label.header.frame_id = "pelvis"
            label.header.stamp = stamp
            label.ns = "progress_test_labels"
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = px
            label.pose.position.y = py
            label.pose.position.z = pz + 0.04
            label.pose.orientation.w = 1.0
            label.scale.z = 0.03
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = str(i)
            markers.markers.append(label)

        self._marker_pub.publish(markers)
