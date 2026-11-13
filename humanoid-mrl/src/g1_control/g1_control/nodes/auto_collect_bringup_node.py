# MIT License
"""Bringup sequencer for headless AUTO_COLLECT workers.

Once the simulation is alive, requests a sim reset (base lock) and walks the
FSM ``damping -> ready -> auto_collect`` over ``TOPIC_FSM_TRIGGER``; AUTO_COLLECT
loop mode then chains episodes. Re-sequences if the FSM leaves AUTO_COLLECT.

Each trigger is re-published until the FSM confirms it, then held for a settle
time on the task clock (sim time). The trigger retry uses the wall clock so it
keeps firing if ``/clock`` stalls.
"""
from __future__ import annotations

import time

import rclpy
from g1_definitions import definitions
from rclpy.node import Node
from std_msgs.msg import Empty, String, UInt8
from unitree_hg.msg import LowState

from g1_control.utils.task_clock import TaskClock


class AutoCollectBringupNode(Node):
    """Drive the FSM into AUTO_COLLECT once the simulation publishes lowstate."""

    def __init__(self) -> None:
        super().__init__("auto_collect_bringup_node")

        # Settle times in sim seconds. ready_settle_s = ReadyController's 10 s ramp
        # plus margin.
        self.declare_parameter("damping_settle_s", 2.0)
        self.declare_parameter("ready_settle_s", 13.0)
        self.declare_parameter("trigger_retry_s", 3.0)

        # Task (sim) clock for settle times. Not `self._clock`: rclpy.Node owns it.
        self._task_clock = TaskClock(self)

        # Wall-clock liveness retry (must fire even if /clock stalls).
        retry_s = float(self.get_parameter("trigger_retry_s").value)
        self._retry_s = max(0.5, retry_s)

        # (FSM trigger to publish, state that confirms it, settle time in it).
        self._stages: list[tuple[str, int, float]] = [
            (
                "damping",
                definitions.FSMState.DAMPING.value,
                float(self.get_parameter("damping_settle_s").value),
            ),
            (
                "ready",
                definitions.FSMState.READY.value,
                float(self.get_parameter("ready_settle_s").value),
            ),
            ("auto_collect", definitions.FSMState.AUTO_COLLECT.value, 0.0),
        ]
        self._stage_idx = 0
        self._stage_confirmed_at: float | None = None
        self._last_trigger_at: float = 0.0
        self._reset_requested = False
        self._done_logged = False

        self._lowstate_seen = False
        self._fsm_state: int | None = None

        self._trigger_pub = self.create_publisher(
            String, definitions.TOPIC_FSM_TRIGGER, 1
        )
        # Locks the base and randomizes the scene (headless stand-in for R).
        self._reset_request_pub = self.create_publisher(
            Empty, definitions.TOPIC_SIM_RESET_REQUEST, 1
        )
        self._lowstate_sub = self.create_subscription(
            LowState, definitions.TOPIC_LOWSTATE, self._lowstate_cb, 1
        )
        self._fsm_sub = self.create_subscription(
            UInt8, definitions.TOPIC_FSMSTATE, self._fsm_cb, 1
        )

        self._timer = self.create_timer(0.5, self._tick)
        self.get_logger().info(
            "Auto-collect bringup: waiting for the simulation, then sequencing "
            "damping -> ready -> auto_collect."
        )

    def _lowstate_cb(self, _msg: LowState) -> None:
        self._lowstate_seen = True

    def _fsm_cb(self, msg: UInt8) -> None:
        self._fsm_state = int(msg.data)

    def _tick(self) -> None:
        if not self._lowstate_seen or self._fsm_state is None:
            self.get_logger().info(
                "Auto-collect bringup: waiting for "
                f"{'lowstate' if not self._lowstate_seen else 'the FSM state topic'}...",
                throttle_duration_sec=5.0,
            )
            return

        # A stalled /clock would make every settle measure zero; hold instead.
        if self._task_clock.stalled():
            self.get_logger().warn(
                "Auto-collect bringup: use_sim_time is set but /clock has "
                "published nothing — holding, every settle would measure zero.",
                throttle_duration_sec=5.0,
            )
            return

        # One-time scene reset (BaseLock + randomization) before READY.
        if not self._reset_requested:
            self._reset_requested = True
            self._reset_request_pub.publish(Empty())
            self.get_logger().info(
                "Auto-collect bringup: simulation alive — requested initial scene "
                "reset (base lock + randomization)."
            )

        if self._stage_idx >= len(self._stages):
            if not self._done_logged:
                self._done_logged = True
                self.get_logger().info(
                    "Auto-collect bringup complete: FSM in AUTO_COLLECT, loop mode "
                    "takes it from here."
                )
            # Re-sequence if the FSM leaves AUTO_COLLECT.
            if self._fsm_state != definitions.FSMState.AUTO_COLLECT.value:
                self.get_logger().warn(
                    "Auto-collect bringup: FSM left AUTO_COLLECT "
                    f"(state {self._fsm_state}) — re-sequencing."
                )
                self._stage_idx = 0
                self._stage_confirmed_at = None
                self._done_logged = False
            return

        trigger, target_state, settle_s = self._stages[self._stage_idx]
        now = self._task_clock()

        if self._fsm_state == target_state:
            if self._stage_confirmed_at is None:
                self._stage_confirmed_at = now
                self.get_logger().info(
                    f"Auto-collect bringup: FSM reached {trigger} — settling "
                    f"{settle_s:.1f} {'sim ' if self._task_clock.sim_time else ''}s."
                )
            if now - self._stage_confirmed_at >= settle_s:
                self._stage_idx += 1
                self._stage_confirmed_at = None
            return

        self._stage_confirmed_at = None
        wall_now = time.monotonic()
        if wall_now - self._last_trigger_at >= self._retry_s:
            self._last_trigger_at = wall_now
            self._trigger_pub.publish(String(data=trigger))
            self.get_logger().info(
                f"Auto-collect bringup: sent FSM trigger '{trigger}'."
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoCollectBringupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
