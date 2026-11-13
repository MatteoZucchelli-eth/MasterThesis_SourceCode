# MIT License
"""Headless sequencer for all simulator VLA evaluation scenarios."""
from __future__ import annotations

import json
import os
import time

import rclpy
from g1_definitions import definitions
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty, String, UInt8
from unitree_hg.msg import LowState

from g1_control.utils.task_clock import TaskClock
from g1_control.vla_eval_report import generate_report


class VlaEvalBringupNode(Node):
    """Select a scene, settle READY, run the policy, and repeat until done."""

    def __init__(self) -> None:
        super().__init__("vla_eval_bringup_node")
        # FSM state for each rollout ("vla" or "vla_planner"); also the FSM trigger
        # string.
        self.declare_parameter("rollout_state", "vla")
        self.declare_parameter("rollout_s", 60.0)
        # End the rollout when VLA_PLANNER reports completion (inert under `vla`).
        self.declare_parameter("stop_on_complete", True)
        self.declare_parameter("damping_settle_s", 2.0)
        # Covers ReadyController's 10 s (500-tick) ramp with margin for a slow
        # control loop (~16.7 s sim time at 30 Hz).
        self.declare_parameter("ready_settle_s", 20.0)
        self.declare_parameter("between_rollouts_s", 2.0)
        self.declare_parameter("scene_settle_s", 2.0)
        self.declare_parameter("trigger_retry_s", 3.0)
        self.declare_parameter("metrics_output_dir", "vla_metrics")

        # rclpy.Node owns ``self._clock``; shadowing it breaks create_timer().
        self._task_clock = TaskClock(self)
        self._retry_s = float(self.get_parameter("trigger_retry_s").value)
        self._rollout_trigger = str(self.get_parameter("rollout_state").value)
        try:
            self._rollout_fsm = definitions.FSMState[self._rollout_trigger.upper()]
        except KeyError:
            raise ValueError(
                f"rollout_state {self._rollout_trigger!r} is not an FSM state; "
                f"expected one of "
                f"{', '.join(state.name.lower() for state in definitions.FSMState)}"
            ) from None
        self._output_dir = str(self.get_parameter("metrics_output_dir").value)
        # Wall clock: compared against CSV mtimes to select this run's sessions.
        self._run_started = time.time()
        self._stage = "waiting_sim"
        self._stage_since: float | None = None
        self._stage_confirmed = False
        self._last_request = 0.0
        self._advance_sent = False
        self._fsm_state: int | None = None
        # Cleared on every stage change, so only this rollout's completion counts.
        self._planner_complete = False
        self._lowstate_seen = False
        self._scenario: dict | None = None
        self._tasks: list[dict] = []
        self._task_idx = 0
        self._completed = 0
        self._total: int | None = None

        self._fsm_pub = self.create_publisher(String, definitions.TOPIC_FSM_TRIGGER, 1)
        self._advance_pub = self.create_publisher(
            Empty, definitions.TOPIC_SIM_EVAL_ADVANCE_REQUEST, 1
        )
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._task_pub = self.create_publisher(
            String, definitions.TOPIC_TASKINSTRUCTION, task_qos
        )
        self._lowstate_sub = self.create_subscription(
            LowState, definitions.TOPIC_LOWSTATE, self._lowstate_cb, 1
        )
        self._fsm_sub = self.create_subscription(
            UInt8, definitions.TOPIC_FSMSTATE, self._fsm_cb, 1
        )
        self._complete_sub = self.create_subscription(
            Bool,
            definitions.TOPIC_VLA_PLANNER_COMPLETE,
            self._complete_cb,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._scenario_sub = self.create_subscription(
            String, definitions.TOPIC_SIM_EVAL_SCENARIO, self._scenario_cb, qos
        )
        self._timer = self.create_timer(0.5, self._tick)
        self.get_logger().info(
            f"Evaluation rollouts will be run in {self._rollout_fsm.name}."
        )
        self._write_status("waiting for simulator and control stack")

    def _lowstate_cb(self, _msg: LowState) -> None:
        self._lowstate_seen = True

    def _fsm_cb(self, msg: UInt8) -> None:
        self._fsm_state = int(msg.data)

    def _complete_cb(self, msg: Bool) -> None:
        self._planner_complete = bool(msg.data)

    def _scenario_cb(self, msg: String) -> None:
        try:
            scenario = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if scenario.get("scenario", -1) < 0 or self._stage != "waiting_scene":
            return
        self._scenario = scenario
        self._tasks = list(
            scenario.get("tasks")
            or [{"instruction": scenario.get("instruction", "")}]
        )
        self._task_idx = 0
        # Total rollouts from the sequencer; the product is a fallback assuming
        # equal task counts per scene.
        self._total = int(
            scenario.get("total_tasks")
            or int(scenario.get("total", 1)) * len(self._tasks)
        )
        self._publish_task()
        self._set_stage("scene_settle")
        self.get_logger().info(
            f"Evaluation {self._completed + 1}/{self._total}: scene "
            f"{scenario['scenario']}, task 1/{len(self._tasks)}"
        )

    def _publish_task(self) -> None:
        instruction = str(self._tasks[self._task_idx]["instruction"])
        self._task_pub.publish(String(data=instruction))
        self.get_logger().info(
            f"Evaluation task {self._task_idx + 1}/{len(self._tasks)}: "
            f"{instruction}"
        )

    def _set_stage(self, stage: str) -> None:
        self._stage = stage
        self._stage_since = self._task_clock()
        self._stage_confirmed = False
        self._planner_complete = False
        if stage == "waiting_scene":
            self._advance_sent = False
        self._write_status(stage.replace("_", " "))

    def _elapsed(self) -> float:
        return (
            0.0
            if self._stage_since is None
            else self._task_clock() - self._stage_since
        )

    def _request(self, publisher, message, description: str) -> None:
        now = time.monotonic()
        if now - self._last_request < self._retry_s:
            return
        self._last_request = now
        publisher.publish(message)
        self.get_logger().info(description)

    def _trigger_until(self, trigger: str, state: definitions.FSMState) -> bool:
        if self._fsm_state == state.value:
            if not self._stage_confirmed:
                self._stage_confirmed = True
                self._stage_since = self._task_clock()
                self._write_status(f"{self._stage.replace('_', ' ')} confirmed")
            return True
        self._stage_confirmed = False
        self._request(
            self._fsm_pub, String(data=trigger), f"Evaluation: requested FSM {trigger}."
        )
        return False

    def _tick(self) -> None:
        if not self._lowstate_seen or self._fsm_state is None:
            return
        if self._task_clock.stalled():
            self._write_status("waiting for simulation clock")
            return

        if self._stage == "waiting_sim":
            self._set_stage("waiting_scene")

        if self._stage == "waiting_scene":
            if not self._advance_sent:
                self._advance_sent = True
                self._advance_pub.publish(Empty())
                self.get_logger().info(
                    "Evaluation: requested the next simulator scenario (V)."
                )
        elif self._stage == "scene_settle" and self._elapsed() >= float(
            self.get_parameter("scene_settle_s").value
        ):
            # DAMPING -> READY only on the first rollout; later ones start from READY.
            self._set_stage("damping" if self._completed == 0 else "vla")
        elif self._stage == "damping":
            if self._trigger_until("damping", definitions.FSMState.DAMPING):
                settle = float(self.get_parameter("damping_settle_s").value)
                if self._elapsed() >= settle:
                    self._set_stage("ready")
        elif self._stage == "ready":
            if self._trigger_until("ready", definitions.FSMState.READY):
                if self._elapsed() >= float(self.get_parameter("ready_settle_s").value):
                    self._set_stage("vla")
        elif self._stage == "vla":
            if self._trigger_until(self._rollout_trigger, self._rollout_fsm):
                early = self._planner_complete and bool(
                    self.get_parameter("stop_on_complete").value
                )
                if early:
                    self.get_logger().info(
                        f"Evaluation: {self._rollout_fsm.name} reported complete "
                        f"after {self._elapsed():.1f} s — ending the rollout early."
                    )
                if early or self._elapsed() >= float(
                    self.get_parameter("rollout_s").value
                ):
                    self._set_stage("return_ready")
        elif self._stage == "return_ready":
            # Wait out the READY ramp so each rollout starts from the default pose.
            if self._trigger_until("ready", definitions.FSMState.READY):
                if self._elapsed() >= float(self.get_parameter("ready_settle_s").value):
                    self._completed += 1
                    self._set_stage("between_rollouts")
        elif self._stage == "between_rollouts" and self._elapsed() >= float(
            self.get_parameter("between_rollouts_s").value
        ):
            # Let the controller flush its CSV before the task (filename/target)
            # changes.
            if self._total is not None and self._completed >= self._total:
                self._set_stage("finalizing")
            elif self._task_idx + 1 < len(self._tasks):
                self._task_idx += 1
                self._publish_task()
                self._set_stage("scene_settle")
            else:
                self._scenario = None
                self._tasks = []
                self._set_stage("waiting_scene")
        elif self._stage == "finalizing" and self._elapsed() >= 2.0:
            report = generate_report(self._output_dir, since=self._run_started)
            self._stage = "complete"
            self._write_status(f"complete; report: {report}")
            self.get_logger().info(
                f"Evaluation complete: {self._completed}/{self._total}; "
                f"report written to {report}"
            )
            self._timer.cancel()

    def _write_status(self, detail: str) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        payload = {
            "status": self._stage,
            "detail": detail,
            "rollout_state": self._rollout_fsm.name,
            "completed": self._completed,
            "total": self._total,
            "scenario": (
                None if self._scenario is None else self._scenario.get("scenario")
            ),
            "task": None if not self._tasks else self._task_idx + 1,
            "tasks_in_scene": len(self._tasks) or None,
            "instruction": (
                None if not self._tasks else self._tasks[self._task_idx]["instruction"]
            ),
        }
        path = os.path.join(self._output_dir, "evaluation_status.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(tmp, path)
        except OSError as error:
            self.get_logger().warning(f"Could not write evaluation status: {error}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlaEvalBringupNode()
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
