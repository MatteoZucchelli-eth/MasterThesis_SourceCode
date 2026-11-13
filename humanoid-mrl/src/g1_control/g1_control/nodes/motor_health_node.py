# MIT License
"""Standalone motor fault monitor for the G1 base motors.

Subscribes only to ``/lowstate`` and publishes nothing; safe to run alongside
any FSM state or against a bag. Reports:

* **torque saturation** -- ``|tau_est|`` vs the URDF ``<limit effort>``.
* **temperature** -- max of ``MotorState.temperature`` [degC].
* **bus voltage** -- ``MotorState.vol``, warned on when it sags.
* **fault word** -- ``MotorState.motorstate`` as raw hex plus set-bit indices
  (not decoded; see Unitree's motor documentation).
* **stream health** -- gaps in ``/lowstate``.

Reporting is windowed: slow fields are not refreshed every message, so each
``report_window_s`` is reduced first (fault words OR-ed, torque maxed, temperature
and voltage median) and only changes in the reduced state are logged.

Usage::

    ros2 run g1_control motor_health_node
    ros2 run g1_control motor_health_node --ros-args -p joints:=right_arm
    ros2 run g1_control motor_health_node --ros-args -p tau_warn_frac:=0.8
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from g1_definitions import definitions
from unitree_hg.msg import LowState

#: Groups selectable via the ``joints`` parameter, on top of JOINT_PR_SLICES.
_ALL = (0, definitions.NUM_MOTOR_G1_BASE)


class MotorHealthNode(Node):
    """Watches ``/lowstate`` and reports motor faults as they appear."""

    def __init__(self) -> None:
        super().__init__("motor_health_node")

        self.declare_parameter("joints", "all")
        self.declare_parameter("urdf_path", "")
        # Fraction of the effort limit that counts as saturating (< 1 to warn early).
        self.declare_parameter("tau_warn_frac", 0.85)
        self.declare_parameter("temp_warn_c", 70.0)
        self.declare_parameter("vol_warn_v", 40.0)
        #: Reduction window [s] (see the module docstring).
        self.declare_parameter("report_window_s", 0.5)
        #: Seconds between "still healthy" summaries. 0 disables them.
        self.declare_parameter("summary_period_s", 10.0)
        #: A /lowstate gap longer than this is reported.
        self.declare_parameter("stale_warn_s", 0.5)

        group = self.get_parameter("joints").value
        lo, hi = definitions.JOINT_PR_SLICES.get(group, _ALL)
        if group not in definitions.JOINT_PR_SLICES and group != "all":
            self.get_logger().warn(
                f"joints:={group!r} is not one of "
                f"{sorted(definitions.JOINT_PR_SLICES)} or 'all' -- watching all."
            )
        self._idx = list(range(lo, hi))
        self._names = definitions.G1_BASE_PR_JOINT_NAMES

        self._tau_frac = float(self.get_parameter("tau_warn_frac").value)
        self._temp_warn = float(self.get_parameter("temp_warn_c").value)
        self._vol_warn = float(self.get_parameter("vol_warn_v").value)
        self._stale_warn = float(self.get_parameter("stale_warn_s").value)

        self._effort = self._load_effort_limits()

        n = definitions.NUM_MOTOR_G1_BASE
        # Per-window accumulators, reduced and cleared by _evaluate.
        self._acc_fault = np.zeros(n, dtype=np.uint32)  # OR-ed: sticky
        self._acc_tau = np.zeros(n)  # max
        self._acc_temp: list[list[float]] = [[] for _ in range(n)]  # median
        self._acc_vol: list[list[float]] = [[] for _ in range(n)]  # median
        self._acc_n = 0
        # Last state actually reported, so a steady fault logs once.
        self._fault_word: dict[int, int] = {}
        self._saturating: set[int] = set()
        self._hot: set[int] = set()
        self._low_volt: set[int] = set()
        #: Worst |tau_est| and temperature seen per motor, for the exit summary.
        self._peak_tau = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        self._peak_temp = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        self._msgs = 0
        self._last_msg_t: float | None = None
        self._gaps = 0

        self.create_subscription(
            LowState, definitions.TOPIC_LOWSTATE, self._on_low_state, 1
        )
        self.create_timer(
            max(0.05, float(self.get_parameter("report_window_s").value)),
            self._evaluate,
        )
        period = float(self.get_parameter("summary_period_s").value)
        if period > 0.0:
            self.create_timer(period, self._summary)

        # 2 s, so the discovery of a second publisher is not raced at startup.
        self.create_timer(2.0, self._check_publishers)
        self._pub_warned = 0

        watched = f"{self._names[lo]} .. {self._names[hi - 1]}"
        self.get_logger().info(
            f"[motor-health] watching {len(self._idx)} motors ({watched}) on "
            f"{definitions.TOPIC_LOWSTATE}. Saturation at "
            f"{self._tau_frac:.0%} of the URDF effort limit, temp > "
            f"{self._temp_warn:.0f}C, vol < {self._vol_warn:.0f}V."
        )

    # ── Setup ────────────────────────────────────────────────────────────────

    def _load_effort_limits(self) -> np.ndarray | None:
        """Per-motor ``<limit effort>`` from the URDF, PR order, or None if
        unavailable."""
        path = str(self.get_parameter("urdf_path").value)
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory

                share = Path(get_package_share_directory("g1_state_publisher"))
                # Both variants carry the same actuator limits.
                for name in (
                    "g1_29_dof_with_hand_drill.urdf.xml",
                    "g1_29_dof_with_hand.urdf.xml",
                ):
                    if (share / "urdf" / name).exists():
                        path = str(share / "urdf" / name)
                        break
            except Exception as exc:  # noqa: BLE001 - diagnostics must not die
                self.get_logger().warn(f"[motor-health] URDF lookup failed: {exc}")

        if not path or not Path(path).exists():
            self.get_logger().warn(
                "[motor-health] no URDF found -- torque-saturation check is OFF. "
                "Pass -p urdf_path:=<file> to enable it."
            )
            return None

        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            self.get_logger().warn(f"[motor-health] cannot parse {path}: {exc}")
            return None

        by_name = {}
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is not None and limit.get("effort") is not None:
                by_name[joint.get("name")] = float(limit.get("effort"))

        effort = np.full(definitions.NUM_MOTOR_G1_BASE, np.nan)
        missing = []
        for i, name in enumerate(self._names[: definitions.NUM_MOTOR_G1_BASE]):
            if name in by_name:
                effort[i] = by_name[name]
            else:
                missing.append(name)
        if missing:
            self.get_logger().warn(
                f"[motor-health] {len(missing)} joints have no effort limit in "
                f"the URDF (first: {missing[0]}); those are not checked."
            )
        self.get_logger().info(f"[motor-health] effort limits from {Path(path).name}")
        return effort

    # ── Callback ─────────────────────────────────────────────────────────────

    def _on_low_state(self, msg: LowState) -> None:
        """Accumulate one message. All judgement happens in :meth:`_evaluate`."""
        self._msgs += 1
        self._acc_n += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_msg_t is not None and now - self._last_msg_t > self._stale_warn:
            self._gaps += 1
            self.get_logger().warn(
                f"[motor-health] {now - self._last_msg_t:.2f} s gap in "
                f"{definitions.TOPIC_LOWSTATE} (gap #{self._gaps})"
            )
        self._last_msg_t = now

        for i in self._idx:
            m = msg.motor_state[i]
            # OR, so a single-message fault survives the reduction.
            self._acc_fault[i] |= np.uint32(m.motorstate)
            self._acc_tau[i] = max(self._acc_tau[i], abs(float(m.tau_est)))
            # Temperature and voltage are reduced by median to reject bad frames.
            self._acc_temp[i].append(max(float(t) for t in m.temperature))
            vol = float(m.vol)
            if vol > 0.0:  # 0 V is "not reported in this message", not a dead bus
                self._acc_vol[i].append(vol)

    def _evaluate(self) -> None:
        """Reduce the window and report only what changed since the last one."""
        if self._acc_n == 0:
            return
        log = self.get_logger()

        for i in self._idx:
            name = self._names[i]

            # None when no message in this window carried a temperature.
            temp = float(np.median(self._acc_temp[i])) if self._acc_temp[i] else None

            word = int(self._acc_fault[i])
            if word != self._fault_word.get(i, 0):
                if word:
                    bits = [b for b in range(32) if word >> b & 1]
                    at_temp = "n/a" if temp is None else f"{temp:.0f}C"
                    log.error(
                        f"[motor-health] FAULT motor {i} ({name}): "
                        f"motorstate=0x{word:08X} bits={bits} "
                        f"(peak |tau|={float(self._acc_tau[i]):.2f} Nm, "
                        f"temp={at_temp})"
                    )
                else:
                    log.info(f"[motor-health] fault CLEARED motor {i} ({name})")
                self._fault_word[i] = word

            tau = float(self._acc_tau[i])
            self._peak_tau[i] = max(self._peak_tau[i], tau)
            if self._effort is not None and np.isfinite(self._effort[i]):
                limit = float(self._effort[i])
                if tau >= self._tau_frac * limit:
                    if i not in self._saturating:
                        self._saturating.add(i)
                        log.error(
                            f"[motor-health] TORQUE motor {i} ({name}): "
                            f"|tau_est|={tau:.2f} Nm >= {self._tau_frac:.0%} of "
                            f"{limit:.0f} Nm. A clipping joint invalidates any "
                            f"Kp*(q_des-q_meas) measurement at this pose."
                        )
                elif i in self._saturating and tau < 0.8 * self._tau_frac * limit:
                    # Hysteresis.
                    self._saturating.discard(i)
                    log.info(
                        f"[motor-health] torque recovered motor {i} ({name}): "
                        f"{tau:.2f} Nm"
                    )

            # Not `continue`-guarded: voltage is checked even without a temperature.
            if temp is not None:
                self._peak_temp[i] = max(self._peak_temp[i], temp)
                if temp >= self._temp_warn and i not in self._hot:
                    self._hot.add(i)
                    log.error(f"[motor-health] TEMP motor {i} ({name}): {temp:.0f} degC")
                elif temp < self._temp_warn - 5.0 and i in self._hot:
                    self._hot.discard(i)
                    log.info(
                        f"[motor-health] temp recovered motor {i} ({name}): {temp:.0f}C"
                    )

            if self._acc_vol[i]:
                vol = float(np.median(self._acc_vol[i]))
                if vol < self._vol_warn and i not in self._low_volt:
                    self._low_volt.add(i)
                    log.warn(
                        f"[motor-health] VOLTAGE motor {i} ({name}): {vol:.1f} V"
                    )
                elif vol >= self._vol_warn + 1.0 and i in self._low_volt:
                    self._low_volt.discard(i)

        self._acc_fault[:] = 0
        self._acc_tau[:] = 0.0
        for i in self._idx:
            self._acc_temp[i].clear()
            self._acc_vol[i].clear()
        self._acc_n = 0

    def _check_publishers(self) -> None:
        """Error if more than one node publishes ``/lowstate`` (e.g. a bag replay onto
        the live topic, which the control loop cannot distinguish)."""
        count = self.count_publishers(definitions.TOPIC_LOWSTATE)
        if count > 1 and count != self._pub_warned:
            self.get_logger().error(
                f"[motor-health] {count} publishers on "
                f"{definitions.TOPIC_LOWSTATE} -- expected 1. Something is "
                "injecting state into the live topic (a bag replay?). The "
                "control loop cannot tell the streams apart: stop it before "
                "trusting any measurement taken now."
            )
        self._pub_warned = count

    # ── Reporting ────────────────────────────────────────────────────────────

    def _summary(self) -> None:
        if self._msgs == 0:
            self.get_logger().warn(
                f"[motor-health] no {definitions.TOPIC_LOWSTATE} messages yet."
            )
            return

        active = []
        if self._saturating:
            active.append(f"torque={[self._names[i] for i in sorted(self._saturating)]}")
        if self._hot:
            active.append(f"temp={[self._names[i] for i in sorted(self._hot)]}")
        if self._low_volt:
            active.append(f"vol={[self._names[i] for i in sorted(self._low_volt)]}")
        faults = {i: w for i, w in self._fault_word.items() if w}
        if faults:
            active.append(
                "faults="
                + ", ".join(f"{self._names[i]}=0x{w:08X}" for i, w in faults.items())
            )

        worst = max(self._idx, key=lambda i: self._peak_tau[i])
        head = f"[motor-health] {self._msgs} msgs"
        tail = (
            f"peak |tau| {self._peak_tau[worst]:.2f} Nm on {self._names[worst]}, "
            f"peak temp {self._peak_temp[self._idx].max():.0f}C"
        )
        if active:
            self.get_logger().error(f"{head} | ACTIVE: {'; '.join(active)} | {tail}")
        else:
            self.get_logger().info(f"{head} | no faults | {tail}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotorHealthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Session peak table. print, not the logger: rosout is invalid at shutdown.
        print("[motor-health] session peaks (PR index, Nm, degC):")
        for i in node._idx:
            if node._peak_tau[i] > 0.0 or node._peak_temp[i] > 0.0:
                limit = (
                    node._effort[i]
                    if node._effort is not None and np.isfinite(node._effort[i])
                    else float("nan")
                )
                print(
                    f"  {i:>2} {node._names[i]:<30} "
                    f"peak |tau|={node._peak_tau[i]:>6.2f} / {limit:>5.1f} Nm "
                    f"({node._peak_tau[i] / limit:>5.0%})  "
                    f"peak temp={node._peak_temp[i]:>5.0f}C"
                )
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
