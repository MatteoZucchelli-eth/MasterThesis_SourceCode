# MIT License
"""The clock a task's own durations are measured on.

On sim, task durations follow ``/clock`` so robot motion is the same at any
real-time factor. These stay on the wall clock:

* **Liveness watchdogs** — a stalled sim stops ``/clock``.
* **Compute waits** — GPU/network latency does not scale with sim rate.
* **Publish throttles** — a load question.

Off sim time this returns ``time.monotonic()`` (the system clock is not
monotonic).
"""
from __future__ import annotations

import time


class TaskClock:
    """Seconds on the clock the robot's task runs on.

    Call it to read the clock::

        self._clock = TaskClock(node)
        elapsed = self._clock() - start

    ``sim_time`` is read once at construction; flipping ``use_sim_time`` on a
    live node is not supported.
    """

    def __init__(self, node, *, sim_time: bool | None = None) -> None:
        if sim_time is None:
            sim_time = bool(
                node.get_parameter("use_sim_time").value
                if node.has_parameter("use_sim_time")
                else False
            )
        self._node = node
        self._sim_time = bool(sim_time)

    @property
    def sim_time(self) -> bool:
        """Whether this clock follows the simulation rather than the wall."""
        return self._sim_time

    def __call__(self) -> float:
        if not self._sim_time:
            return time.monotonic()
        return self._node.get_clock().now().nanoseconds * 1e-9

    def sleep_until(self, deadline: float, stall_s: float = 5.0) -> bool:
        """Block until this clock reaches ``deadline``. False if it stopped first.

        Paces work at a fixed rate in robot time (e.g. VLA inference), so
        chunks per robot-second are independent of the real-time factor.

        Off sim time this is a plain sleep. On sim time it polls and returns
        False only if the clock makes no progress for ``stall_s`` wall seconds.
        """
        if not self._sim_time:
            time.sleep(max(0.0, deadline - self()))
            return True
        last_seen = self()
        last_progress = time.monotonic()
        while True:
            now = self()
            if now >= deadline:
                return True
            if now > last_seen:
                last_seen = now
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > stall_s:
                return False
            time.sleep(0.005)

    def stalled(self) -> bool:
        """True when sim time is expected but ``/clock`` has published nothing.

        ROS time pinned at 0 means ``use_sim_time`` without a publisher; every
        duration would stay 0.
        """
        return self._sim_time and self() <= 0.0
