# MIT License
"""The headless bringup sequence, and which clock each of its waits runs on.

The node walks ``damping -> ready -> auto_collect``. Settle holds run on sim
time; the trigger retry runs on the wall clock.
"""

from types import SimpleNamespace

import pytest
from g1_definitions import definitions

rclpy = pytest.importorskip("rclpy")

from g1_control.nodes import auto_collect_bringup_node as bringup  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class _Publisher:
    """Stand-in for a rclpy publisher that just records what it was handed."""

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Clock:
    """A clock the test advances by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    @property
    def sim_time(self):
        return True

    def stalled(self):
        return False


@pytest.fixture
def node(monkeypatch):
    # Separate sim and wall clocks.
    sim = _Clock()
    wall = _Clock()
    monkeypatch.setattr(bringup.time, "monotonic", wall)
    monkeypatch.setattr(bringup, "TaskClock", lambda _node: sim)
    node = bringup.AutoCollectBringupNode()
    node._trigger_pub = _Publisher()
    node._reset_request_pub = _Publisher()
    # Mark the sim alive; the sequence idles until lowstate arrives.
    node._lowstate_cb(SimpleNamespace())
    try:
        yield SimpleNamespace(node=node, clock=sim, wall=wall)
    finally:
        node.destroy_node()


def _run(fixture, state, *, after=0.0, ticks=1):
    """Advance both clocks by ``after`` s, report ``state``, then tick (rtf 1)."""
    node = fixture.node
    for _ in range(ticks):
        fixture.clock.t += after
        fixture.wall.t += after
        node._fsm_cb(SimpleNamespace(data=int(state)))
        node._tick()


def _triggers(node):
    return [msg.data for msg in node._trigger_pub.messages]


def test_the_sequence_is_damping_then_ready_then_auto_collect(node):
    assert [trigger for trigger, _, _ in node.node._stages] == [
        "damping",
        "ready",
        "auto_collect",
    ]


def test_each_stage_is_held_for_its_settle_before_the_next_trigger(node):
    """The arm is handed over from a settled pose, not a moving one."""
    n = node.node
    damping_s = float(n.get_parameter("damping_settle_s").value)
    ready_s = float(n.get_parameter("ready_settle_s").value)
    retry_s = n._retry_s

    _run(node, definitions.FSMState.ZERO_TORQUE.value)
    assert _triggers(n) == ["damping"]

    # DAMPING confirmed, but not yet settled: no READY trigger.
    _run(node, definitions.FSMState.DAMPING.value)
    _run(node, definitions.FSMState.DAMPING.value, after=damping_s - 0.1)
    assert _triggers(n) == ["damping"]

    # Past the settle; the trigger goes out once the wall-clock retry gate clears.
    _run(node, definitions.FSMState.DAMPING.value, after=0.2)
    _run(node, definitions.FSMState.DAMPING.value, after=retry_s)
    assert _triggers(n)[-1] == "ready"

    # Same for the pose interpolation: AUTO_COLLECT waits it out.
    _run(node, definitions.FSMState.READY.value)
    _run(node, definitions.FSMState.READY.value, after=ready_s - 0.1)
    assert "auto_collect" not in _triggers(n)

    _run(node, definitions.FSMState.READY.value, after=0.2)
    _run(node, definitions.FSMState.READY.value, after=retry_s)
    assert _triggers(n)[-1] == "auto_collect"


def test_settle_times_are_the_robots_seconds_not_the_walls(node):
    """Settle times follow sim time, not ``time.monotonic()``.

    The sim clock is frozen while the wall clock runs (rtf 0); no stage may settle.
    """
    n = node.node

    def wall_only(state, seconds, ticks=5):
        for _ in range(ticks):
            node.wall.t += seconds / ticks
            n._fsm_cb(SimpleNamespace(data=int(state)))
            n._tick()

    _run(node, definitions.FSMState.DAMPING.value)
    damping_stage = n._stage_idx
    wall_only(definitions.FSMState.DAMPING.value, 3600.0)
    assert n._stage_idx == damping_stage, "DAMPING settled on wall seconds"

    _run(
        node,
        definitions.FSMState.DAMPING.value,
        after=float(n.get_parameter("damping_settle_s").value),
    )
    _run(node, definitions.FSMState.READY.value)
    ready_stage = n._stage_idx
    wall_only(definitions.FSMState.READY.value, 3600.0)
    assert n._stage_idx == ready_stage, "READY settled on wall seconds"
    assert "auto_collect" not in _triggers(n)

    # The same duration on the robot's own clock does advance it.
    _run(
        node,
        definitions.FSMState.READY.value,
        after=float(n.get_parameter("ready_settle_s").value),
    )
    _run(node, definitions.FSMState.READY.value)
    assert _triggers(n)[-1] == "auto_collect"


def test_the_trigger_retry_stays_on_the_wall_clock(node):
    """The retry must keep firing while ``/clock`` is stalled."""
    n = node.node
    retry_s = n._retry_s

    # ZERO_TORQUE never matches the "damping" stage, so every tick takes the
    # retry path. Sim time is frozen throughout.
    _run(node, definitions.FSMState.ZERO_TORQUE.value)
    assert _triggers(n) == ["damping"]

    for _ in range(3):
        node.wall.t += retry_s
        n._fsm_cb(SimpleNamespace(data=definitions.FSMState.ZERO_TORQUE.value))
        n._tick()

    assert _triggers(n) == ["damping"] * 4, "retry followed the frozen sim clock"


def test_a_clock_that_never_started_holds_the_sequence(node, monkeypatch):
    """``use_sim_time`` with no ``/clock`` publisher: the sequence holds."""
    n = node.node
    monkeypatch.setattr(node.clock, "stalled", lambda: True)

    _run(node, definitions.FSMState.DAMPING.value, ticks=5)

    assert _triggers(n) == []
    assert n._stage_idx == 0


def test_losing_auto_collect_restarts_the_sequence(node):
    """An external trigger or node restart drops the FSM out of AUTO_COLLECT."""
    n = node.node
    n._stage_idx = len(n._stages)

    _run(node, definitions.FSMState.ZERO_TORQUE.value)
    assert n._stage_idx == 0
    _run(node, definitions.FSMState.ZERO_TORQUE.value)
    assert _triggers(n)[-1] == "damping"
