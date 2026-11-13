# MIT License
"""When the evaluation sequencer ends a rollout and moves to the next scene.

``rollout_s`` is a budget: a rollout ends early when the planner reports
completion, and a completion latched by one rollout must not end the next.
"""

import json
from types import SimpleNamespace

import pytest
from g1_definitions import definitions

rclpy = pytest.importorskip("rclpy")

from g1_control.nodes import vla_eval_bringup_node as bringup  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class _Publisher:
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

    def stalled(self):
        return False


@pytest.fixture
def node(monkeypatch, tmp_path):
    sim = _Clock()
    wall = _Clock()
    monkeypatch.setattr(bringup.time, "monotonic", wall)
    monkeypatch.setattr(bringup, "TaskClock", lambda _node: sim)
    node = bringup.VlaEvalBringupNode()
    node._timer.cancel()  # the test drives _tick by hand
    node._fsm_pub = _Publisher()
    node._advance_pub = _Publisher()
    node._task_pub = _Publisher()
    node._output_dir = str(tmp_path)
    # rollout_state is read once in __init__; only VLA_PLANNER reports completion.
    node._rollout_trigger = "vla_planner"
    node._rollout_fsm = definitions.FSMState.VLA_PLANNER
    node._lowstate_cb(SimpleNamespace())
    try:
        yield SimpleNamespace(node=node, clock=sim, wall=wall)
    finally:
        node.destroy_node()


def _run(fixture, state, *, after=0.0, ticks=1):
    """Wait ``after`` seconds, report ``state`` as the FSM state, then tick."""
    node = fixture.node
    for _ in range(ticks):
        fixture.clock.t += after
        fixture.wall.t += after
        node._fsm_cb(SimpleNamespace(data=int(state)))
        node._tick()


def _reach_rollout(fixture):
    """Drive the sequence up to a confirmed, running first rollout."""
    node = fixture.node
    fsm = definitions.FSMState
    retry_s = node._retry_s

    _run(fixture, fsm.ZERO_TORQUE.value)  # waiting_sim -> waiting_scene
    node._scenario_cb(
        SimpleNamespace(
            data=json.dumps(
                {"scenario": 0, "total": 2, "instruction": "drill the green circle"}
            )
        )
    )
    assert node._stage == "scene_settle"

    scene_s = float(node.get_parameter("scene_settle_s").value)
    _run(fixture, fsm.ZERO_TORQUE.value, after=scene_s + 0.1)
    assert node._stage == "damping"

    damping_s = float(node.get_parameter("damping_settle_s").value)
    _run(fixture, fsm.DAMPING.value, after=retry_s)
    _run(fixture, fsm.DAMPING.value, after=damping_s + 0.1)
    assert node._stage == "ready"

    ready_s = float(node.get_parameter("ready_settle_s").value)
    _run(fixture, fsm.READY.value, after=retry_s)
    _run(fixture, fsm.READY.value, after=ready_s + 0.1)
    assert node._stage == "vla"

    # Confirm the rollout state so _elapsed() is measured from the handover,
    # the same reset the node applies before it starts counting the budget.
    _run(fixture, fsm.VLA_PLANNER.value, after=retry_s)
    assert node._stage == "vla"
    return node


def test_a_completed_rollout_ends_without_spending_the_rest_of_the_budget(node):
    n = _reach_rollout(node)
    rollout_s = float(n.get_parameter("rollout_s").value)

    _run(node, definitions.FSMState.VLA_PLANNER.value, after=rollout_s / 4.0)
    assert n._stage == "vla", "nothing has reported completion yet"

    n._complete_cb(SimpleNamespace(data=True))
    _run(node, definitions.FSMState.VLA_PLANNER.value)
    assert n._stage == "return_ready"


def test_without_a_completion_the_rollout_still_runs_its_full_budget(node):
    n = _reach_rollout(node)
    rollout_s = float(n.get_parameter("rollout_s").value)

    _run(node, definitions.FSMState.VLA_PLANNER.value, after=rollout_s - 0.1)
    assert n._stage == "vla"

    _run(node, definitions.FSMState.VLA_PLANNER.value, after=0.2)
    assert n._stage == "return_ready"


def test_stop_on_complete_false_ignores_the_flag(node):
    """The escape hatch for a run that wants a fixed hold window in every CSV."""
    n = _reach_rollout(node)
    n.set_parameters([rclpy.parameter.Parameter("stop_on_complete", value=False)])

    n._complete_cb(SimpleNamespace(data=True))
    _run(node, definitions.FSMState.VLA_PLANNER.value, after=1.0)
    assert n._stage == "vla"


def test_a_previous_rollouts_completion_does_not_end_the_next_one(node):
    """The flag is latched; the node clears its copy on every stage change, so
    a True only counts if it arrives after this rollout began.
    """
    n = _reach_rollout(node)
    fsm = definitions.FSMState
    retry_s = n._retry_s

    n._complete_cb(SimpleNamespace(data=True))
    _run(node, fsm.VLA_PLANNER.value)
    assert n._stage == "return_ready"

    ready_s = float(n.get_parameter("ready_settle_s").value)
    _run(node, fsm.READY.value, after=retry_s)
    _run(node, fsm.READY.value, after=ready_s + 0.1)
    assert n._stage == "between_rollouts"
    assert n._completed == 1

    between_s = float(n.get_parameter("between_rollouts_s").value)
    _run(node, fsm.READY.value, after=between_s + 0.1)
    assert n._stage == "waiting_scene"

    node.node._scenario_cb(
        SimpleNamespace(
            data=json.dumps(
                {"scenario": 1, "total": 2, "instruction": "drill the red square"}
            )
        )
    )
    scene_s = float(n.get_parameter("scene_settle_s").value)
    _run(node, fsm.READY.value, after=scene_s + 0.1)
    # Later scenarios skip the DAMPING bringup and go straight back to VLA.
    assert n._stage == "vla"

    _run(node, fsm.VLA_PLANNER.value, after=retry_s)
    _run(node, fsm.VLA_PLANNER.value, after=1.0)
    assert n._stage == "vla", "the stale completion must not end this rollout"
