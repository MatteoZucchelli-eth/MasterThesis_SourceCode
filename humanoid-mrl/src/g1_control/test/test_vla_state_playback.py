# MIT License
"""VLA state chunk playback against the real VLAActionManager, with fake clients.

The coarse role drives for the whole session; the fine detector only supplies
the estimate.
"""

import types

import numpy as np

from g1_control.states.vla import VlaHandler
from g1_control.vla.sources import VlaRole, VlaSources
from g1_control.vla_action_manager import VLAActionManager

N_JOINTS = 21
CHUNK = 30


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.running = False
        self.start_count = 0
        self._anchored = None

    def start_inference_loop(self) -> None:
        self.running = True
        self.start_count += 1

    def pause_inference_loop(self) -> None:
        self.running = False

    def get_tip_target_anchored(self):
        return self._anchored

    def get_tip_target(self):
        return None if self._anchored is None else self._anchored[0]

    def get_prev_tip_chunk(self):
        return None

    def predict(self, distance: float, seq: int, stamp: float = 0.0) -> None:
        """Publish a tip->target estimate ``distance`` metres straight ahead."""
        pred = np.array([distance, 0.0, 0.0, 1, 0, 0, 0, 1, 0], dtype=float)
        anchor = np.array([0.0, 0.0, 0.0, 1, 0, 0, 0, 1, 0], dtype=float)
        self._anchored = (pred, anchor, stamp, seq)


class _FakeMetrics:
    def __init__(self) -> None:
        self.source = None

    def set_estimate_source(self, client) -> None:
        self.source = client

    def start(self, **kwargs) -> None: ...

    def record(self, *args, **kwargs) -> None: ...

    def finalize(self) -> None: ...


class _FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg.data)


class _FakeNode:
    def __init__(self) -> None:
        self.publisher = _FakePublisher()

    def get_logger(self):
        return types.SimpleNamespace(
            info=lambda *a, **k: None,
            warn=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )

    def has_parameter(self, name: str) -> bool:
        return False

    def create_publisher(self, *args, **kwargs):
        return self.publisher


def _context():
    return types.SimpleNamespace(
        control_dt=0.02,
        full_q=np.zeros(40),
        upper_body_indices=list(range(N_JOINTS)),
        robot_state=types.SimpleNamespace(
            upper_body_target=None, upper_body_target_chunk=None
        ),
        targets=types.SimpleNamespace(left_9d=None, right_9d=None, clear=lambda: None),
    )


def _role(name: str) -> VlaRole:
    return VlaRole(
        _FakeClient(name),
        VLAActionManager(chunk_size=CHUNK, num_upper_body_joints=N_JOINTS),
    )


def _deliver_chunk(role: VlaRole, value: float) -> None:
    """Simulate an inference thread landing a chunk in the role's buffer."""
    with role.actions.action_lock:
        for array in role.actions.action_buffer.values():
            array.fill(value)
    role.actions.new_action_chunk.set()


def _build(sources: VlaSources):
    node = _FakeNode()
    ctx = _context()
    metrics = _FakeMetrics()
    handler = VlaHandler(
        node,
        ctx,
        sources,
        metrics,
        types.SimpleNamespace(enabled=False, reset=lambda: None),
        types.SimpleNamespace(publish=lambda *a, **k: None),
    )
    return handler, ctx, node, metrics


# ── Playback ─────────────────────────────────────────────────────────────────


def _drive(mode_sources: VlaSources):
    handler, ctx, _node, metrics = _build(mode_sources)
    handler.on_enter(previous_state=0)
    return handler, ctx, metrics


def test_the_coarse_model_alone_drives():
    """The coarse model keeps the arm however close the detector says the target is."""
    coarse = _role("coarse")
    detector = _FakeClient("detector")
    handler, ctx, metrics = _drive(
        VlaSources(coarse=coarse, fine=VlaRole(detector))
    )

    assert metrics.source is coarse.client
    _deliver_chunk(coarse, 0.5)
    handler.update()

    for i in range(5):
        coarse.client.predict(0.001, seq=i + 1)
        handler.update()

    assert coarse.client.running
    assert coarse.client.start_count == 1
    assert np.allclose(ctx.robot_state.upper_body_target_chunk, 0.5)


def test_only_the_coarse_client_is_started_but_both_are_paused_on_exit():
    """Only the coarse client is started here, but on_exit pauses every client."""
    coarse = _role("coarse")
    detector = _FakeClient("detector")
    handler, _ctx, _metrics = _drive(
        VlaSources(coarse=coarse, fine=VlaRole(detector))
    )

    assert coarse.client.running
    assert not detector.running

    handler.on_exit()
    assert not coarse.client.running
    assert not detector.running


def test_a_chunk_pending_at_exit_is_not_delivered_to_the_next_session():
    """on_enter resets the buffer, so a chunk landed before the pause is not
    replayed."""
    coarse = _role("coarse")
    handler, ctx, _metrics = _drive(
        VlaSources(coarse=coarse, fine=VlaRole(_FakeClient("detector")))
    )
    _deliver_chunk(coarse, 0.5)  # lands after the last update() of the session
    handler.on_exit()

    handler.on_enter(previous_state=0)
    handler.update()
    assert ctx.robot_state.upper_body_target_chunk is None
