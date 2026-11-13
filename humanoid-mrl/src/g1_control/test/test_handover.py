# MIT License
"""Unit tests for the handover conditions and gate (numpy only, no ROS)."""

import numpy as np

from g1_control.vla.handover import (
    FineEstimateFresh,
    HandoverCondition,
    HandoverEvidence,
    HandoverGate,
    SeqFreshness,
    Verdict,
)


def _pose_9d(position, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    position = np.asarray(position, dtype=float)
    return np.concatenate([position, rotation[:, 0], rotation[:, 1]])


def _anchored(pred_position, seq=1, anchor_position=(0.0, 0.0, 0.0)):
    return (_pose_9d(pred_position), _pose_9d(anchor_position), 0.0, seq)


class _Switchable(HandoverCondition):
    """A condition whose answer the test can flip, to model one that flickers."""

    def __init__(self, passing=True):
        self.passing = passing

    def evaluate(self, evidence):
        return Verdict.ok() if self.passing else Verdict.no("blocked")


def _pass():
    class _P(HandoverCondition):
        def evaluate(self, evidence):
            return Verdict.ok()

    return _P()


def _fail(reason="nope"):
    class _F(HandoverCondition):
        def evaluate(self, evidence):
            return Verdict.no(reason)

    return _F()


# ── The detector is alive and confident ──────────────────────────────────────


def test_freshness_condition_separates_absent_from_stale():
    condition = FineEstimateFresh(2.0)
    assert "no estimate" in condition.evaluate(HandoverEvidence()).reason
    stale = HandoverEvidence(fine=_anchored([0.05, 0.0, 0.0]), fine_age_s=5.0)
    assert "5.0 s old" in condition.evaluate(stale).reason
    fresh = HandoverEvidence(fine=_anchored([0.05, 0.0, 0.0]), fine_age_s=0.5)
    assert condition.evaluate(fresh).passed


# ── SeqFreshness ─────────────────────────────────────────────────────────────


def test_freshness_ages_on_the_callers_clock_not_the_tuples_stamp():
    freshness = SeqFreshness()
    assert freshness.update(None, 0.0) == float("inf")
    assert freshness.update(_anchored([0.1, 0.0, 0.0], seq=1), 10.0) == 0.0
    # Same prediction re-polled: it is not getting any fresher.
    assert freshness.update(_anchored([0.1, 0.0, 0.0], seq=1), 12.5) == 2.5
    assert freshness.update(_anchored([0.1, 0.0, 0.0], seq=2), 13.0) == 0.0


# ── HandoverGate ─────────────────────────────────────────────────────────────


def _evidence(seq, now=0.0):
    return HandoverEvidence(fine=_anchored([0.04, 0.0, 0.0], seq=seq), now=now)


def test_gate_fires_after_consecutive_passing_detections():
    gate = HandoverGate(_pass(), consecutive=3)
    assert not gate.update(_evidence(1))
    assert not gate.update(_evidence(2))
    assert gate.update(_evidence(3))
    assert gate.fired


def test_gate_judges_once_per_detection_however_often_it_is_polled():
    gate = HandoverGate(_pass(), consecutive=2)
    for _ in range(10):
        assert not gate.update(_evidence(1))  # same inference, polled at 50 Hz
    assert gate.update(_evidence(2))


def test_gate_reports_why_it_is_blocked():
    gate = HandoverGate(_fail("the detector is stale"), consecutive=1)
    assert not gate.update(_evidence(1))
    assert "the detector is stale" in gate.reason


def test_gate_lockout_stops_a_flickering_condition_chattering():
    """A condition that recovers must re-prove itself, not fire instantly."""
    condition = _Switchable(passing=False)
    # consecutive=1, so without the lockout the first pass after the failure
    # would fire the gate immediately.
    gate = HandoverGate(condition, consecutive=1, rearm_s=5.0)
    assert not gate.update(_evidence(1, now=0.0))

    condition.passing = True
    assert not gate.update(_evidence(2, now=2.0))  # still locked out
    assert "locked out" in gate.reason
    assert gate.update(_evidence(3, now=6.0))  # lockout expired


def test_gate_latches_and_resets():
    gate = HandoverGate(_pass(), consecutive=1)
    assert gate.update(_evidence(1))
    assert gate.update(_evidence(2))  # stays fired without re-judging
    gate.reset()
    assert not gate.fired and gate.reason == ""


def test_gate_ignores_ticks_with_no_detection_at_all():
    gate = HandoverGate(_pass(), consecutive=1)
    assert not gate.update(HandoverEvidence(fine=None))
    assert not gate.fired


def test_gate_debounces_on_the_detectors_sequence():
    gate = HandoverGate(_pass(), consecutive=2)
    repeat = HandoverEvidence(fine=_anchored([0.05, 0, 0], seq=7))
    assert not gate.update(repeat)
    assert not gate.update(repeat)  # same reading, not a second opinion
    assert gate.update(HandoverEvidence(fine=_anchored([0.05, 0, 0], seq=8)))
