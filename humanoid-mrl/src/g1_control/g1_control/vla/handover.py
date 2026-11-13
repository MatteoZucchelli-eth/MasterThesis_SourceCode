# MIT License
"""Gate for handing the arm from the driving model to the close-range detector.

:class:`FineEstimateFresh` requires a recent detector estimate (visibility is
already gated by ``perception_min_visibility``). :class:`HandoverGate` dedupes
by sequence number, debounces, latches, and optionally locks out after a
failure. Conditions are pure functions of a :class:`HandoverEvidence` snapshot.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandoverEvidence:
    """Evidence sampled on one control tick.

    ``fine`` is the detector's ``(pred_9d, anchor_9d, stamp, seq)`` tuple;
    ``fine_age_s`` comes from :class:`SeqFreshness` on the caller's clock.
    """

    fine: tuple | None = None
    fine_age_s: float = float("inf")
    now: float = 0.0


@dataclass(frozen=True)
class Verdict:
    """A condition's answer, carrying why when the answer is no."""

    passed: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "Verdict":
        return cls(True)

    @classmethod
    def no(cls, reason: str) -> "Verdict":
        return cls(False, reason)


class HandoverCondition(ABC):
    """One question asked of the evidence."""

    def reset(self) -> None:
        """Drop any history, for conditions that keep some."""

    @abstractmethod
    def evaluate(self, evidence: HandoverEvidence) -> Verdict:
        """Whether this condition is satisfied, and why not if it is not."""


class FineEstimateFresh(HandoverCondition):
    """The detector produced a reading within ``max_age_s``."""

    def __init__(self, max_age_s: float) -> None:
        self._max_age_s = float(max_age_s)

    def evaluate(self, evidence: HandoverEvidence) -> Verdict:
        if evidence.fine is None or not np.isfinite(evidence.fine_age_s):
            return Verdict.no("the close-range source has produced no estimate")
        if evidence.fine_age_s > self._max_age_s:
            return Verdict.no(
                f"its newest estimate is {evidence.fine_age_s:.1f} s old "
                f"(need <= {self._max_age_s:.1f})"
            )
        return Verdict.ok()


class SeqFreshness:
    """Age of a sequence-numbered slot's newest value, on the caller's clock.

    The tuple's own stamp is wall time and differs from robot time under sim.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_seq: int | None = None
        self._last_change: float | None = None

    def update(self, anchored: tuple | None, now: float) -> float:
        """Fold in the newest slot contents; return its age in seconds."""
        if anchored is not None and anchored[3] != self._last_seq:
            self._last_seq = anchored[3]
            self._last_change = now
        return float("inf") if self._last_change is None else now - self._last_change


@dataclass
class GateStatus:
    """What the gate did with the newest detection, for logging."""

    streak: int = 0
    locked_for_s: float = 0.0
    reason: str = ""


class HandoverGate:
    """Debounced, latching trigger over a condition.

    Judges only fresh detections (deduped by sequence number) and fires after
    ``consecutive`` passes in a row. With ``rearm_s``, a failure starts a
    lockout during which passes don't count.
    """

    def __init__(
        self,
        condition: HandoverCondition,
        consecutive: int,
        rearm_s: float = 0.0,
    ) -> None:
        self._condition = condition
        self._consecutive = max(1, int(consecutive))
        self._rearm_s = max(0.0, float(rearm_s))
        self.reset()

    def reset(self) -> None:
        self._condition.reset()
        self._streak = 0
        self._fired = False
        self._last_seq: int | None = None
        self._locked_until: float | None = None
        self._status = GateStatus()

    @property
    def fired(self) -> bool:
        return self._fired

    @property
    def status(self) -> GateStatus:
        """State after the last *judged* detection (not every poll)."""
        return self._status

    @property
    def reason(self) -> str:
        """Why the newest judged detection did not fire the gate."""
        return self._status.reason

    def update(self, evidence: HandoverEvidence) -> bool:
        """Fold in this tick's evidence; return the fired state."""
        now = evidence.now
        if self._fired or evidence.fine is None:
            return self._fired
        seq = evidence.fine[3]
        if seq == self._last_seq:
            return self._fired
        self._last_seq = seq

        verdict = self._condition.evaluate(evidence)
        locked_for = self._lockout_remaining(now)
        if not verdict.passed:
            self._streak = 0
            self._locked_until = now + self._rearm_s
            self._status = GateStatus(
                streak=0, locked_for_s=self._rearm_s, reason=verdict.reason
            )
            return False

        if locked_for > 0.0:
            self._status = GateStatus(
                streak=0,
                locked_for_s=locked_for,
                reason=f"conditions met, but locked out for another {locked_for:.1f} s",
            )
            return False

        self._streak += 1
        self._fired = self._streak >= self._consecutive
        self._status = GateStatus(
            streak=self._streak,
            reason=(
                ""
                if self._fired
                else f"conditions met {self._streak}/{self._consecutive} times in a row"
            ),
        )
        return self._fired

    def _lockout_remaining(self, now: float) -> float:
        if self._locked_until is None:
            return 0.0
        return max(0.0, self._locked_until - now)
