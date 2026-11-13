# MIT License
"""In-memory episode state and acceptance policy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeCapture:
    """Own one recorder transaction, independent of ROS and disk I/O."""

    frames: list[dict] = field(default_factory=list)
    image_lags: list[float] = field(default_factory=list)
    clamped_frames: int = 0
    camera_sync_fault: bool = False
    worst_camera_skew: float = 0.0
    # Why the last sampling tick produced no row; reported if the episode ends empty.
    blocked_reason: str | None = None
    recording: bool = False
    next_frame_stamp: float | None = None
    last_camera_stamps: dict[str, float | None] = field(default_factory=dict)

    def start(self, camera_stamps: dict[str, float | None]) -> None:
        """Open a clean transaction after the currently cached camera frames."""
        self.discard()
        self.recording = True
        self.last_camera_stamps = dict(camera_stamps)

    def freeze(self) -> None:
        """Stop sampling while retaining the transaction for save or discard."""
        self.recording = False

    def discard(self) -> None:
        """Reset the transaction without returning frames for persistence."""
        self.frames.clear()
        self.image_lags.clear()
        self.clamped_frames = 0
        self.camera_sync_fault = False
        self.worst_camera_skew = 0.0
        self.blocked_reason = None
        self.recording = False
        self.next_frame_stamp = None
        self.last_camera_stamps.clear()

    def cameras_ready(self, stamps: dict[str, float], max_skew_s: float) -> bool:
        """Accept only fresh camera frames that belong to the same instant."""
        # Stale frames are unavailable, not skewed; check freshness first.
        if not self.cameras_fresh(stamps):
            return False
        skew = max(stamps.values()) - min(stamps.values())
        if skew > max_skew_s:
            # Caller drops this row; RecorderNode advances the row clock.
            return False
        self.worst_camera_skew = max(self.worst_camera_skew, skew)
        return True

    def cameras_fresh(self, stamps: dict[str, float]) -> bool:
        """Whether every selected frame is newer than the previous row/start."""
        return all(
            previous is None or stamps[key] > previous
            for key, previous in self.last_camera_stamps.items()
        )

    def skip_row(self, period_s: float) -> None:
        """Give up on the row currently due and aim at the next instant."""
        self.next_frame_stamp += period_s

    def frame_due(self, stamp: float, period_s: float) -> bool:
        """Return whether the image clock has reached the next sample."""
        if self.next_frame_stamp is None:
            self.next_frame_stamp = stamp
        return stamp >= self.next_frame_stamp

    def append(
        self,
        frame: dict,
        *,
        lag_s: float,
        camera_stamps: dict[str, float],
        reference_stamp: float,
        period_s: float,
    ) -> None:
        """Append a synchronized row and advance its drift-free due time."""
        self.frames.append(frame)
        self.image_lags.append(lag_s)
        self.blocked_reason = None
        self.last_camera_stamps = dict(camera_stamps)
        self.next_frame_stamp += period_s
        if self.next_frame_stamp < reference_stamp - period_s:
            self.next_frame_stamp = reference_stamp

    def failures(self, *, stale_s: float, skew_s: float) -> list[str]:
        """Describe hard data-integrity failures that forbid persistence."""
        if not self.frames:
            reason = self.blocked_reason or "the sampler never ran"
            return [f"no synchronized frames ({reason})"]

        failures = []
        if self.camera_sync_fault:
            failures.append(
                f"cross-camera skew reached {self.worst_camera_skew:.3f}s "
                f"(limit {skew_s:.3f}s)"
            )
        if self.clamped_frames:
            failures.append(
                f"{self.clamped_frames}/{len(self.frames)} frames had no bracketing state"
            )
        if self.image_lags and max(self.image_lags) > stale_s:
            failures.append(
                f"image lag reached {max(self.image_lags):.2f}s (limit {stale_s:.2f}s)"
            )
        if any(np.isnan(frame["tip_target_dist"]) for frame in self.frames):
            failures.append("one or more frames have no tip-to-target label")
        return failures
