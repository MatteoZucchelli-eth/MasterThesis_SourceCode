# MIT License
"""Dataset schema: recorded end-effector frames, state/action layout and rates."""

VLA_CHUNK_SIZE = 30
"""Number of action steps in a single VLA inference chunk"""

RECORD_FPS: int = 30
"""Dataset recording / playback frequency in Hz"""

# ── End-effector frames ───────────────────────────────────────────────────────

LEFT_PALM_FRAME = "left_hand_palm_joint"
"""Left palm frame. Also the frame XR teleoperation drives (the operator's hand)."""

RIGHT_PALM_FRAME = "right_hand_palm_joint"
"""Right palm frame. The XR teleoperation input frame; NOT the recorded eef."""

DRILL_TIP_FRAME = "drill_tip_link"
"""Drill bit tip: a fixed child of right_drill_link, only in the drill URDF variant."""

LEFT_EEF_FRAME = LEFT_PALM_FRAME
"""Left end-effector frame. No tool, so the palm itself.

Driven by teleoperation IK and drawn in RViz, but *not* recorded: the left arm
holds its default pose for the whole drilling task (see STATE_KEY_SIZES).
"""

RIGHT_EEF_FRAME = DRILL_TIP_FRAME
"""Right end-effector recorded as state/action, and the frame the right-arm IK drives.

A palm-orientation error is amplified by the ~0.20 m lever arm to the tip
(1 deg -> ~3.5 mm), so predicting the tip directly matches the loss to the task.
Teleoperation inputs palm poses; the tip is what gets recorded.
"""

RIGHT_EEF_KEY = "right_drill_tip_eef_9d"
"""Dataset state/action key for the right end-effector (drill tip) 9D pose."""

# ── State / action layout ─────────────────────────────────────────────────────

STATE_KEY_SIZES: dict[str, int] = {
    RIGHT_EEF_KEY: 9,  # FK drill tip pose: xyz (3) + rot6d (6)
    "right_arm": 7,  # shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
}
"""Ordered key→dim mapping for the 16-dim observation state vector.

Only the drilling arm is described; the left arm and waist hold their default
pose for the whole task.
"""

ACTION_KEY_SIZES: dict[str, int] = {
    RIGHT_EEF_KEY: 9,  # IK target drill tip pose: xyz (3) + rot6d (6)
}
"""Ordered key→dim mapping for the 9-dim action vector.

State and action share these keys: the fine-tuning script pairs them by name to
derive the relative action, so both must be absolute poses of the same frame.
"""


def _build_slices(key_sizes: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    idx = 0
    for k, n in key_sizes.items():
        out[k] = (idx, idx + n)
        idx += n
    return out


STATE_SLICES: dict[str, tuple[int, int]] = _build_slices(STATE_KEY_SIZES)
ACTION_SLICES: dict[str, tuple[int, int]] = _build_slices(ACTION_KEY_SIZES)
STATE_DIM: int = sum(STATE_KEY_SIZES.values())
ACTION_DIM: int = sum(ACTION_KEY_SIZES.values())
