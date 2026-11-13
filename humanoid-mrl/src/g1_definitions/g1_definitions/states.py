# MIT License
"""Operating-state enums shared across the g1 packages."""

from enum import Enum


class FSMState(Enum):
    ZERO_TORQUE = 0
    DAMPING = 1
    MOVE = 2
    READY = 3
    VLA = 5
    TELEOP = 6
    GOAL_POSE = 7
    AUTO_COLLECT = 8
    VLA_PLANNER = 9
    FK_CALIBRATION = 11


class AutoCollectPhase(Enum):
    """Phases of the scripted AUTO_COLLECT dataset-collection state.

    Driven externally over :data:`TOPIC_AUTO_COLLECT_PHASE` so a recorder script
    can sequence an episode. ``APPROACHING`` and ``GOING_TO_DEFAULT`` fall back
    to ``HOLDING`` on their own once their motion completes.
    """

    TRAJECTORY_GENERATION = 0
    """Plan a drill-tip trajectory from the current pose to the red target."""

    APPROACHING = 1
    """Execute the planned trajectory, driving the tip onto the target."""

    GOING_TO_DEFAULT = 2
    """Interpolate the upper body back to its default pose."""

    HOLDING = 3
    """Hold the last commanded upper-body target."""

    @classmethod
    def parse(cls, text: str) -> "AutoCollectPhase":
        """Resolve a phase from its (case-insensitive) name or its numeric value."""
        key = text.strip()
        try:
            return cls[key.upper()]
        except KeyError:
            pass
        return cls(int(key))


class VlaPlannerPhase(Enum):
    """Substates of the VLA_PLANNER state (perceive -> stop -> plan -> execute).

    The state advances through them on its own; the current phase is echoed
    (latched) on :data:`TOPIC_VLA_PLANNER_PHASE_STATUS` so eval tooling can
    segment a session.
    """

    APPROACHING = 0
    """VLA action chunks drive the arm while the model's own tip->target
    distance estimate is still above the approach threshold."""

    ACCUMULATING = 1
    """The arm holds still while tip->target predictions are accumulated into
    a converged target-pose estimate."""

    FINALIZING = 2
    """The straight push drives the tip from the re-measured standoff pose
    onto the estimated target pose. It ends the run."""

    HOLDING = 3
    """Terminal: hold the last commanded pose (after success or a give-up).

    Values 4 and 5 are reserved: recorded sessions' ``phase`` columns use them."""

    RECOVERING = 6
    """Drive back to the default pose, then restart ``APPROACHING``.

    Entered when the standoff batch is beyond the handover distance (perception
    far mode) or the approach times out."""

    STANDOFF = 7
    """A planned transit drives the tip to the standoff knot, where the arm stops
    and re-accumulates before ``FINALIZING``."""

    VISUAL_SERVOING = 8
    """Hold depth while wrist-camera feedback drives the drill-tip pixel onto
    the target pixel, at the standoff and before the push."""
