# MIT License
"""Advisory finite state machine for the G1 operating states.

Decodes RC buttons and ``TOPIC_FSM_TRIGGER`` requests into transitions and
broadcasts the current state as UInt8 on ``TOPIC_FSMSTATE``. State behaviour
lives in g1_control.
"""
from g1_common.rc import parse_buttons
from g1_definitions.definitions import (
    FSMState,
    TOPIC_FSM_TRIGGER,
    TOPIC_FSMSTATE,
    TOPIC_LOWSTATE,
)
from rclpy.node import Node
from std_msgs.msg import String, UInt8
from transitions import Machine, MachineError

from unitree_hg.msg import LowState

from g1_state_machine.utils import get_fsm_state_from_buttons, print_transition_info

_WHOLE_BODY = [
    FSMState.READY,
    FSMState.MOVE,
    FSMState.VLA,
    FSMState.TELEOP,
    FSMState.GOAL_POSE,
    FSMState.AUTO_COLLECT,
    FSMState.VLA_PLANNER,
    FSMState.FK_CALIBRATION,
]

# (trigger, allowed source states, destination). dest=None (docs) fires the
# after-callback without changing state.
_TRANSITIONS = [
    ("damping", [FSMState.ZERO_TORQUE, *_WHOLE_BODY], FSMState.DAMPING),
    ("ready", [FSMState.DAMPING, *_WHOLE_BODY], FSMState.READY),
    ("move", _WHOLE_BODY, FSMState.MOVE),
    ("vla", _WHOLE_BODY, FSMState.VLA),
    ("teleoperation", _WHOLE_BODY, FSMState.TELEOP),
    # Drive the right palm via IK to an external 3D goal pose.
    ("goal_pose", _WHOLE_BODY, FSMState.GOAL_POSE),
    # Scripted dataset collection onto the red target, phases on
    # TOPIC_AUTO_COLLECT_PHASE.
    ("auto_collect", _WHOLE_BODY, FSMState.AUTO_COLLECT),
    # VLA approach, stop to accumulate target detections, then a planned
    # trajectory drives the tip onto the estimate.
    ("vla_planner", _WHOLE_BODY, FSMState.VLA_PLANNER),
    # Drive the drill tip to an AprilTag measured by the wrist camera (FK calibration).
    ("fk_calibration", _WHOLE_BODY, FSMState.FK_CALIBRATION),
    (
        "docs",
        [
            FSMState.ZERO_TORQUE,
            FSMState.DAMPING,
            FSMState.READY,
            FSMState.MOVE,
            FSMState.VLA,
            FSMState.TELEOP,
        ],
        None,
    ),
]

# Keep-alive republish period for late joiners; changes publish immediately.
_KEEPALIVE_PERIOD_S = 0.1


class G1FSM(Node):
    def __init__(self):
        super().__init__(
            "finite_state_machine", automatically_declare_parameters_from_overrides=True
        )

        config_objects = self.get_parameters_by_prefix("transition_rc_mapping")
        self.config = {name: param.value for name, param in config_objects.items()}
        self.rc_buttons = None
        self.last_state = None

        self.machine = Machine(
            model=self,
            states=list(FSMState),
            initial=FSMState.ZERO_TORQUE,
            after_state_change="_publish_state",
        )
        for trigger, sources, dest in _TRANSITIONS:
            # No source-state guard for docs: it never changes state.
            after = "_print_docs" if trigger == "docs" else None
            self.machine.add_transition(trigger, sources, dest, after=after)

        self.state_pub = self.create_publisher(UInt8, TOPIC_FSMSTATE, 1)

        self.low_state_sub = self.create_subscription(
            LowState, TOPIC_LOWSTATE, self._low_state_callback, 1
        )
        # External transition requests (trigger name), applied like an RC button.
        self.trigger_sub = self.create_subscription(
            String, TOPIC_FSM_TRIGGER, self._trigger_callback, 1
        )
        self.timer = self.create_timer(_KEEPALIVE_PERIOD_S, self._publish_state)

    def _print_docs(self):
        print_transition_info(self.config, self.get_logger())

    def _publish_state(self):
        self.state_pub.publish(UInt8(data=int(self.state.value)))

    def _low_state_callback(self, msg: LowState):
        """Apply RC-button transitions decoded from the wireless remote."""
        self.rc_buttons = parse_buttons(msg.wireless_remote)
        rc_input_state = get_fsm_state_from_buttons(self.rc_buttons, self.config)

        if rc_input_state and self.last_state != rc_input_state:
            self.update_fsm(rc_input_state)
            self.get_logger().info(f"FSM state: {self.state}")
            self.last_state = rc_input_state

    def _trigger_callback(self, msg: String):
        """Apply an externally requested transition (same path as an RC button)."""
        trigger = msg.data
        if not trigger:
            return
        self.update_fsm(trigger)
        self.get_logger().info(
            f"External trigger '{trigger}' -> FSM state: {self.state}"
        )

    def update_fsm(self, trigger: str):
        """Fire a transition by name, logging (not raising) invalid requests."""
        try:
            getattr(self, trigger)()
        except (MachineError, AttributeError) as e:
            self.get_logger().info(
                f"Invalid transition, check state machine structure. Error: {e}"
            )
            self.get_logger().info(f"Current state: {self.state}, {trigger}")
