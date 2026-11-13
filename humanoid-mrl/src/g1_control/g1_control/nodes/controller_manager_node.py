# MIT License
# Decoupled Whole-Body Policy Node for Unitree G1
from __future__ import annotations

import logging
import os

import numpy as np
import rclpy

from g1_common import CRC
from g1_common.camera import preparer_from_camera_info
from g1_common.data import RobotState
from g1_common.image_transport import subscribe_image_latest, transport_name
from g1_definitions import definitions
from g1_teleoperation import TeleoperationManager
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String, UInt8
from tf2_ros import Buffer, TransformListener
from unitree_hg.msg import LowCmd, LowState

from g1_control.controllers import (
    DampingController,
    ReadyController,
    WholeBodyRLController,
    ZeroTorqueController,
)
from g1_control.nodes.step_publisher import ControlStepPublisher
from g1_control.states import (
    AutoCollectHandler,
    ControlContext,
    FkCalibrationHandler,
    GoalPoseHandler,
    MetricRecorder,
    ProgressTestHarness,
    RcHandler,
    TeleopHandler,
    VlaDebugPublisher,
    VlaHandler,
    VlaPlannerHandler,
)
from g1_control.utils import nonfinite_motors, resolve_eef_frames
from g1_control.utils.drill_task_sampler import resolve_target_frame
from g1_control.utils.g1_supplemental_info import G1SupplementalInfo
from g1_control.utils.obs_ood_scorer import ObsOODScorer
from g1_control.utils.task_clock import TaskClock
from g1_control.utils.utils import load_config
from g1_control.utils.wbc_robot_model import RobotModel
from g1_control.vla import (
    CameraIntrinsics,
    PerceptionClient,
    TipTargetLifter,
    VLAClient,
    VlaPlannerConfig,
    VlaRole,
    VlaSources,
)
from g1_control.vla_action_manager import VLAActionManager

_FSM = definitions.FSMState


class ControllerManagerNode(Node):
    """ROS2 node running the G1 control loop.

    Owns ROS I/O, the control timer and the controller/handler registries;
    per-state behaviour lives in :mod:`g1_control.states` handlers sharing a
    :class:`~g1_control.states.ControlContext`.
    """

    def __init__(self):
        super().__init__(
            "controller_manager_node",
            automatically_declare_parameters_from_overrides=True,
        )

        self._setup_config_and_model()
        self._setup_state_and_controllers()
        self._setup_ros_interfaces()
        self._setup_context()
        self._setup_vla()
        self._setup_handlers()

        self.get_logger().info("Controller Manager Node initialized")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _setup_config_and_model(self) -> None:
        """Load YAML config and build the robot kinematics model."""
        gear_wbc_config_path = self.get_parameter("gear_wbc_config_path").value
        self._config = load_config(gear_wbc_config_path)
        experiment_path = os.environ.get("G1_CONTROLLER_EXPERIMENT")
        if experiment_path:
            self._config.update(load_config(experiment_path).get("controller", {}))
            self.get_logger().warning(
                f"Experimental controller overlay: {experiment_path}"
            )
        integral_override = float(self.get_parameter("arm_integral_ki_override").value)
        if integral_override >= 0.0:
            self._config["arm_integral_ki"] = integral_override
            self.get_logger().warning(
                f"Diagnostic override: arm_integral_ki={integral_override:g}"
            )
        self._control_dt = float(self._config["simulation_dt"]) * int(
            self._config["control_decimation"]
        )

        self.robot_model = RobotModel(
            self.get_parameter("urdf_path").value,
            self.get_parameter("asset_path").value,
            set_floating_base=False,
            supplemental_info=G1SupplementalInfo(),
        )

        # State/action EEF frames; the right one is the drill tip (task error frame).
        # The palm remains the XR input frame for teleoperation.
        self._left_eef_frame, self._right_eef_frame = resolve_eef_frames(
            self.robot_model
        )

    def _setup_state_and_controllers(self) -> None:
        """Compute default joint positions and instantiate the controllers."""
        config = self._config
        damping_array = np.array(config["motor_kds"], dtype=np.float64)
        stiffness_array = np.array(config["motor_kps"], dtype=np.float64)

        lower_body_indices = self.robot_model.get_joint_group_indices("lower_body")
        self._upper_body_indices = self.robot_model.get_joint_group_indices(
            "upper_body"
        )
        self.actuated_indices = self.robot_model.get_body_actuated_joint_indices()

        self._default_upper_body_pose = self.robot_model.get_initial_upper_body_pose()
        default_q = self._build_default_q(lower_body_indices, config)
        default_joint_pos = self.robot_model.get_body_actuated_joints(default_q)

        self.robot_state = RobotState()

        gains = self.get_parameters_by_prefix("gains")

        # One WBC instance shared by all whole-body states (handlers feed targets).
        # wbc_eval:=true pins legs/waist to q0 (no RL locomotion).
        wbc = WholeBodyRLController(
            self.robot_state,
            gains,
            robot_model=self.robot_model,
            config=config,
            balance_model_path=self.get_parameter("balance_model_path").value,
            walk_model_path=self.get_parameter("walk_model_path").value,
            eval=bool(self._param("wbc_eval", False)),
            # Node ROS clock (sim time in Isaac), matching the control timer.
            clock=lambda: self.get_clock().now().nanoseconds * 1e-9,
        )
        self.controllers = {
            _FSM.ZERO_TORQUE.value: ZeroTorqueController(self.robot_state, gains),
            _FSM.DAMPING.value: DampingController(
                self.robot_state, gains, damping_array=damping_array
            ),
            _FSM.READY.value: ReadyController(
                self.robot_state,
                gains,
                stiffness_array=stiffness_array,
                damping_array=damping_array,
                default_joint_pos=default_joint_pos,
                control_dt=self._control_dt,
                robot_model=self.robot_model,
            ),
            **{
                state.value: wbc
                for state in (
                    _FSM.MOVE,
                    _FSM.VLA,
                    _FSM.TELEOP,
                    _FSM.GOAL_POSE,
                    _FSM.AUTO_COLLECT,
                    _FSM.VLA_PLANNER,
                    _FSM.FK_CALIBRATION,
                )
            },
        }

    def _build_default_q(self, lower_body_indices, config) -> np.ndarray:
        """Assemble the full Pinocchio q vector with configured default angles."""
        default_q = self.robot_model.q_zero.copy()

        for i, idx in enumerate(lower_body_indices):
            if i < len(config["default_angles"]):
                default_q[idx] = float(config["default_angles"][i])

        for i, idx in enumerate(self._upper_body_indices):
            default_q[idx] = float(self._default_upper_body_pose[i])

        return default_q

    def _setup_ros_interfaces(self) -> None:
        """Create publishers, subscribers and the control timer."""
        # Wrist camera geometry, filled in by _camera_info_callback. Set before any
        # subscription so callbacks never see them missing.
        self._wrist_preparer = None
        self._tip_lifters: list[TipTargetLifter] = []

        self.cmd_pub = self.create_publisher(LowCmd, definitions.TOPIC_LOWCMD, 1)
        self.state_sub = self.create_subscription(
            LowState, definitions.TOPIC_LOWSTATE, self._state_callback, 1
        )
        self.fsm_state_sub = self.create_subscription(
            UInt8, definitions.TOPIC_FSMSTATE, self._fsm_state_callback, 1
        )

        # Latest frame only; BEST_EFFORT must match the sim publisher (a RELIABLE
        # subscriber would receive nothing).
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Compressed transport by default: the raw wrist stream stalls the robot's
        # RELIABLE usb_cam writer (see g1_common.image_transport). Simulation passes
        # raw topics. A '/compressed' suffix selects CompressedImage.
        head_topic = self._param(
            "head_image_topic", definitions.TOPIC_CAMERA + "/compressed"
        )
        wrist_topic = self._param(
            "wrist_image_topic", definitions.TOPIC_CAMERA_WRIST + "/compressed"
        )
        # Same view keys as the dataset recorder: head → ego_view, hand → wrist_view.
        self._camera_workers = []
        self.camera_sub = subscribe_image_latest(
            self,
            head_topic,
            lambda img, _msg: self._image_callback(img, "ego_view"),
            cam_qos,
        )
        self._camera_workers.append(self.camera_sub)
        self.camera_wrist_sub = subscribe_image_latest(
            self,
            wrist_topic,
            lambda img, _msg: self._image_callback(img, "wrist_view"),
            cam_qos,
        )
        self._camera_workers.append(self.camera_wrist_sub)
        self.get_logger().info(
            f"ego_view: '{head_topic}' [{transport_name(head_topic)}]; "
            f"wrist_view: '{wrist_topic}' [{transport_name(wrist_topic)}]"
        )
        # Wrist CameraInfo: sizes the square crop, supplies distortion, and gives
        # the back-projection intrinsics.
        self.camera_info_wrist_sub = self.create_subscription(
            CameraInfo,
            definitions.TOPIC_CAMERA_WRIST_INFO,
            self._camera_info_callback,
            cam_qos,
        )
        self.task_sub = self.create_subscription(
            String,
            definitions.TOPIC_TASKINSTRUCTION,
            self._task_instruction_callback,
            1,
        )
        self.control_timer = self.create_timer(self._control_dt, self._control_loop)
        self.crc = CRC()

        # TF: look up the wall target; drill_tip_link TF is published by
        # robot_state_publisher from the URDF fixed joint.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_low_state_msg = None
        # Last finite command, re-sent in place of a non-finite one.
        self._last_good_cmd: LowCmd | None = None

    def _setup_context(self) -> None:
        """Build the shared control context the state handlers work against."""
        # The drill-tip frame only exists in the drill URDF variant; without it the
        # per-frame task metric cannot be computed and is published as invalid.
        has_drill_tip = self.robot_model.has_frame(definitions.DRILL_TIP_FRAME)
        if not has_drill_tip:
            self.get_logger().warn(
                f"'{definitions.DRILL_TIP_FRAME}' not in the URDF — the tip→target "
                "task metric will be recorded as invalid. Load the drill URDF variant."
            )

        self._ctx = ControlContext(
            robot_model=self.robot_model,
            robot_state=self.robot_state,
            tf_buffer=self._tf_buffer,
            default_upper_body_pose=self._default_upper_body_pose,
            upper_body_indices=self._upper_body_indices,
            left_eef_frame=self._left_eef_frame,
            right_eef_frame=self._right_eef_frame,
            control_dt=self._control_dt,
            full_q=self.robot_model.q_zero.copy(),
            has_drill_tip=has_drill_tip,
        )

    def _setup_vla(self) -> None:
        """Initialise VLA client, action manager and teleoperation manager."""
        chunk_size = int(
            self.get_parameter("vla_chunk_size").value
            if self.has_parameter("vla_chunk_size")
            else definitions.VLA_CHUNK_SIZE
        )
        self.get_logger().info(f"Setting up VLA with chunk size: {chunk_size}")
        # Sized from the model's upper-body joint group.
        self.vla_actions = VLAActionManager(
            chunk_size=chunk_size,
            num_upper_body_joints=len(self._upper_body_indices),
        )

        self._ood_scorer = None
        dataset_path = (
            self.get_parameter("dataset_path").value
            if self.has_parameter("dataset_path")
            else ""
        )
        if dataset_path:
            self._ood_scorer = ObsOODScorer(dataset_path)

        self.vla_client = self._build_action_client(
            self.get_parameter("server_host").value,
            int(self.get_parameter("server_port").value),
            self.vla_actions,
            ood_scorer=self._ood_scorer,
            ood_correction_enabled=bool(self._param("ood_correction_enabled", False)),
        )

        # Coarse (approach) = action VLA with a buffer; fine (close-range) = detector.
        self._vla_sources = VlaSources(
            coarse=VlaRole(self.vla_client, self.vla_actions),
            fine=VlaRole(self._build_perception_client()),
        )
        # Resolved once; sensor callbacks fan out over it.
        self._vla_clients = self._vla_sources.unique

        self.teleoperation_manager = TeleoperationManager()

    def _build_action_client(
        self,
        host: str,
        port: int,
        actions: VLAActionManager,
        ood_scorer=None,
        ood_correction_enabled: bool = False,
    ) -> VLAClient:
        """An action VLA client bound to its own action buffer.

        Each client gets its own RobotModel: it runs IK/FK on a background
        thread and pinocchio Data is not thread-safe.
        """
        return VLAClient(
            host,
            port,
            self.get_parameter("inference_dt").value,
            actions.action_buffer,
            actions.action_lock,
            actions.new_action_chunk,
            RobotModel(
                self.get_parameter("urdf_path").value,
                self.get_parameter("asset_path").value,
                set_floating_base=False,
                supplemental_info=G1SupplementalInfo(),
            ),
            self._default_upper_body_pose,
            ood_scorer=ood_scorer,
            ood_correction_enabled=ood_correction_enabled,
            # Lifts a server perception head, if any, to a tip-frame estimate.
            tip_target_lifter=self._tip_target_lifter(),
            clock=TaskClock(self),
            # De-jitter: moving average within a chunk, deadband between chunks.
            smooth_radius=int(self._param("vla_smooth_radius", 2)),
            action_deadband=float(self._param("vla_action_deadband", 0.0)),
            action_deadband_rot_deg=float(
                self._param("vla_action_deadband_rot_deg", 1.0)
            ),
        )

    def _build_perception_client(self) -> PerceptionClient:
        """Detector-only model on the fine server.

        Lifts 2D detections to a tip-frame target pose that VLA_PLANNER plans onto.
        """
        # Own inference rate, independent of the action model's replanning cadence.
        return PerceptionClient(
            self._param("fine_server_host", self.get_parameter("server_host").value),
            int(self._param("fine_server_port", 5556)),
            float(
                self._param(
                    "perception_inference_dt", self.get_parameter("inference_dt").value
                )
            ),
            self._tip_target_lifter(),
            clock=TaskClock(self),
        )

    def _tip_target_lifter(self) -> TipTargetLifter:
        """A detection -> drill-tip-frame lifter, one per client.

        Created without intrinsics; :meth:`_camera_info_callback` sets them on
        every registered lifter once the prepared wrist frame is known.
        """
        lifter = TipTargetLifter(
            self._tf_buffer,
            camera_frame=self._param(
                "perception_camera_frame", "hand_camera_optical_frame"
            ),
            tip_frame=definitions.DRILL_TIP_FRAME,
            min_visibility=float(self._param("perception_min_visibility", 0.0)),
        )
        self._tip_lifters.append(lifter)
        return lifter

    def _param(self, name: str, default):
        """Read a YAML-declared parameter, falling back to ``default``."""
        return self.get_parameter(name).value if self.has_parameter(name) else default

    def _vla_planner_config(self) -> VlaPlannerConfig:
        """Read the VLA_PLANNER tuning (vla_planner.yaml) into its config."""
        defaults = VlaPlannerConfig()
        return VlaPlannerConfig(
            **{
                field: type(default)(self._param(f"vla_planner_{field}", default))
                for field, default in vars(defaults).items()
            }
        )

    def _setup_handlers(self) -> None:
        """Build the per-FSM-state handler registry and the recorder publisher."""
        ctx = self._ctx
        self._metrics = MetricRecorder(self, ctx, self.vla_client)
        # Fine-source recorder for VLA_PLANNER (pred_rel_*/rel_err_*); coarse in drv_*.
        self._perception_metrics = MetricRecorder(
            self,
            ctx,
            self._vla_sources.fine.client,
            driver_client=self._vla_sources.coarse.client,
        )
        progress_test = ProgressTestHarness(self, ctx, self._metrics)
        vla_debug = VlaDebugPublisher(self)

        self._rc_handler = RcHandler(self, ctx)
        self._handlers = {
            _FSM.TELEOP.value: TeleopHandler(ctx, self.teleoperation_manager),
            _FSM.VLA.value: VlaHandler(
                self,
                ctx,
                self._vla_sources,
                self._metrics,
                progress_test,
                vla_debug,
            ),
            _FSM.GOAL_POSE.value: GoalPoseHandler(
                self, ctx, self.controllers[_FSM.GOAL_POSE.value]
            ),
            _FSM.AUTO_COLLECT.value: AutoCollectHandler(self, ctx),
            _FSM.FK_CALIBRATION.value: FkCalibrationHandler(
                self, ctx, self.controllers[_FSM.FK_CALIBRATION.value]
            ),
            _FSM.VLA_PLANNER.value: VlaPlannerHandler(
                self,
                ctx,
                self._vla_sources.coarse.client,
                self._vla_sources.fine.client,
                self.vla_actions,
                self._perception_metrics,
                vla_debug,
                self._vla_planner_config(),
                # Projects the target back into the wrist frame for visual servoing.
                tip_projector=self._tip_target_lifter(),
                # Approach is frozen on the interpolator output sent to the motors.
                controller=self.controllers[_FSM.VLA_PLANNER.value],
            ),
        }
        self._step_pub = ControlStepPublisher(self, ctx)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _state_callback(self, msg: LowState) -> None:
        """Latch the newest LowState; unpacking happens once per tick.

        LowState arrives at 500 Hz on the control loop's thread, so per-message
        work would delay the 50 Hz tick.
        """
        self._latest_low_state_msg = msg

    def _ingest_low_state(self, msg: LowState) -> None:
        """Unpack the latched LowState into the shared state, once per tick."""
        measured_joints = np.array(
            [m.q for m in msg.motor_state[: definitions.NUM_MOTOR_G1_BASE]]
        )
        self.robot_state.joint_pos = measured_joints
        self.robot_state.joint_vel = np.array(
            [m.dq for m in msg.motor_state[: definitions.NUM_MOTOR_G1_BASE]]
        )
        self.robot_state.joint_torque = np.array(
            [m.tau_est for m in msg.motor_state[: definitions.NUM_MOTOR_G1_BASE]]
        )
        self.robot_state.quat = np.array(msg.imu_state.quaternion)
        self.robot_state.gyro = np.array(msg.imu_state.gyroscope)
        self.robot_state.mode_machine = msg.mode_machine
        self._ctx.latest_rc = msg.wireless_remote

        for client in self._vla_clients:
            client.set_robot_state(self.robot_state)

        self._ctx.full_q[:] = self.robot_model.q_zero
        self._ctx.full_q[self.actuated_indices] = measured_joints

        left_eef_pose, right_eef_pose = self.robot_model.get_eef_poses(
            self._ctx.full_q, self._left_eef_frame, self._right_eef_frame
        )
        self._ctx.current_left_eef_9d = self.robot_model.se3_to_9d_pose(left_eef_pose)
        self._ctx.current_right_eef_9d = self.robot_model.se3_to_9d_pose(right_eef_pose)

        # Only the drill tip reaches the policy.
        for client in self._vla_clients:
            client.set_right_eef(self._ctx.current_right_eef_9d)

    def _fsm_state_callback(self, msg: UInt8) -> None:
        """Route an FSM transition: exit the old state, activate, enter the new."""
        previous_state = self._ctx.fsm_state
        new_state = msg.data
        if previous_state == new_state:
            return
        self._ctx.fsm_state = new_state

        try:
            state_name = _FSM(new_state).name
        except ValueError:
            state_name = "UNKNOWN"
        self.get_logger().info(f"FSM State changed: {state_name} ({new_state})")

        # on_enter reads full_q/eef poses, which may not be unpacked yet.
        if self._latest_low_state_msg is not None:
            self._ingest_low_state(self._latest_low_state_msg)

        exiting = self._handlers.get(previous_state)
        if exiting is not None:
            exiting.on_exit()

        if new_state in self.controllers:
            controller = self.controllers[new_state]
            previous_controller = self.controllers.get(previous_state)
            if controller is previous_controller:
                # Shared WBC instance: keep the learned arm feedforward.
                controller.on_activate(reset_arm_integral=False)
            else:
                controller.on_activate()

        entering = self._handlers.get(new_state)
        if entering is not None:
            entering.on_enter(previous_state)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Adopt the wrist camera geometry from the first CameraInfo.

        Builds the square/rectify preparer and hands its intrinsics to every
        lifter. Later messages are ignored.
        """
        if self._wrist_preparer is not None:
            return
        side = min(int(msg.width), int(msg.height))
        try:
            self._wrist_preparer = preparer_from_camera_info(msg, side)
        except ValueError as exc:
            # Unsupported lens: lifters stay without intrinsics, detections dropped.
            self.get_logger().error(
                f"Wrist camera unusable: {exc}", throttle_duration_sec=10.0
            )
            return
        # Size included for the in-frame check of the handover gate.
        intrinsics = CameraIntrinsics(
            fx=self._wrist_preparer.intrinsics.fx,
            fy=self._wrist_preparer.intrinsics.fy,
            cx=self._wrist_preparer.intrinsics.cx,
            cy=self._wrist_preparer.intrinsics.cy,
            width=self._wrist_preparer.intrinsics.width,
            height=self._wrist_preparer.intrinsics.height,
        )
        for lifter in self._tip_lifters:
            lifter.set_intrinsics(intrinsics)
        self.get_logger().info(
            f"Wrist camera: {msg.width}x{msg.height} -> {side}x{side} via "
            f"{type(self._wrist_preparer).__name__}; back-projecting with "
            f"fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
            f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}"
        )

    def _image_callback(self, img: np.ndarray, view: str) -> None:
        """Route one decoded rgb8 frame to every VLA client."""
        if view == "wrist_view":
            # Square + rectify; dropped until CameraInfo has arrived.
            if self._wrist_preparer is None:
                self.get_logger().warning(
                    "Wrist frame dropped: no CameraInfo on "
                    f"{definitions.TOPIC_CAMERA_WRIST_INFO} yet",
                    throttle_duration_sec=5.0,
                )
                return
            img = self._wrist_preparer(img)
        for client in self._vla_clients:
            client.set_image(img, view)

    def _task_instruction_callback(self, msg: String) -> None:
        """Forward the task to the VLA clients and set the scoring target frame."""
        self.get_logger().info(f"Received a new task: {msg.data}")
        for client in self._vla_clients:
            client.set_task(msg.data)

        frame = resolve_target_frame(msg.data)

        # AUTO_COLLECT sets the frame from its sampled colour; the parse is only
        # a cross-check there.
        if self._ctx.fsm_state == _FSM.AUTO_COLLECT.value:
            if frame != self._ctx.drill_target_frame:
                self.get_logger().error(
                    f"Task resolver disagrees with AUTO_COLLECT: '{msg.data}' "
                    f"parsed as {frame}, episode sampled "
                    f"{self._ctx.drill_target_frame}. Keeping the sampled "
                    "frame; the sampler/resolver pair has drifted."
                )
            return

        if frame is None:
            self.get_logger().error(
                f"Task '{msg.data}' names no single known target square — "
                "target-relative metrics will be recorded as NaN until a "
                "resolvable task arrives. Scoring against the previous target "
                "would silently attribute this run to the wrong square."
            )
        elif frame != self._ctx.drill_target_frame:
            self.get_logger().info(f"Scoring target -> {frame}.")
        self._ctx.drill_target_frame = frame

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if self._latest_low_state_msg is None:
            self.get_logger().warn("No state received yet", throttle_duration_sec=5.0)
            return

        # One LowState snapshot per tick.
        self._ingest_low_state(self._latest_low_state_msg)

        # High-level commands for the active state (twist / upper-body targets).
        self._handlers.get(self._ctx.fsm_state, self._rc_handler).update()

        active_controller = self.controllers.get(
            self._ctx.fsm_state, self.controllers[_FSM.DAMPING.value]
        )
        motor_cmd_msg = active_controller.get_control()

        # Non-finite commands are replaced by the last good one (not skipped).
        if motor_cmd_msg is not None:
            bad = nonfinite_motors(motor_cmd_msg)
            if bad:
                self.get_logger().error(
                    f"Non-finite command on motors {bad} — holding the last good "
                    "command. Suspect an IK failure upstream.",
                    throttle_duration_sec=1.0,
                )
                motor_cmd_msg = self._last_good_cmd
            else:
                self._last_good_cmd = motor_cmd_msg

        if motor_cmd_msg is not None:
            motor_cmd_msg.crc = self.crc.Crc(motor_cmd_msg)
            self.cmd_pub.publish(motor_cmd_msg)

        # Warn when the joint-speed limit makes the upper-body goal finish late.
        lateness = getattr(active_controller, "last_goal_lateness_s", 0.0)
        if lateness > 0.2:
            peak = getattr(active_controller, "last_chunk_peak_speed", 0.0)
            self.get_logger().warn(
                f"Upper-body goal finishes {lateness:.2f} s later than asked: it "
                f"wants up to {peak:.2f} rad/s and upper_body_max_joint_speed is "
                f"{self._config.get('upper_body_max_joint_speed', 0.0):.2f}.",
                throttle_duration_sec=5.0,
            )

        # Synced state/action snapshot for the dataset recorder (its only state source).
        self._step_pub.publish(self._latest_low_state_msg)

    def destroy_node(self):
        """Stop image workers before their subscriptions and logger disappear."""
        for worker in getattr(self, "_camera_workers", []):
            worker.close()
        return super().destroy_node()


def main():
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    node = ControllerManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Flush any in-progress metrics CSV.
        node._metrics.finalize()
        node._perception_metrics.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
