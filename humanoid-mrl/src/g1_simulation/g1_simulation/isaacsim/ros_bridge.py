# MIT License
"""ROS bridge for the Isaac simulation: LowCmd/LowState, IMU, TF and cameras."""

import json

import numpy as np
from builtin_interfaces.msg import Time

from g1_common import CRC
from g1_common.joint_converter import JointOrderingConverter
from g1_definitions import definitions
from isaacsim.core.prims import Articulation

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Empty, String
from unitree_hg.msg import LowCmd, LowState, MotorState

from g1_simulation.input_device import setup_input_device
from g1_simulation.isaacsim import calibration_tag, scene
from g1_simulation.isaacsim.camera_graph import (
    build_camera_graph,
    HAND_CAMERA_CENTRE_PX,
    HAND_CAMERA_FX_PX,
    HAND_CAMERA_SIZE_PX,
    HEAD_CAMERA_CENTRE_PX,
    HEAD_CAMERA_FX_PX,
    HEAD_CAMERA_SIZE_PX,
    head_depth_frame_skip,
    head_depth_topic,
)

HEAD_CAMERA_OPTICAL_FRAME = "d435_color_optical_frame"
"""Head image frame_id, same as on the robot.

On hardware realsense2_camera publishes it (not the URDF); in sim this bridge
does, see _publish_head_camera_optical_frame.
"""

# Per-joint rotor armature [kg m^2], matching the physical G1 drives.
_ARMATURES = {
    "left_hip_pitch_joint": 0.01017752004,
    "right_hip_pitch_joint": 0.01017752004,
    "waist_yaw_joint": 0.01017752004,
    "left_hip_roll_joint": 0.025101925,
    "right_hip_roll_joint": 0.025101925,
    "waist_roll_joint": 0.00721945,
    "left_hip_yaw_joint": 0.01017752004,
    "right_hip_yaw_joint": 0.01017752004,
    "waist_pitch_joint": 0.00721945,
    "left_knee_joint": 0.025101925,
    "right_knee_joint": 0.025101925,
    "left_shoulder_pitch_joint": 0.003609725,
    "right_shoulder_pitch_joint": 0.003609725,
    "left_ankle_pitch_joint": 0.00721945,
    "right_ankle_pitch_joint": 0.00721945,
    "left_shoulder_roll_joint": 0.003609725,
    "right_shoulder_roll_joint": 0.003609725,
    "left_ankle_roll_joint": 0.00721945,
    "right_ankle_roll_joint": 0.00721945,
    "left_shoulder_yaw_joint": 0.003609725,
    "right_shoulder_yaw_joint": 0.003609725,
    "left_elbow_joint": 0.003609725,
    "right_elbow_joint": 0.003609725,
    "left_wrist_roll_joint": 0.003609725,
    "right_wrist_roll_joint": 0.003609725,
    "left_wrist_pitch_joint": 0.00425,
    "right_wrist_pitch_joint": 0.00425,
    "left_wrist_yaw_joint": 0.00425,
    "right_wrist_yaw_joint": 0.00425,
}


def sim_stamp(sim_time_s: float) -> Time:
    """Convert ``World.current_time`` to a ROS stamp, carrying nanosec rounding."""
    sec = int(sim_time_s)
    nanosec = int(round((sim_time_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:  # rounding up at a second boundary
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


class RosBridge(Node):
    def __init__(
        self,
        articulation: Articulation,
        calibration: bool = False,
        calibration_rpy_deg=(0.0, 0.0, 0.0),
        calibration_xyz_m=(0.0, 0.0, 0.0),
    ) -> None:
        from isaacsim.sensors.physics import IMUSensor

        super().__init__(node_name="isaaclab_ros_bridge")

        self._articulation = articulation
        stiffness, damping = self._articulation.get_dof_gains()
        self._articulation.set_dof_gains(0.0 * stiffness, 0.0 * damping)
        self._articulation.set_dof_armatures(list(_ARMATURES.values()))

        self._joint_converter = JointOrderingConverter(
            definitions.G1_BASE_PR_JOINT_NAMES, self._articulation.joint_names
        )
        self._input_device = self._setup_input_device()

        self._imu_in_pelvis = IMUSensor(
            prim_path="/World/robot/pelvis/imu_in_pelvis/imu",
            name="pelvis_imu",
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size=20,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        self._imu_in_pelvis.resume()

        # Head (torso d435) and wrist (hand_cam) RGB + camera_info publishers.
        # Each renders at its real counterpart's native size (aspect sets vFOV).
        self._calibration = calibration
        self._calibration_rpy_deg = (
            calibration_rpy_deg if calibration else (0.0, 0.0, 0.0)
        )
        self._calibration_xyz_m = calibration_xyz_m if calibration else (0.0, 0.0, 0.0)
        self._publish_head_camera_optical_frame()
        head_centre = build_camera_graph(
            graph_path="/ROS_Camera",
            camera_prim_path="/World/robot/torso_link/d435_link/camera",
            rgb_topic=definitions.TOPIC_CAMERA,
            frame_id=HEAD_CAMERA_OPTICAL_FRAME,
            width=HEAD_CAMERA_SIZE_PX[0],
            height=HEAD_CAMERA_SIZE_PX[1],
            fx_px=HEAD_CAMERA_FX_PX,
            centre_px=HEAD_CAMERA_CENTRE_PX,
            # Depth on the head only: consumed by g1_sensors' wall_plane_node.
            depth_topic=head_depth_topic(),
            depth_frame_skip=head_depth_frame_skip(),
        )
        hand_centre = build_camera_graph(
            graph_path="/ROS_Camera_Hand",
            camera_prim_path=(
                "/World/robot/hand_drill_hilti/hand_cam_link_01/camera_hand"
            ),
            rgb_topic=definitions.TOPIC_CAMERA_WRIST,
            frame_id="hand_camera_optical_frame",
            width=HAND_CAMERA_SIZE_PX[0],
            height=HAND_CAMERA_SIZE_PX[1],
            fx_px=HAND_CAMERA_FX_PX,
            centre_px=HAND_CAMERA_CENTRE_PX,
            # The asset's camera_hand prim ignores its lens; render via a child prim.
            child_camera_name="lens_fixed",
        )
        self._setup_camera_info(
            [
                (
                    "/camera/d435/color/camera_info",
                    HEAD_CAMERA_OPTICAL_FRAME,
                    HEAD_CAMERA_SIZE_PX,
                    HEAD_CAMERA_FX_PX,
                    head_centre,
                ),
                (
                    "/camera/hand/color/camera_info",
                    "hand_camera_optical_frame",
                    HAND_CAMERA_SIZE_PX,
                    HAND_CAMERA_FX_PX,
                    hand_centre,
                ),
            ]
        )

        self._command_sub = self.create_subscription(
            LowCmd,
            definitions.TOPIC_LOWCMD,
            self._command_sub_cb,
            1,
        )
        self._state_pub = self.create_publisher(
            LowState,
            definitions.TOPIC_LOWSTATE,
            1,
        )

        # /clock and LowState.tick both come from World.current_time in step().
        # This node itself runs on the system clock: it produces sim time, so it
        # must not wait on it. _sim_time_s is the stamp source, updated per step().
        self._sim_time_s = 0.0
        self._clock_pub = self.create_publisher(
            Clock,
            "/clock",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

        # Published from step(), not the carb keyboard thread.
        self._reset_pub = self.create_publisher(
            Empty,
            definitions.TOPIC_SIM_RESET,
            1,
        )

        # Active evaluation scenario (latched), fed from the EvaluationSequencer.
        self._eval_scenario_pub = self.create_publisher(
            String,
            definitions.TOPIC_SIM_EVAL_SCENARIO,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # Active scenario's instruction (latched); the controller re-resolves its
        # drill target frame from each prompt.
        self._task_pub = self.create_publisher(
            String,
            definitions.TOPIC_TASKINSTRUCTION,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        # Programmatic reset for headless runs; runs on the world-step executor,
        # so it never interleaves with a physics step.
        self._reset_handler = None
        self._reset_request_sub = self.create_subscription(
            Empty,
            definitions.TOPIC_SIM_RESET_REQUEST,
            self._reset_request_cb,
            1,
        )
        self._eval_advance_request_sub = self.create_subscription(
            Empty,
            definitions.TOPIC_SIM_EVAL_ADVANCE_REQUEST,
            self._eval_advance_request_cb,
            1,
        )

        self._crc = CRC()

        self._jpos_des = None
        self._jvel_des = None
        self._stiffnesses = None
        self._dampings = None
        self._jeff_des = None

        # Pelvis and targets are broadcast wrt "world" every step (targets move).
        from tf2_ros import TransformBroadcaster

        self._tf_broadcaster = TransformBroadcaster(self)

        # Optional RobotPoseResetter: maps TF frame names to the prim with that
        # visible colour (colours are shuffled on reset).
        self._pose_resetter = None

        # Optional EvaluationSequencer; announcements are drained in step().
        self._eval_sequencer = None

        # Optional callable run on sim/eval_advance_request.
        self._eval_advance_handler = None

    def set_pose_resetter(self, pose_resetter) -> None:
        """Register the RobotPoseResetter so target TFs track actual colours."""
        self._pose_resetter = pose_resetter

    def set_eval_sequencer(self, eval_sequencer) -> None:
        """Register the EvaluationSequencer so scenario changes are announced."""
        self._eval_sequencer = eval_sequencer

    def set_eval_advance_handler(self, handler) -> None:
        """Register the callable run on a sim/eval_advance_request message.

        It must do everything a V-press does, including re-engaging BaseLock at
        the new scenario's pose, not just advance the sequencer."""
        self._eval_advance_handler = handler

    def set_reset_handler(self, handler) -> None:
        """Register the callable run on a sim/reset_request message.

        The handler should do everything a reset-key press does (randomize the
        scene and re-engage the base lock); the reset announcement on
        TOPIC_SIM_RESET then flows through the normal pop_reset_event() path."""
        self._reset_handler = handler

    def _reset_request_cb(self, _msg: Empty) -> None:
        if self._reset_handler is None:
            self._logger.warning(
                "sim/reset_request received but no reset handler is registered."
            )
            return
        self._logger.info("sim/reset_request received — resetting the scene.")
        self._reset_handler()

    def _eval_advance_request_cb(self, _msg: Empty) -> None:
        if self._eval_advance_handler is None:
            self._logger.warning(
                "sim/eval_advance_request received but no evaluation advance "
                "handler is registered."
            )
            return
        self._logger.info("sim/eval_advance_request received — advancing scenario.")
        self._eval_advance_handler()

    def publish_target_transforms(self):
        """Broadcast every wall cell's front-face pose (wrt "world") each step.

        All nine ``{colour}_{shape}_target`` frames are published, including ones
        hidden by an evaluation scenario (they keep their last pose)."""
        if self._calibration:
            return

        import omni.usd
        from geometry_msgs.msg import TransformStamped
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        stamp = self._stamp()

        if self._pose_resetter is not None:
            frame_to_prim = self._pose_resetter.target_frame_to_prim_path
        else:
            frame_to_prim = {
                frame: prim_path
                for (prim_path, frame, _) in scene.WALL_OBJECTS.values()
            }

        transforms = []
        for child, prim_path in frame_to_prim.items():
            translation = cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(prim_path)
            ).ExtractTranslation()
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = child
            # Front face centre (toward the robot, -X), not the cube centre.
            t.transform.translation.x = float(translation[0]) - scene.TARGET_DEPTH / 2
            t.transform.translation.y = float(translation[1])
            t.transform.translation.z = float(translation[2])
            t.transform.rotation.w = 1.0
            transforms.append(t)
        self._tf_broadcaster.sendTransform(transforms)

    def _stamp(self):
        """Return the current sim instant as a stamp (not ``self.get_clock()``).

        This node's own clock is the system clock; everything it publishes is
        stamped in sim time, the same value as /clock and ``LowState.tick``.
        """
        return sim_stamp(self._sim_time_s)

    def _publish_pelvis_transform(self):
        """Broadcast the live pelvis pose as a dynamic transform wrt "world"."""
        from geometry_msgs.msg import TransformStamped

        pos, ori = self._articulation.get_world_poses()
        pos = np.array(pos).flatten()  # (x, y, z)
        ori = np.array(ori).flatten()  # (w, x, y, z)

        t = TransformStamped()
        t.header.stamp = self._stamp()
        t.header.frame_id = "world"
        t.child_frame_id = "pelvis"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(ori[0])
        t.transform.rotation.x = float(ori[1])
        t.transform.rotation.y = float(ori[2])
        t.transform.rotation.z = float(ori[3])
        self._tf_broadcaster.sendTransform(t)

    def _setup_camera_info(self, cameras):
        """Build each camera's CameraInfo from the robot-measured constants; log K.

        Isaac's ROS2CameraInfoHelper is not used: it pins cx, cy to the image
        centre and forces fy = fx.
        """
        from sensor_msgs.msg import CameraInfo

        self._camera_info = []
        for topic, frame_id, size_px, fx_px, centre_px in cameras:
            width, height = size_px
            cx_px, cy_px = centre_px
            msg = CameraInfo()
            msg.header.frame_id = frame_id
            msg.width, msg.height = int(width), int(height)
            # Ideal pinhole; zero distortion selects g1_common.camera's CentreCrop.
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0] * 5
            msg.k = [fx_px, 0.0, cx_px, 0.0, fx_px, cy_px, 0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [fx_px, 0.0, cx_px, 0.0, 0.0, fx_px, cy_px, 0.0, 0.0, 0.0, 1.0, 0.0]
            self._camera_info.append((self.create_publisher(CameraInfo, topic, 1), msg))
            self._logger.info(
                f"{topic}: publishing {width}x{height} fx=fy={fx_px:.6f} "
                f"cx={cx_px:.6f} cy={cy_px:.6f} frame_id={frame_id} d=zeros"
            )

    def _publish_camera_info(self):
        """Restamp and publish both CameraInfos every step, like the real drivers."""
        stamp = self._stamp()
        for publisher, msg in self._camera_info:
            msg.header.stamp = stamp
            publisher.publish(msg)

    def _publish_head_camera_optical_frame(self):
        """Stand in for realsense2_camera's static ``d435_link -> optical`` TF.

        Rotation rpy = (-pi/2, 0, -pi/2), i.e. quaternion (-0.5, 0.5, -0.5, 0.5),
        plus the optional calibration perturbation. Not in the URDF because on the
        robot realsense2_camera owns this frame.

        Known gap: the real chain has an extra d435_link -> color imager hop
        (14.7 mm in y, 0.60 deg); the USD prim's offset from the imager is
        unmeasured, so zero translation is assumed here. Colour and depth share
        one render product, so their registration is exact.
        """
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        # System clock is fine: tf2 ignores static transform stamps.
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "d435_link"
        t.child_frame_id = HEAD_CAMERA_OPTICAL_FRAME
        base_quaternion = (-0.5, 0.5, -0.5, 0.5)
        injected_translation = calibration_tag.quaternion_rotate(
            base_quaternion, self._calibration_xyz_m
        )
        (
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ) = map(float, injected_translation)
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = map(
            float,
            calibration_tag.perturbed_optical_quaternion(
                base_quaternion, self._calibration_rpy_deg
            ),
        )
        self._static_tf_broadcaster.sendTransform(t)
        if self._calibration:
            self._logger.info(
                f"Injected head-camera RPY: {self._calibration_rpy_deg} deg, "
                f"XYZ: {self._calibration_xyz_m} m"
            )
            self._publish_calibration_tag_frame()

    def _publish_calibration_tag_frame(self):
        """Publish the simulated equivalent of the mocap tag frame."""
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()
        # System clock is fine: tf2 ignores static transform stamps.
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = calibration_tag.TAG_FRAME
        t.transform.translation.x = float(
            scene.TARGET_X - calibration_tag.PATTERN_INSET_M
        )
        t.transform.translation.y = float(calibration_tag.TAG_CENTRE_YZ[0])
        t.transform.translation.z = float(calibration_tag.TAG_CENTRE_YZ[1])
        t.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(t)
        self._logger.info(
            f"Published '{calibration_tag.TAG_FRAME}' at "
            f"x={t.transform.translation.x:.4f} "
            f"y={t.transform.translation.y:.3f} "
            f"z={t.transform.translation.z:.3f} (world)."
        )

    def _setup_input_device(self):
        self._logger.info("Setting up input device...")
        try:
            input_device = setup_input_device(self._logger)
            self._logger.info("Input device configured")
            return input_device
        except Exception as e:
            self._logger.error(f"Input device init failed: {e}")
            raise RuntimeError(
                "No input device available (joystick or keyboard)"
            ) from e

    def _command_sub_cb(self, msg: LowCmd):
        joint_positions = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        joint_velocities = np.zeros_like(joint_positions)
        joint_efforts = np.zeros_like(joint_positions)
        stiffnesses = np.zeros_like(joint_positions)
        dampings = np.zeros_like(joint_positions)

        for joint_idx in range(definitions.NUM_MOTOR_G1_BASE):
            joint_positions[joint_idx] = msg.motor_cmd[joint_idx].q
            joint_velocities[joint_idx] = msg.motor_cmd[joint_idx].dq
            joint_efforts[joint_idx] = msg.motor_cmd[joint_idx].tau
            stiffnesses[joint_idx] = msg.motor_cmd[joint_idx].kp
            dampings[joint_idx] = msg.motor_cmd[joint_idx].kd

        self._jpos_des = self._joint_converter.from_source(joint_positions)
        self._jvel_des = self._joint_converter.from_source(joint_velocities)
        self._stiffnesses = self._joint_converter.from_source(stiffnesses)
        self._dampings = self._joint_converter.from_source(dampings)
        self._jeff_des = self._joint_converter.from_source(joint_efforts)

    def compute_torques(self):
        if self._stiffnesses is None:
            return

        self._articulation.set_dof_gains(self._stiffnesses, self._dampings)
        self._articulation.set_dof_position_targets(self._jpos_des)
        self._articulation.set_dof_velocity_targets(self._jvel_des)
        self._articulation.set_dof_efforts(self._jeff_des)

    def step(self, sim_time_s: float) -> None:
        # Stamp everything this step with the physics clock; see _stamp().
        self._sim_time_s = sim_time_s
        joint_positions = self._articulation.get_dof_positions()
        joint_velocities = self._articulation.get_dof_velocities()
        # Joint reaction force projected on the axis; equals g_sim(q) at a static hold.
        joint_efforts = self._articulation.get_dof_projected_joint_forces()

        lowstate_msg = LowState()
        joint_positions = self._joint_converter.to_source(
            np.squeeze(joint_positions.numpy())
        )
        joint_velocities = self._joint_converter.to_source(
            np.squeeze(joint_velocities.numpy())
        )
        joint_efforts = self._joint_converter.to_source(
            np.squeeze(joint_efforts.numpy())
        )

        for joint_idx in range(definitions.NUM_MOTOR_G1_BASE):
            motor_state = MotorState()
            motor_state.q = joint_positions[joint_idx].item()
            motor_state.dq = joint_velocities[joint_idx].item()
            motor_state.tau_est = joint_efforts[joint_idx].item()
            lowstate_msg.motor_state[joint_idx] = motor_state

        imu_state = self._imu_in_pelvis.get_current_frame()
        lowstate_msg.imu_state.accelerometer = imu_state["lin_acc"]
        lowstate_msg.imu_state.quaternion = imu_state["orientation"]
        lowstate_msg.imu_state.gyroscope = imu_state["ang_vel"]

        lowstate_msg.wireless_remote = self._input_device.read_state()

        # Sim time in ms, comparable to image stamps so the recorder can pair
        # frames with interpolated state. On hardware tick is boot-relative.
        lowstate_msg.tick = int(round(sim_time_s * 1000.0)) & 0xFFFFFFFF

        # World.current_time advances per physics step (the timeline only per render).
        clock_msg = Clock()
        clock_msg.clock = self._stamp()
        self._clock_pub.publish(clock_msg)

        lowstate_msg.mode_machine = 5
        lowstate_msg.crc = self._crc.Crc(lowstate_msg)

        self._state_pub.publish(lowstate_msg)

        self._publish_pelvis_transform()
        self.publish_target_transforms()
        self._publish_camera_info()

        # Drain a pending reset (after the TFs, so targets are already updated).
        # A reset also leaves evaluation mode, so clear the latched scenario.
        if self._pose_resetter is not None and self._pose_resetter.pop_reset_event():
            self._reset_pub.publish(Empty())
            self._eval_scenario_pub.publish(String(data='{"scenario": -1}'))
            self._logger.info(f"Sim reset: published on {definitions.TOPIC_SIM_RESET}")

        # Drain a pending scenario; the instruction goes out before the scenario tag
        # so the controller is retargeted before metrics are tagged.
        if self._eval_sequencer is not None:
            announcement = self._eval_sequencer.pop_scenario_announcement()
            if announcement is not None:
                instruction = json.loads(announcement).get("instruction")
                if instruction:
                    self._task_pub.publish(String(data=instruction))
                self._eval_scenario_pub.publish(String(data=announcement))
                self._logger.info(f"Eval scenario: {announcement}")
