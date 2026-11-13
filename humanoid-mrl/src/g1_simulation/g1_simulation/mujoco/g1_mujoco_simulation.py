# MIT License
#
# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings
from functools import partial

import mujoco
import mujoco.viewer
import numpy as np
from ament_index_python.packages import get_package_share_directory

from g1_common import CRC
from g1_definitions import definitions
from geometry_msgs.msg import TransformStamped, Vector3
from rclpy.node import Node
from tap import Tap
from tf2_ros import TransformBroadcaster
from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState, MotorCmd, MotorState

from g1_simulation.elastic_band import ElasticBandCore
from g1_simulation.input_device import setup_input_device


class SimulatorCliArgs(Tap):
    """Mujoco simulator node for the unitree G1."""

    robot_scene: str = "scene_29dof.xml"
    """Scene description xml file name"""
    print_scene_information: bool = False
    """Print Simulator statistics"""
    enable_elastic_band: bool = False
    """Whether the simulation should start with the gantry enabled or not"""
    simulate_dt: float = 0.002
    """Time step of the simulation"""
    viewer_dt: float = 0.02
    """Time step of the GUI"""


class G1MujocoSimulator:
    """Main Mujoco Simulation Node"""

    def __init__(self, args: SimulatorCliArgs, logger) -> None:
        self._logger = logger
        self._load_mujoco_simulator(args.robot_scene, args.simulate_dt)
        self._input_device = setup_input_device(self._logger)

        # HACK(@naefjo): Not the most elegant way of detecting whether or not we are using the correct xml...
        # One fix would be to add an explicit identifier to the body of the robot.
        self._g1_with_hands = (
            mujoco.mj_name2id(
                self._mj_model,
                mujoco._enums.mjtObj.mjOBJ_BODY,
                "right_wrist_yaw_link",
            )
            == -1
        )
        """Whether the loaded model has actuated hands"""

        self._num_motor = self._mj_model.nu
        """Number of actuated degrees of freedom of G1"""

        self._dim_motor_sensor = definitions.MOTOR_SENSOR_NUM * self._num_motor

        # Centralized sensor index definitions
        self._motor_q_start_idx = 0
        self._motor_q_end_idx = self._num_motor
        self._motor_dq_start_idx = self._num_motor
        self._motor_dq_end_idx = 2 * self._num_motor
        self._motor_tau_start_idx = 2 * self._num_motor
        self._motor_tau_end_idx = 3 * self._num_motor

        # IMU sensors (come after all motor sensors)
        self._quat_start_idx = self._dim_motor_sensor
        self._quat_end_idx = self._quat_start_idx + 4
        self._gyro_start_idx = self._quat_end_idx
        self._gyro_end_idx = self._gyro_start_idx + 3
        self._acc_start_idx = self._gyro_end_idx
        self._acc_end_idx = self._acc_start_idx + 3

        self._elastic_band = ElasticBand("torso_link", True)
        self._elastic_band.init(self._mj_model, self._mj_data)

        self.viewer = mujoco.viewer.launch_passive(
            self._mj_model,
            self._mj_data,
            key_callback=self._elastic_band.mujoco_key_callback,
        )

        self._crc = CRC()

        self._latest_low_cmd = None

        if args.print_scene_information:
            self.print_scene_information()

    def _load_mujoco_simulator(self, scene_file: str, timestep: float) -> None:
        """Loads Mujoco configuration file and sets up mujoco state"""

        scene_file_path = os.path.join(
            get_package_share_directory("g1_simulation"),
            "assets",
            "mujoco",
            "g1",
            scene_file,
        )
        self._mj_model = mujoco.MjModel.from_xml_path(scene_file_path)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mj_model.opt.timestep = timestep

    def get_pelvis_pose(self):
        """
        Get pelvis pose (position, quaternion).

        Returns:
            Position is [x, y, z]
            Quaternion is [w, x, y, z] (MuJoCo convention)
        """

        pelvis_link_id = mujoco.mj_name2id(
            self._mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, "pelvis"
        )
        if pelvis_link_id == -1:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        pos = self._mj_data.xpos[pelvis_link_id]
        quat = self._mj_data.xquat[pelvis_link_id]

        return pos, quat

    def update_visualization(self) -> None:
        """Updates the mujoco viewer"""

        if not self.viewer.is_running():
            return

        self.viewer.sync()

    def to_euler_angles(self, qw, qx, qy, qz):

        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = np.sqrt(1 + 2 * (qw * qy - qx * qz))
        cosp = np.sqrt(1 - 2 * (qw * qy - qx * qz))
        pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def step(self):
        self._elastic_band.step(self._mj_data)

        self._compute_torques()

        mujoco.mj_step(self._mj_model, self._mj_data)

        return self.prepare_msgs()

    def _compute_torques(self):
        # TODO(@naefjo): recompute hand PD torques every step, not only on HandCmd.

        if self._mj_data is None or self._latest_low_cmd is None:
            return

        for i in range(definitions.NUM_MOTOR_G1_BASE):
            # If the motor is disabled, we dont apply any torque.
            if self._latest_low_cmd.motor_cmd[i].mode == 0:
                self._mj_data.ctrl[i] = 0
                self._logger.warn(
                    f"Joint {i} inactive (lowcmd.mode = 0)",
                    throttle_duration_sec=5,
                )
                continue

            self._mj_data.ctrl[i] = self._compute_pd_action(
                self._latest_low_cmd.motor_cmd[i],
                self._mj_data.sensordata[self._motor_q_start_idx + i],
                self._mj_data.sensordata[self._motor_dq_start_idx + i],
            )

    @staticmethod
    def _compute_pd_action(motor_cmd: MotorCmd, motor_pos, motor_vel) -> float:
        """Computes the motor torque corresponding to the PD law in the motor_cmd"""

        return np.nan_to_num(
            motor_cmd.tau
            + motor_cmd.kp * (motor_cmd.q - motor_pos)
            + motor_cmd.kd * (motor_cmd.dq - motor_vel)
        )

    def set_hand_cmd(self, msg: HandCmd, start_idx: int) -> None:
        """Writes hand commands into the simulator data struct"""

        if not self._g1_with_hands:
            warnings.warn("Trying to set HandCmd on a robot without hands!")
            return

        if self._mj_data is None:
            return

        for index in definitions.HandIndices:
            i = start_idx + index.value
            self._mj_data.ctrl[i] = self._compute_pd_action(
                msg.motor_cmd[i],
                self._mj_data.sensordata[self._motor_q_start_idx + i],
                self._mj_data.sensordata[self._motor_dq_start_idx + i],
            )

    def set_low_cmd(self, msg: LowCmd) -> None:
        """Writes low command into the simulator struct"""

        self._latest_low_cmd = msg

    def prepare_msgs(self):
        low_state_msg = LowState()
        left_hand_state_msg = HandState()
        right_hand_state_msg = HandState()

        low_state_msg.mode_pr = definitions.G1JointControlMode.PR_Mode.value

        # NOTE(@bmicha) mode_machine set to match our robot, otherwise take from here
        # https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
        low_state_msg.mode_machine = 5

        q_low_cmd = self._mj_data.sensordata[
            self._motor_q_start_idx : self._motor_q_start_idx
            + definitions.NUM_MOTOR_G1_BASE
        ]
        dq_low_cmd = self._mj_data.sensordata[
            self._motor_dq_start_idx : self._motor_dq_start_idx
            + definitions.NUM_MOTOR_G1_BASE
        ]
        tau_est_low_cmd = self._mj_data.sensordata[
            self._motor_tau_start_idx : self._motor_tau_start_idx
            + definitions.NUM_MOTOR_G1_BASE
        ]

        for motor_state, q, dq, tau_est in zip(
            low_state_msg.motor_state, q_low_cmd, dq_low_cmd, tau_est_low_cmd
        ):
            motor_state.q = q
            motor_state.dq = dq
            motor_state.tau_est = tau_est

        low_state_msg.imu_state.quaternion = self._mj_data.sensordata[
            self._quat_start_idx : self._quat_end_idx
        ]
        low_state_msg.imu_state.gyroscope = self._mj_data.sensordata[
            self._gyro_start_idx : self._gyro_end_idx
        ]
        low_state_msg.imu_state.accelerometer = self._mj_data.sensordata[
            self._acc_start_idx : self._acc_end_idx
        ]
        low_state_msg.wireless_remote = self._input_device.read_state()

        qw, qx, qy, qz = low_state_msg.imu_state.quaternion

        roll, pitch, yaw = self.to_euler_angles(qw, qx, qy, qz)

        low_state_msg.imu_state.rpy = [roll, pitch, yaw]

        low_state_msg.crc = self._crc.Crc(low_state_msg)

        # TODO: this should ideally be set to None instead of default msgs for the robot without hands
        if not self._g1_with_hands:
            return low_state_msg, left_hand_state_msg, right_hand_state_msg

        for index in definitions.HandIndices:
            left_mj_data_idx = definitions.LEFT_HAND_START_IDX + index.value
            right_mj_data_idx = definitions.RIGHT_HAND_START_IDX + index.value

            left_hand_motor_state = MotorState()
            left_hand_motor_state.q = self._mj_data.sensordata[
                self._motor_q_start_idx + left_mj_data_idx
            ]
            left_hand_motor_state.dq = self._mj_data.sensordata[
                self._motor_dq_start_idx + left_mj_data_idx
            ]
            left_hand_motor_state.tau_est = self._mj_data.sensordata[
                self._motor_tau_start_idx + left_mj_data_idx
            ]
            left_hand_state_msg.motor_state.append(left_hand_motor_state)

            right_hand_motor_state = MotorState()
            right_hand_motor_state.q = self._mj_data.sensordata[
                self._motor_q_start_idx + right_mj_data_idx
            ]
            right_hand_motor_state.dq = self._mj_data.sensordata[
                self._motor_dq_start_idx + right_mj_data_idx
            ]
            right_hand_motor_state.tau_est = self._mj_data.sensordata[
                self._motor_tau_start_idx + right_mj_data_idx
            ]
            right_hand_state_msg.motor_state.append(right_hand_motor_state)

        return low_state_msg, left_hand_state_msg, right_hand_state_msg

    def print_scene_information(self) -> None:
        """Print Mujoco scene information to stdout"""

        print("\n")

        print("<<------------- Link ------------->> ")
        for i in range(self._mj_model.nbody):
            name = mujoco.mj_id2name(self._mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)

        print("\n<<------------- Joint ------------->> ")
        for i in range(self._mj_model.njnt):
            name = mujoco.mj_id2name(
                self._mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i
            )
            if name:
                print("joint_index:", i, ", name:", name)

        print("\n<<------------- Actuator ------------->>")
        for i in range(self._mj_model.nu):
            name = mujoco.mj_id2name(
                self._mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                print("actuator_index:", i, ", name:", name)

        print("\n<<------------- Sensor ------------->>")
        index = 0
        for i in range(self._mj_model.nsensor):
            name = mujoco.mj_id2name(
                self._mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                print(
                    "sensor_index:",
                    index,
                    ", name:",
                    name,
                    ", dim:",
                    self._mj_model.sensor_dim[i],
                )
            index = index + self._mj_model.sensor_dim[i]
        print("\n")


class SimulationNode(Node):
    """Ros wrapper for MuJoCo simulator"""

    def __init__(self, args: SimulatorCliArgs) -> None:
        super().__init__("g1_mujoco_simulator")

        self._simulator = G1MujocoSimulator(args, self.get_logger())

        self._viewer_timer = self.create_timer(args.viewer_dt, self._viewer_callback)
        self._simulation_timer = self.create_timer(
            args.simulate_dt, self._simulation_callback
        )

        self._tf_broadcaster = TransformBroadcaster(self)

        self._setup_publishers()

        self._setup_subscribers()

    def is_active(self):
        return self._simulator.viewer.is_running()

    def _setup_publishers(self) -> None:
        self._left_hand_state_pub = self.create_publisher(
            HandState, definitions.TOPIC_HANDSTATE_LEFT, 1
        )
        self._right_hand_state_pub = self.create_publisher(
            HandState, definitions.TOPIC_HANDSTATE_RIGHT, 1
        )
        self._low_state_pub = self.create_publisher(
            LowState, definitions.TOPIC_LOWSTATE, 1
        )
        self._base_vel_pub = self.create_publisher(Vector3, "state/base_velocity", 1)

    def _setup_subscribers(self) -> None:
        self._left_hand_cmd_sub = self.create_subscription(
            HandCmd,
            definitions.TOPIC_HANDCMD_LEFT,
            partial(self._hand_cmd_callback, start_idx=definitions.LEFT_HAND_START_IDX),
            1,
        )
        self._right_hand_cmd_sub = self.create_subscription(
            HandCmd,
            definitions.TOPIC_HANDCMD_RIGHT,
            partial(
                self._hand_cmd_callback, start_idx=definitions.RIGHT_HAND_START_IDX
            ),
            1,
        )

        self._low_cmd_sub = self.create_subscription(
            LowCmd,
            definitions.TOPIC_LOWCMD,
            self._low_cmd_callback,
            1,
        )

    def _viewer_callback(self):
        self._simulator.update_visualization()

    def _simulation_callback(self) -> None:
        low_state, left_hand, right_hand = self._simulator.step()

        self.publish_data(low_state, right_hand, left_hand)

        self._publish_pelvis_tf()

    def _publish_pelvis_tf(self):
        pos, quat = self._simulator.get_pelvis_pose()

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "pelvis"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])

        self._tf_broadcaster.sendTransform(t)

    def _hand_cmd_callback(self, msg, start_idx):
        self._simulator.set_hand_cmd(msg, start_idx)

    def _low_cmd_callback(self, msg):
        self._simulator.set_low_cmd(msg)

    def publish_data(
        self, low_state_msg, left_hand_state_msg, right_hand_state_msg
    ) -> None:
        self._low_state_pub.publish(low_state_msg)
        self._left_hand_state_pub.publish(left_hand_state_msg)
        self._right_hand_state_pub.publish(right_hand_state_msg)


class ElasticBand:
    """MuJoCo adapter for the shared elastic band (keys 7/8/9 in the viewer)."""

    def __init__(self, band_attached_link_name: str, enable: bool):
        self._core = ElasticBandCore(anchor=(0.0, 0.0, 1.7), enable=enable)
        self._band_attached_link_name = band_attached_link_name
        self._band_attached_link = None

    def init(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
        """Initialize with mujoco data"""

        self._band_attached_link = mj_model.body(self._band_attached_link_name).id

    def step(self, mj_data: mujoco.MjData) -> None:
        """Apply the force to the robot"""

        if not self._core.enabled:
            return

        if self._band_attached_link is None:
            raise RuntimeError("did not call `on_init` before starting the simulator")

        force = self._core.force(mj_data.qpos[:3], mj_data.qvel[:3])
        mj_data.xfrc_applied[self._band_attached_link, :3] = force

    def mujoco_key_callback(self, key) -> None:
        """Key callback for mujoco viewer"""

        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self._core.lower()
        if key == glfw.KEY_8:
            self._core.raise_()
        if key == glfw.KEY_9:
            self._core.toggle()
