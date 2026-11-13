# MIT License
# Whole-body RL controller for Unitree G1

from __future__ import annotations

import time as time_module
from collections.abc import Callable

import numpy as np
import pinocchio as pin
from g1_common.data import RobotState
from g1_definitions import definitions

from unitree_hg.msg import LowCmd

from g1_control.base import BaseController

from g1_control.policy import (
    G1DecoupledWholeBodyPolicy,
    G1GearWbcPolicy,
    InterpolationPolicy,
)
from g1_control.utils import build_low_cmd
from g1_control.utils.wbc_robot_model import RobotModel

_DEFAULT_BASE_HEIGHT = 0.74
_DEFAULT_NAV_CMD = np.array([0.0, 0.0, 0.0])

# Chunk dead-band (rad): hold if every waypoint is within this of the last
# commanded pose.
_CHUNK_MOTION_THRESHOLD = 0.0

# Gravity in the upright pelvis frame; rotated by the measured base attitude each tick.
_GRAVITY_PELVIS = np.array([0.0, 0.0, -9.81])


class WholeBodyRLController(BaseController):
    """Decoupled whole-body controller: ONNX locomotion for the lower body,
    interpolation for the upper body, plus arm gravity feedforward.
    """

    def __init__(
        self,
        robot_state: RobotState,
        gains: dict,
        robot_model: RobotModel,
        config: dict,
        balance_model_path: str,
        walk_model_path: str,
        eval: bool = False,
        clock: Callable[[], float] = time_module.monotonic,
    ):
        super().__init__(robot_state, gains)

        # Must share the control timer's clock (ROS /clock in sim, steady on hardware).
        self._clock = clock

        self._control_dt = float(config["simulation_dt"]) * int(
            config["control_decimation"]
        )
        self.stiffness = np.array(config["motor_kps"], dtype=np.float64)
        self.damping = np.array(config["motor_kds"], dtype=np.float64)

        self._robot_model = robot_model
        self._eval = eval

        # ── Build policies ───────────────────────────────────────────────────
        init_time = self._now()

        upper_body_policy = InterpolationPolicy(
            init_time=init_time,
            init_values={
                "target_upper_body_pose": self._robot_model.get_initial_upper_body_pose(),
                "base_height_command": np.array([_DEFAULT_BASE_HEIGHT]),
                "navigate_cmd": _DEFAULT_NAV_CMD.copy(),
            },
            max_change_rate=float(config.get("upper_body_max_joint_speed", 1.0)),
        )

        lower_body_policy = G1GearWbcPolicy(
            robot_model=self._robot_model,
            config=config,
            balance_model_path=balance_model_path,
            walk_model_path=walk_model_path,
        )

        self._wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=self._robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )

        # Last VLA chunk, so each is fed to the interpolator once.
        self._last_chunk_id = None
        self._last_chunk_peak_speed = 0.0
        # Last upper-body target sent to the interpolator (static-chunk detection).
        self._last_commanded_target: np.ndarray | None = None
        self._upper_body_indices = self._robot_model.get_joint_group_indices(
            "upper_body"
        )

        # PR-order mask of arm joints receiving feedforward; waist and legs get tau=0.
        self._grav_comp_mask = np.zeros(definitions.NUM_MOTOR_G1_BASE, dtype=bool)
        for key in ("left_arm", "right_arm"):
            s, e = definitions.JOINT_PR_SLICES[key]
            self._grav_comp_mask[s:e] = True

        # Arm integral term: Ki * integral(position error) added to the gravity
        # FF to null residual model error. Ungated; the per-joint clamp (Nm,
        # scalar or 7-vector) is the only anti-windup. Ki <= 0 disables it.
        self._arm_int_ki = float(config.get("arm_integral_ki", 0.0))
        clamp = np.asarray(config.get("arm_integral_clamp", 0.0), dtype=np.float64)
        self._arm_int_clamp = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        for key in ("left_arm", "right_arm"):
            s, e = definitions.JOINT_PR_SLICES[key]
            self._arm_int_clamp[s:e] = clamp  # broadcasts scalar or 7-vector
        self._arm_int_state = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        # Time of the last integrator step; None = no interval yet.
        self._arm_int_last_t: float | None = None

        # Send the reference velocity as the motors' dq target (see get_control).
        self._command_reference_velocity = bool(
            config.get("command_reference_velocity", False)
        )

        # Diagnostics (PR order): gravity FF, total FF, and commanded action.
        self._last_g_PR = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        self._last_tau_ff = np.zeros(definitions.NUM_MOTOR_G1_BASE)
        self._last_action_PR = np.zeros(definitions.NUM_MOTOR_G1_BASE)

        # Private gravity-only model for the FF.
        self._ff_model, self._ff_data = self._build_ff_model()

        if self._eval:
            # Fixed default lower-body pose (PR 0-14); no RL lower-body control in eval.
            _default_q = self._robot_model.q_zero.copy()
            _lower_body_indices = self._robot_model.get_joint_group_indices(
                "lower_body"
            )
            for i, idx in enumerate(_lower_body_indices):
                if i < len(config["default_angles"]):
                    _default_q[idx] = float(config["default_angles"][i])
            _default_pr = self._robot_model.get_body_actuated_joints(_default_q)
            self._lower_body_end = definitions.JOINT_PR_SLICES["waist"][1]  # = 15
            self._default_lower_body_PR = _default_pr[: self._lower_body_end].copy()

    def _build_ff_model(self) -> tuple[pin.Model, pin.Data]:
        """Return a private pinocchio model+data used only for the gravity FF.

        Separate from the shared ``robot_model`` because its gravity is rotated
        every tick.
        """
        model = self._robot_model.pinocchio_wrapper.model.copy()
        return model, model.createData()

    @property
    def last_gravity_ff(self) -> np.ndarray:
        """Last gravity feedforward g(q) (PR order), for diagnostics."""
        return self._last_g_PR

    @property
    def last_tau_ff(self) -> np.ndarray:
        """Last total arm feedforward, gravity + integral (PR order)."""
        return self._last_tau_ff

    @property
    def last_action(self) -> np.ndarray:
        """Last commanded joint position (PR order) -- the motors' q target."""
        return self._last_action_PR

    @property
    def last_upper_body_action(self) -> np.ndarray:
        """Last motor command in the planner's upper-body order.

        Live interpolator output (not the chunk's final waypoint), so a state
        can hold it without a target step.
        """
        q = self._robot_model.get_configuration_from_actuated_joints(
            self._last_action_PR
        )
        return np.asarray(q[self._upper_body_indices], dtype=np.float64).copy()

    @property
    def arm_integral_ki(self) -> float:
        """Gain of the arm feedforward integrator; 0 when disabled."""
        return self._arm_int_ki

    @property
    def is_eval(self) -> bool:
        """True when the lower body is pinned to q0 instead of RL-controlled."""
        return self._eval

    @property
    def last_goal_lateness_s(self) -> float:
        """Seconds the rate limit added to the last upper-body goal's end.

        0.0 means the goal fit inside upper_body_max_joint_speed.
        """
        return self._wbc_policy.upper_body_policy.last_goal_lateness_s

    @property
    def last_chunk_peak_speed(self) -> float:
        """Fastest joint speed the last VLA chunk asked for [rad/s].

        Largest step between consecutive raw waypoints times RECORD_FPS; compare
        against upper_body_max_joint_speed. 0.0 until the first chunk.
        """
        return self._last_chunk_peak_speed

    def _now(self) -> float:
        """Read the injected control clock, or wall monotonic if none was."""
        return getattr(self, "_clock", time_module.monotonic)()

    def on_activate(self, *, reset_arm_integral: bool = True) -> None:
        """Called on every transition into a state this controller serves.

        Keep the arm integral when only the handler changes; reset it after
        another controller had control.
        """
        self._wbc_policy.activate_policy()
        if reset_arm_integral:
            self._reset_arm_integral()
        # Reset the chunk dead-band reference.
        self._last_chunk_id = None
        self._last_commanded_target = None
        now = self._now()
        if self.robot_state.joint_pos is not None:
            q_full = self._robot_model.get_configuration_from_actuated_joints(
                self.robot_state.joint_pos
            )
            upper_body_indices = self._robot_model.get_joint_group_indices("upper_body")
            current_upper_body = q_full[upper_body_indices]
            self._wbc_policy.upper_body_policy.reset(
                now, current_values={"target_upper_body_pose": current_upper_body}
            )
        else:
            self._wbc_policy.upper_body_policy.reset(now)

    def _send_chunk_to_interpolator(
        self, chunk: np.ndarray, start_index: int = 0
    ) -> None:
        """Forward a VLA chunk to the interpolation policy and update the reference.

        start_index skips leading dead-band waypoints; remaining waypoints keep
        their original time offsets.
        """
        self._last_commanded_target = chunk[-1].copy()
        now = self._now()
        vla_dt = 1.0 / definitions.RECORD_FPS
        # Peak speed from the chunk's own deltas only (excludes tracking backlog).
        self._last_chunk_peak_speed = (
            float(np.abs(np.diff(chunk, axis=0)).max() / vla_dt)
            if len(chunk) > 1
            else 0.0
        )
        chunk_slice = chunk[start_index:]
        target_times = [
            now + ((start_index + i + 1) * vla_dt) for i in range(len(chunk_slice))
        ]
        self._wbc_policy.set_goal(
            {
                "target_upper_body_pose": list(chunk_slice),
                "target_time": target_times,
                # Splice the re-plan at the currently commanded point so the
                # command stays continuous and the rate limit applies.
                "interpolation_garbage_collection_time": now,
            }
        )

    def schedule_upper_body_target(
        self, target: np.ndarray, duration_s: float
    ) -> None:
        """Schedule one upper-body waypoint without the per-tick Path-B replan.

        Used by calibration states; the duration is still extended if the
        joint-speed limit requires it.
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (len(self._upper_body_indices),):
            raise ValueError(
                "upper-body target has shape "
                f"{target.shape}, expected {(len(self._upper_body_indices),)}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("upper-body target contains NaN or infinity")
        if duration_s <= 0.0:
            raise ValueError("upper-body goal duration must be positive")

        now = self._now()
        self._last_commanded_target = target.copy()
        self._last_chunk_peak_speed = 0.0
        self._wbc_policy.set_goal(
            {
                "target_upper_body_pose": target,
                "target_time": now + float(duration_s),
                "interpolation_garbage_collection_time": now,
            }
        )

    def _reset_arm_integral(self) -> None:
        """Drop the accumulated feedforward torque and restart its clock."""
        self._arm_int_state[:] = 0.0
        self._arm_int_last_t = None

    def _accumulate_arm_error(self, action_PR: np.ndarray, now: float) -> np.ndarray:
        """Advance the arm feedforward integrator and return its state.

        Integrates over the measured tick interval; the state is clamped so
        Ki * state stays within the per-joint limit.
        """
        dt = self._control_dt
        if self._arm_int_last_t is not None:
            # Bound dt so a late tick does not integrate its whole gap.
            dt = min(max(now - self._arm_int_last_t, 0.0), 4.0 * self._control_dt)
        self._arm_int_last_t = now

        err_PR = action_PR - self.robot_state.joint_pos
        self._arm_int_state += err_PR * dt

        state_lim = self._arm_int_clamp / self._arm_int_ki
        np.clip(self._arm_int_state, -state_lim, state_lim, out=self._arm_int_state)
        return self._arm_int_state

    def get_control(self) -> LowCmd | None:
        """Reads from shared RobotState and computes the WBC action."""

        if self.robot_state.joint_pos is None or self.robot_state.joint_vel is None:
            return None

        # 1. Feed velocity command (from RC or VLA) directly to lower body policy
        if self.robot_state.twist_command is not None:
            self._wbc_policy.lower_body_policy.cmd = (
                self.robot_state.twist_command.copy()
            )

        # 2. Forward upper body targets
        # Path A: VLA Chunk processing (list of future targets)
        if (
            hasattr(self.robot_state, "upper_body_target_chunk")
            and self.robot_state.upper_body_target_chunk is not None
        ):
            # Only send to the interpolator if it's a BRAND NEW chunk
            if self._last_chunk_id is not self.robot_state.upper_body_target_chunk:
                self._last_chunk_id = self.robot_state.upper_body_target_chunk

                chunk = self.robot_state.upper_body_target_chunk

                # Hold on static chunks (all waypoints within the dead-band).
                if self._last_commanded_target is not None:
                    disps = np.linalg.norm(chunk - self._last_commanded_target, axis=1)
                    moving = np.argwhere(disps >= _CHUNK_MOTION_THRESHOLD)
                    # Only hold if the arm actually reached the held pose.
                    q_meas_full = (
                        self._robot_model.get_configuration_from_actuated_joints(
                            self.robot_state.joint_pos
                        )
                    )
                    reach_err = np.linalg.norm(
                        q_meas_full[self._upper_body_indices]
                        - self._last_commanded_target
                    )
                    if moving.size == 0 and reach_err < _CHUNK_MOTION_THRESHOLD:
                        pass  # static chunk and pose reached — hold
                    elif moving.size == 0:
                        # Static chunk, pose not reached: re-send.
                        self._send_chunk_to_interpolator(chunk)
                    else:
                        # Send from the first waypoint outside the dead-band.
                        self._send_chunk_to_interpolator(
                            chunk, start_index=int(moving[0])
                        )
                else:
                    # First chunk: always accept.
                    self._send_chunk_to_interpolator(chunk)

        # Path B: Teleoperation fallback (single target, point-by-point tracking)
        elif (
            hasattr(self.robot_state, "upper_body_target")
            and self.robot_state.upper_body_target is not None
        ):
            now = self._now()
            self._wbc_policy.set_goal(
                {
                    "target_upper_body_pose": self.robot_state.upper_body_target,
                    "target_time": now + self._control_dt,
                    # Splice at `now`; see _send_chunk_to_interpolator.
                    "interpolation_garbage_collection_time": now,
                }
            )

        # 3. Build observation dict for the policy using the shared state
        q_full = self._robot_model.get_configuration_from_actuated_joints(
            self.robot_state.joint_pos
        )
        dq_full = self._robot_model.get_configuration_from_actuated_joints(
            self.robot_state.joint_vel
        )

        w, x, y, z = self.robot_state.quat
        floating_base_pose = np.array([0.0, 0.0, 0.0, w, x, y, z])
        floating_base_vel = np.array([0.0, 0.0, 0.0, *self.robot_state.gyro])

        obs = {
            "q": q_full,
            "dq": dq_full,
            "floating_base_pose": floating_base_pose,
            "floating_base_vel": floating_base_vel,
        }

        # 4. Compute action
        self._wbc_policy.set_observation(obs)
        now = self._now()
        action = self._wbc_policy.get_action(time=now)

        # 5. Convert from pinocchio order to PR order and clip
        action_PR = self._robot_model.get_body_actuated_joints(action["q"])
        action_PR = np.clip(
            action_PR,
            definitions.DOF_POS_LOWER_LIMIT_LIST,
            definitions.DOF_POS_UPPER_LIMIT_LIST,
        )
        # Reference velocity from the interpolating spline.
        dq_ref_PR = self._robot_model.get_body_actuated_joints(action["dq"])

        if self._eval:
            # Hold lower body at the default q0 pose — no RL locomotion control.
            action_PR[: self._lower_body_end] = self._default_lower_body_PR

        # 5b. Arm gravity feedforward (includes the drill payload on the right
        # wrist). Fixed-base, all-revolute model: nq == nv, so position index
        # mapping applies to g(q). Gravity is rotated into the pelvis frame
        # using the measured base orientation.
        model = self._ff_model
        data = self._ff_data
        w, x, y, z = (float(v) for v in self.robot_state.quat)
        R_wp = pin.Quaternion(w, x, y, z).normalized().toRotationMatrix()
        model.gravity.linear = R_wp.T @ _GRAVITY_PELVIS
        g_full = pin.computeGeneralizedGravity(model, data, q_full)
        g_PR = self._robot_model.get_body_actuated_joints(g_full)

        # 5c. Integral term (see __init__).
        if self._arm_int_ki > 0.0:
            tau_ff = g_PR + self._arm_int_ki * self._accumulate_arm_error(
                action_PR, now
            )
        else:
            tau_ff = g_PR

        # 6. Generate LowCmd message. The gravity + integral feedforward is applied
        # only to the arm joints (see _grav_comp_mask); all others command tau = 0.
        tau = np.where(self._grav_comp_mask, tau_ff, 0.0)
        self._last_g_PR = g_PR
        self._last_tau_ff = tau
        self._last_action_PR = action_PR
        return build_low_cmd(
            self.robot_state.mode_machine,
            q=action_PR,
            kp=self.stiffness,
            kd=self.damping,
            tau=tau,
            # Reference dq makes Kd damp tracking error rather than motion.
            dq=dq_ref_PR if self._command_reference_velocity else None,
        )
