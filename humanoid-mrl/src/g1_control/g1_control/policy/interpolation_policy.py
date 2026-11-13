import numbers
import time as time_module
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.interpolate as si

# Smallest knot gap; PchipInterpolator requires strictly increasing x.
_MIN_KNOT_GAP_S = 1e-6


class InterpolationPolicy:
    """Interpolates between waypoints for smooth upper-body motion."""

    def __init__(
        self,
        init_time: float,
        init_values: Dict[str, np.ndarray],
        max_change_rate: float,
    ):
        self.last_action = init_values
        self.concat_order = sorted(init_values.keys())
        self.concat_dims = []
        for key in self.concat_order:
            vec = np.array(init_values[key])
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
                init_values[key] = vec
            assert vec.ndim == 1, f"Shape of {key} must be (D,). Got {vec.shape}."
            self.concat_dims.append(vec.shape[0])

        self.init_values_concat = self._concat_vecs(init_values, 1)
        self.max_change_rate = max_change_rate
        #: Seconds by which the last :meth:`set_goal`'s final waypoint had to be
        #: pushed back to respect ``max_change_rate``. 0.0 means the goal fit.
        self.last_goal_lateness_s = 0.0
        self.reset(init_time)

    def reset(
        self,
        init_time: Optional[float] = None,
        current_values: Optional[Dict[str, np.ndarray]] = None,
    ):
        if init_time is None:
            init_time = time_module.monotonic()
        if current_values is not None:
            concat = self.init_values_concat.copy()
            offset = 0
            for key, dim in zip(self.concat_order, self.concat_dims):
                if key in current_values:
                    concat[0, offset : offset + dim] = current_values[key]
                offset += dim
        else:
            concat = self.init_values_concat
        self.interp = PoseTrajectoryInterpolator(np.array([init_time]), concat)
        self.last_waypoint_time = init_time

    def _concat_vecs(self, values: Dict[str, np.ndarray], length: int) -> np.ndarray:
        concat_vecs = []
        for key in self.concat_order:
            if key in values:
                vec = np.array(values[key])
                if vec.ndim == 1:
                    vec = np.tile(vec, (length, 1))
                concat_vecs.append(vec)
            else:
                concat_vecs.append(np.tile(self.last_action[key], (length, 1)))
        return np.concatenate(concat_vecs, axis=1)

    def _unconcat_vecs(self, concat_vec: np.ndarray) -> Dict[str, np.ndarray]:
        curr_idx = 0
        action = {}
        for key, dim in zip(self.concat_order, self.concat_dims):
            action[key] = concat_vec[curr_idx : curr_idx + dim]
            curr_idx += dim
        return action

    def set_goal(self, goal: Dict[str, Any]) -> None:
        if "target_time" not in goal:
            return
        assert "interpolation_garbage_collection_time" in goal
        target_time = goal.pop("target_time")
        interpolation_garbage_collection_time = goal.pop(
            "interpolation_garbage_collection_time"
        )

        if isinstance(target_time, list):
            for key, vec in goal.items():
                assert isinstance(vec, list)
                assert len(vec) == len(target_time)
        else:
            target_time = [target_time]
            for key in goal:
                goal[key] = [goal[key]]

        concat_vecs = self._concat_vecs(goal, len(target_time))
        for tt, vec in zip(target_time, concat_vecs):
            if tt < interpolation_garbage_collection_time:
                continue
            self.interp = self.interp.schedule_waypoint(
                pose=vec,
                time=tt,
                max_change_rate=self.max_change_rate,
                interpolation_garbage_collection_time=interpolation_garbage_collection_time,
                last_waypoint_time=self.last_waypoint_time,
            )
            self.last_waypoint_time = tt

        # Delay the rate limit added to this goal's end (actual vs. requested end).
        self.last_goal_lateness_s = max(
            0.0, float(self.interp.times[-1]) - float(self.last_waypoint_time)
        )

    def get_action(self, time: Optional[float] = None) -> Dict[str, Any]:
        if time is None:
            time = time_module.monotonic()
        concat_vec = self.interp(time)
        self.last_action.update(self._unconcat_vecs(concat_vec))
        return self.last_action

    def get_velocity(self, time: Optional[float] = None) -> Dict[str, np.ndarray]:
        """d/dt of what :meth:`get_action` returns, same keys, per unit time.

        Analytic reference velocity off the spline; zero once the trajectory is
        exhausted.
        """
        if time is None:
            time = time_module.monotonic()
        return self._unconcat_vecs(self.interp.velocity(time))


class PoseTrajectoryInterpolator:
    """Monotone cubic (PCHIP) interpolation through timed waypoints.

    C1 and non-overshooting. Notes:

    * ``max_change_rate`` bounds the average slope between knots; PCHIP may
      locally exceed it (up to ~1.5x).
    * :meth:`trim` rebuilds the curve and re-estimates derivatives, so
      tick-by-tick re-trimming is effectively piecewise secant.
    """

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        times = np.asarray(times)
        poses = np.asarray(poses)
        self.num_joint = len(poses[0])

        # Knots stored explicitly; `times`/`poses` are part of the interface.
        self._times = times
        self._poses = poses

        if len(times) == 1:
            self.single_step = True
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.pose_interp = si.PchipInterpolator(
                times, poses, axis=0, extrapolate=False
            )
            self._vel_interp = self.pose_interp.derivative()

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.unique(np.concatenate([[start_t], keep_times, [end_t]]))
        return PoseTrajectoryInterpolator(times=all_times, poses=self(all_times))

    def schedule_waypoint(
        self,
        pose,
        time,
        max_change_rate=np.inf,
        interpolation_garbage_collection_time=None,
        last_waypoint_time=None,
    ) -> "PoseTrajectoryInterpolator":
        if not isinstance(max_change_rate, np.ndarray):
            max_change_rate = np.array([max_change_rate] * self.num_joint)

        start_time = self.times[0]
        end_time = self.times[-1]

        if interpolation_garbage_collection_time is not None:
            if time <= interpolation_garbage_collection_time:
                return self
            start_time = max(interpolation_garbage_collection_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = interpolation_garbage_collection_time
                else:
                    end_time = max(
                        last_waypoint_time, interpolation_garbage_collection_time
                    )
            else:
                end_time = interpolation_garbage_collection_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        trimmed_interp = self.trim(start_time, end_time)
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pose_min_duration = np.max(np.abs(end_pose - pose) / max_change_rate)
        duration = max(duration, pose_min_duration)
        # Keep knots strictly increasing for PCHIP.
        last_waypoint_time = end_time + max(duration, _MIN_KNOT_GAP_S)

        times = np.append(trimmed_interp.times, [last_waypoint_time])
        poses = np.append(trimmed_interp.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t = np.array([t])
        if self.single_step:
            pose = np.tile(self._poses[0], (len(t), 1))
        else:
            t_clipped = np.clip(t, self.times[0], self.times[-1])
            pose = self.pose_interp(t_clipped)
        return pose[0] if is_single else pose

    def velocity(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """Analytic d/dt of :meth:`__call__`, zero outside the trajectory span
        (where the pose is held)."""
        is_single = isinstance(t, numbers.Number)
        t = np.array([t]) if is_single else np.asarray(t)
        if self.single_step:
            vel = np.zeros((len(t), self.num_joint))
        else:
            times = self.times
            vel = np.zeros((len(t), self.num_joint))
            inside = (t > times[0]) & (t < times[-1])
            if inside.any():
                vel[inside] = self._vel_interp(t[inside])
        return vel[0] if is_single else vel
