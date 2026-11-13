# qualisys_driver -- vendored

ROS 2 driver for the Qualisys motion capture system. Vendored from upstream
rather than pinned in `deploy_deps.repos` because we patch it: the issues below
are all in the driver's own code, and they need to be iterated on during limited
mocap lab time, which a fork-push-and-rebuild cycle makes painful.

* Upstream: <https://github.com/MOCAP4ROS2-Project/mocap4ros2_qualisys>
* Vendored at commit `36e7751a56ca8b974f4e1dba9abe17fd67c70bab`
* License: Apache-2.0, see `LICENSE.mocap4ros2_qualisys`

Only the `qualisys_driver` package is vendored. Upstream's
`dependency_repos.repos` is *not* used: `mocap4r2_msgs` is installed from the
jazzy binary (`ros-jazzy-mocap4r2-msgs`, see `docker/Dockerfile`), and
`mocap4r2_control` is not needed at all -- `launch/qualisys.launch.py` emits its
own configure and activate transitions.

The Qualisys C++ SDK is fetched by CMake at build time (`FetchContent`, tag
`rt_protocol_1.23`), so building this package needs network access. The upstream
README's manual SDK install is obsolete; ignore it.

## What this package does and does not do

Publishes `/markers` (`mocap4r2_msgs/Markers`) and `rigid_bodies`
(`mocap4r2_msgs/RigidBodies`), the latter keyed by the 6DOF body names defined in
QTM on the Windows host. It publishes **no TF** -- it constructs a
`tf2_ros::TransformBroadcaster` and never uses it. Turning `rigid_bodies` into
the `<colour>_<shape>_target` frames the controller resolves against is
`g1_mocap`'s job, not this package's.

Note it only publishes when a topic has at least one subscriber
(`get_subscription_count() > 0`), so a passive monitor that never subscribes sees
an idle system rather than a broken one.

## Upstream quirks worth knowing

Kept as-is unless a patch is listed in the section below.

* `publish_rate` is a polling wall timer (`GetCurrentFrame` + `ReceiveRTPacket`),
  not a QTM stream rate. It is *not* limited by the mocap system: QTM captures at
  whatever the lab configured. The period is `1000 / publish_rate` in **integer**
  milliseconds, so only rates dividing 1000 are achieved exactly -- 200 Hz gives
  5 ms, but 150 Hz gives 6 ms (=166 Hz), and anything >= 1000 gives a 0 ms busy
  loop. Our config sets 50. Measured against the lab QTM (capturing at 300 Hz),
  the achieved rate runs below the configured one -- 50 gives ~33 Hz and 200
  gives ~160 Hz -- with occasional 100-150 ms stalls in both.
* `header.stamp` is `rclcpp::Clock().now()` at *publish* time, not the QTM
  capture time, so it carries up to one polling period of unknown latency. The
  `frame_number` field is the reliable clock for aligning mocap against recorded
  camera frames.
* `matrixToQuaternion()` reads the 3x3 as **column-major**, matching the SDK.
  This is easy to misread as a transpose bug: against the usual row-major
  formula every sign is flipped. It is deliberate and consistent -- all four
  branches negate the antisymmetric terms and leave the symmetric ones alone,
  which is exactly the conjugate, i.e. the transposed reading. Still worth one
  known-90-deg check in the lab, but the code is not wrong.
* `on_deactivate` and `on_cleanup` call `timer_->reset()`, which on
  `rclcpp::TimerBase` *restarts* a timer rather than cancelling it, so
  deactivating keeps polling a socket `stop_qualisys()` already disconnected.
  Worse, `timer_` is only assigned in `on_activate`, so a configure -> cleanup
  transition with no activate in between dereferences a null `shared_ptr`. Do not
  cycle lifecycle states on this driver.
* `use_markers_with_id` is declared and never read.
* Requires QTM speaking RT protocol 1.23.

## Local patches

Add an entry here for every deviation from the vendored commit, so the next
upstream bump knows what to carry forward.

* **`header.stamp` carries the QTM capture time, not the publish time.** Upstream
  stamped both messages with `rclcpp::Clock().now()` at publish, which includes
  the poll round trip and the timer's scheduling delay -- and varies per sample,
  so it is jitter rather than a calibratable offset. `capture_time()` now reads
  `CRTPacket::GetTimeStamp()` instead. QTM counts microseconds from measurement
  start, so the first packet after activation anchors a `(ros, qtm)` pair and
  later stamps are that anchor plus the QTM delta: intervals come from the mocap
  clock and are exact, leaving one constant offset (roughly the round trip at
  anchor time) in place of per-sample noise. The anchor is reset in
  `on_activate`, since a new measurement restarts QTM's clock.
* **Dropped the hardcoded `use_sim_time: true`.** The constructor's default
  `NodeOptions` carried a `use_sim_time` parameter override. Overrides from
  `NodeOptions` are applied *after* the ones parsed from `--params-file`, so it
  won over any config and could not be turned off from a launch file. Nothing
  publishes `/clock` on the robot, so ROS time was pinned at 0: `capture_time()`
  anchored on `now()` == 0 and every `header.stamp` came out as time-since-QTM-
  measurement-start near the epoch (e.g. `sec: 2337`) rather than a ROS time.
  The parameter is now left at its default (false) and settable from the params
  file if the driver is ever run against a simulated clock.
* `config/qualisys_driver_params.yaml`: `host_port` -> `port` (upstream's key was
  never declared, so it was silently ignored), dropped the undeclared
  `n_markers`, and `publish_rate` 10 -> 50.
