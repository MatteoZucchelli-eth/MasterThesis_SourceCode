# MIT License
"""Topic names shared across the g1 packages."""

TOPIC_LOWCMD = "/lowcmd"
"""LowCmd Topic Name"""

TOPIC_LOWSTATE = "/lowstate"
"""LowState Topic Name"""

TOPIC_LOWSTATE_THROTTLED = "/lf/lowstate"
"""Low Frequency LowState Topic Name"""

TOPIC_HANDCMD_LEFT = "/dex3/left/cmd"
"""Left Hand HandCmd Topic Name"""

TOPIC_HANDSTATE_LEFT = "/dex3/left/state"
"""Left Hand HandState Topic Name"""

TOPIC_HANDCMD_RIGHT = "/dex3/right/cmd"
"""Right Hand HandCmd Topic Name"""

TOPIC_HANDSTATE_RIGHT = "/dex3/right/state"
"""Right Hand HandState Topic Name"""

TOPIC_FSMSTATE = "fsm"
"""FSM State Topic Name"""

TOPIC_FSM_TRIGGER = "fsm_trigger"
"""External FSM transition trigger (std_msgs/String transition name, e.g.
``"teleoperation"``), equivalent to an RC button press."""

TOPIC_AUTO_COLLECT_PHASE = "auto_collect/phase"
"""Phase command for the AUTO_COLLECT state (std_msgs/String holding an
:class:`AutoCollectPhase` name, e.g. ``"approaching"``). Case-insensitive; the
numeric value of the enum is also accepted."""

TOPIC_AUTO_COLLECT_PHASE_STATUS = "auto_collect/phase_status"
"""Phase the AUTO_COLLECT state is currently in (std_msgs/String, latched)."""

TOPIC_VLA_PLANNER_PHASE_STATUS = "vla_planner/phase_status"
"""Phase the VLA_PLANNER state is currently in (std_msgs/String, latched,
lower-case :class:`VlaPlannerPhase` name). The state sequences its own phases."""

TOPIC_VLA_PLANNER_COMPLETE = "vla_planner/complete"
"""Whether VLA_PLANNER has reached completion (std_msgs/Bool, latched): the
finalize pose held inside ``vla_planner_accumulate_settle_tol_mm`` for
``vla_planner_complete_hold_s``.

Republished ``False`` on every state entry. Reflects the planner's own estimate;
actual success is scored offline by ``vla_success``."""

TOPIC_SIM_RESET = "sim/reset"
"""Emitted by the simulation (std_msgs/Empty) after a reset has teleported the
robot and resampled the targets. AUTO_COLLECT retracts to its start pose on it."""

TOPIC_SIM_RESET_REQUEST = "sim/reset_request"
"""Request a scene reset (std_msgs/Empty), equivalent to the reset key. Answered
with :data:`TOPIC_SIM_RESET`."""

TOPIC_SIM_EVAL_SCENARIO = "sim/eval_scenario"
"""Active evaluation scenario (std_msgs/String, latched). JSON payload
``{"scenario": <id>, "pose_level": ..., "target_level": ...}`` on V-press;
``{"scenario": -1}`` on R-press. MetricRecorder tags its CSVs with it."""

TOPIC_SIM_EVAL_ADVANCE_REQUEST = "sim/eval_advance_request"
"""Advance to the next evaluation scenario (std_msgs/Empty), equivalent to V."""

TOPIC_CAMERA = "/camera/d435/color/image_raw"
"""Head (torso d435) camera image topic name."""

TOPIC_CAMERA_INFO = "/camera/d435/color/camera_info"
"""Head camera intrinsics + distortion, published alongside TOPIC_CAMERA."""

TOPIC_CAMERA_WRIST = "/camera/hand/color/image_raw"
"""Wrist (drill-mounted hand_cam) camera image topic name."""

TOPIC_CAMERA_WRIST_INFO = "/camera/hand/color/camera_info"
"""Wrist camera intrinsics + distortion; source of truth for wrist geometry on sim and
robot."""

TOPIC_TASKINSTRUCTION = "task"
"""Task Instruction Topic Name"""

TOPIC_RECORDED_STEP = "/controller_manager/recorded_step"
"""Synchronized state+action snapshot (g1_interfaces/ControlStep) emitted by the
control loop each tick, consumed by the dataset recorder."""

TOPIC_RECORD_EPISODE = "/record_episode"
"""Episode lifecycle commands for the recorder (std_msgs/UInt8:
1 start, 2 freeze, 0 save, 3 discard)."""

TOPIC_CAMERA_DEPTH = "/camera/d435/depth/image_rect_raw"
"""Head (torso d435) rectified depth image, same name on sim and robot.

Encoding: 32FC1 metres (Isaac) or 16UC1 millimetres (realsense2_camera); read
it off the message (g1_common.depth.DEPTH_SCALES). On the robot this is the
unregistered IR stream in ``d435_depth_optical_frame``; prefer
:data:`TOPIC_CAMERA_DEPTH_ALIGNED` there."""

TOPIC_CAMERA_DEPTH_ALIGNED = "/camera/d435/aligned_depth_to_color/image_raw"
"""Head depth registered onto the colour image (16UC1 mm, ``d435_color_optical_frame``).
Robot only.

Pair with :data:`TOPIC_CAMERA_DEPTH_ALIGNED_INFO`. In sim :data:`TOPIC_CAMERA_DEPTH`
is already registered.
"""

TOPIC_CAMERA_DEPTH_ALIGNED_INFO = "/camera/d435/aligned_depth_to_color/camera_info"
"""Intrinsics for :data:`TOPIC_CAMERA_DEPTH_ALIGNED` (equal to the colour K while
alignment is on). Robot only."""

TOPIC_CAMERA_DEPTH_INFO = "/camera/d435/color/camera_info"
"""Intrinsics to deproject :data:`TOPIC_CAMERA_DEPTH` with (the colour camera_info).

Valid in sim only, where depth is registered to colour. On the robot use
:data:`TOPIC_CAMERA_DEPTH_ALIGNED` / :data:`TOPIC_CAMERA_DEPTH_ALIGNED_INFO`
(wall_plane node parameters, set by real_launch.py)."""

TOPIC_WALL_POINTS = "/wall/points"
"""Decimated point cloud (sensor_msgs/PointCloud2) the wall plane was fitted
from, in the plane's frame. Visualisation only."""

TOPIC_WALL_PLANE = "/wall/plane"
"""The fitted wall plane (g1_interfaces/WallPlane)."""

TOPIC_WALL_PLANE_MARKER = "/wall/plane_marker"
"""RViz visualisation of :data:`TOPIC_WALL_PLANE` (visualization_msgs/Marker): a
disc on the fitted plane plus an arrow along its normal. Visualisation only."""
