# MIT License
"""Debug/visualization topic names internal to g1_control.

Cross-package topics live in :mod:`g1_definitions.topics`; these are only a
contract between the controller manager and its own debug tooling.
"""

TOPIC_VLA_CHUNK_RIGHT = "vla_debug/right_palm_chunk"
TOPIC_VLA_CHUNK_RIGHT_PREV = "vla_debug/right_palm_chunk_prev"
"""Latest / previous VLA drill-tip chunks (Float32MultiArray), for RViz path
debugging. Right tip only."""

TOPIC_VLA_TIP_TARGET = "vla_debug/tip_target_pose"
"""Fine-role (detector) predicted target, 9D pose in the drill-tip frame
(Float32MultiArray)."""

TOPIC_VLA_TIP_TARGET_DRIVER = "vla_debug/tip_target_pose_driver"
"""Coarse-role (driving model) tip_target head, 9D pose in the drill-tip frame
(Float32MultiArray); scored as ``drv_pred_rel_*`` in the metrics CSV. Empty
without an estimate."""

TOPIC_VLA_TIP_TARGET_FUSED = "vla_debug/tip_target_pose_fused"
"""Batch/filtered target estimate, 9D pose in the pelvis frame
(Float32MultiArray): an EMA over anchored predictions, or in VLA_PLANNER the
accumulator's batch mean the finalize leg was planned on (frozen per leg)."""

TOPIC_VLA_TIP_TARGET_LIVE = "vla_debug/tip_target_pose_live"
"""The finalize leg's live target, 9D pose in the pelvis frame
(Float32MultiArray): the planned estimate plus the rate-limited, clamped
correction. Published even when ``vla_planner_finalize_correct_target`` is off."""

TOPIC_VLA_TIP_TARGET_LIVE_FINE_ONLY = "vla_debug/tip_target_pose_live_fine_only"
"""The live target computed on the detector's own depth instead of the donated
range (Float32MultiArray, pelvis frame). Never applied; published only with
``vla_planner_depth_source: stereo`` and ``vla_planner_finalize_shadow_fine_only``."""

TOPIC_VLA_PLANNER_TRAJECTORY = "vla_planner/trajectory"
TOPIC_VLA_PLANNER_MARKERS = "vla_planner/markers"
"""Finalize-phase planned tip trajectory and estimated-target markers (latched)."""

TOPIC_IK_GOAL_MARKERS = "ik_debug/goal_pose"
"""Goal-pose state markers (MarkerArray, latched)."""

TOPIC_FK_CALIBRATION_POSE = "fk_calibration/averaged_pose"
"""Averaged AprilTag pose (PoseStamped), expressed in the pelvis frame."""

TOPIC_FK_CALIBRATION_START = "fk_calibration/start"
"""Start one fresh FK calibration round (std_msgs/Empty)."""

TOPIC_FK_CALIBRATION_HAND_OVERLAY = "fk_calibration/hand_overlay/compressed"
TOPIC_FK_CALIBRATION_HEAD_OVERLAY = "fk_calibration/head_overlay/compressed"
"""Compressed RViz images with tag (red) and drill-tip (green) crosses."""

TOPIC_PROGRESS_TEST_WAYPOINTS = "progress_test/waypoints"
"""Progress-test harness waypoint markers (MarkerArray, latched)."""

TOPIC_PERCEPTION_DETECTION = "perception/detection"
"""Latest fine perception detection (Float32MultiArray, VLA_PLANNER): flat
``[u_px, v_px, visibility, visible, depth_m, drill_u, drill_v, servo_target_u,
servo_target_v]`` in the wrist image. Optional points are NaN; empty when
nothing can be drawn."""

TOPIC_PERCEPTION_TARGET_OVERLAY = "perception/target_overlay"
"""Wrist camera frame with the perception target pixel drawn on it, for RViz.
Published only as a CompressedImage on ``<this>/compressed`` (sensor QoS)."""
