# MIT License
"""Unit tests for the perception back-projection primitives (numpy only, no ROS)."""

import logging
from contextlib import contextmanager

import numpy as np

from g1_control.vla.perception import (
    backproject_uv,
    CameraIntrinsics,
    PerceptionDetection,
    target_pose_9d_in_camera,
    target_pose_in_camera,
)
from g1_control.vla.tip_target import DetectionSlot, TipTargetLifter
from tf2_ros import TransformException

# Hand camera: /camera/hand/color/camera_info (720x720, zero distortion).
_INTR = CameraIntrinsics(
    fx=388.0903405714017,
    fy=388.0903405714017,
    cx=360.0,
    cy=360.0,
    width=720,
    height=720,
)


class _NoTransforms:
    """A TF buffer that never has the transform, as tf2 reports it."""

    def lookup_transform(self, *_args, **_kwargs):
        raise TransformException("no such transform")


@contextmanager
def _captured_logs(name: str, level: int = logging.INFO):
    """Collect one logger's messages (caplog is not available everywhere)."""
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger(name)
    handler, previous = _Collect(level), logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _project(xyz, intr):
    """Forward pinhole projection (inverse of backproject_uv), for round-trips."""
    x, y, z = xyz
    return np.array([intr.fx * x / z + intr.cx, intr.fy * y / z + intr.cy])


def test_from_k_reads_intrinsics():
    k = [388.09, 0.0, 360.0, 0.0, 388.09, 360.0, 0.0, 0.0, 1.0]
    intr = CameraIntrinsics.from_k(k, width=720, height=720)
    assert intr.fx == 388.09 and intr.fy == 388.09
    assert intr.cx == 360.0 and intr.cy == 360.0


def test_backproject_principal_point_is_on_axis():
    # A pixel at the principal point lies on the optical axis: X=Y=0, Z=depth.
    xyz = backproject_uv(np.array([360.0, 360.0]), 0.25, _INTR)
    assert np.allclose(xyz, [0.0, 0.0, 0.25])


def test_backproject_sign_and_scale():
    # +u is +X, +v is +Y (optical frame), scaling linearly with depth.
    xyz = backproject_uv(np.array([360.0 + _INTR.fx, 360.0 - _INTR.fy]), 1.0, _INTR)
    assert np.allclose(xyz, [1.0, -1.0, 1.0])


def test_backproject_roundtrip():
    for xyz in ([0.05, -0.03, 0.20], [-0.11, 0.07, 0.15], [0.0, 0.0, 0.5]):
        xyz = np.array(xyz)
        uv = _project(xyz, _INTR)
        recovered = backproject_uv(uv, xyz[2], _INTR)
        assert np.allclose(recovered, xyz, atol=1e-9)


def test_target_pose_position_matches_backprojection():
    det = PerceptionDetection(
        uv_px=np.array([400.0, 300.0]),
        visibility=0.9,
        visible=True,
        depth_m=0.18,
        wall_normal_cam=np.array([0.3, -0.2, 0.93]),
    )
    mat = target_pose_in_camera(det, _INTR)
    assert np.allclose(mat[:3, 3], backproject_uv(det.uv_px, det.depth_m, _INTR))


def test_target_pose_x_axis_is_wall_normal_and_orthonormal():
    normal = np.array([0.3, -0.2, 0.93])
    det = PerceptionDetection(
        uv_px=np.array([360.0, 360.0]),
        visibility=1.0,
        visible=True,
        depth_m=0.2,
        wall_normal_cam=normal,
    )
    R = target_pose_in_camera(det, _INTR)[:3, :3]
    # x-axis is the (normalised) wall normal.
    assert np.allclose(R[:, 0], normal / np.linalg.norm(normal), atol=1e-9)
    # Columns orthonormal, right-handed.
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_target_pose_x_axis_handles_normal_parallel_to_reference():
    # Normal along the +y reference must still yield an orthonormal frame.
    det = PerceptionDetection(
        uv_px=np.array([360.0, 360.0]),
        visibility=1.0,
        visible=True,
        depth_m=0.2,
        wall_normal_cam=np.array([0.0, 1.0, 0.0]),
    )
    R = target_pose_in_camera(det, _INTR)[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-9)


def test_pose_9d_layout():
    det = PerceptionDetection(
        uv_px=np.array([420.0, 330.0]),
        visibility=1.0,
        visible=True,
        depth_m=0.16,
        wall_normal_cam=np.array([0.1, 0.0, 1.0]),
    )
    pose_9d = target_pose_9d_in_camera(det, _INTR)
    mat = target_pose_in_camera(det, _INTR)
    assert pose_9d.shape == (9,)
    assert np.allclose(pose_9d[:3], mat[:3, 3])  # xyz
    assert np.allclose(pose_9d[3:6], mat[:3, 0])  # rot6d col 0
    assert np.allclose(pose_9d[6:9], mat[:3, 1])  # rot6d col 1


def _outputs(**overrides) -> dict:
    """A perception head's outputs, with the batch dims the servers prepend."""
    return {
        "target_uv_px": np.array([[400.0, 300.0]]),  # (1, 2)
        "target_uv_processed_px": np.array([[366.0, 271.0]]),
        "visibility": np.array([0.87]),
        "visible": np.array([True]),
        "target_depth_m": np.array([0.164]),
        "wall_normal_cam": np.array([[0.3, -0.2, 0.93]]),  # (1, 3)
        **overrides,
    }


def test_from_outputs_squeezes_batch_dims():
    det = PerceptionDetection.from_outputs(_outputs())
    assert det.uv_px.shape == (2,) and np.allclose(det.uv_px, [400.0, 300.0])
    assert det.visible is True
    assert np.isclose(det.visibility, 0.87)
    assert np.isclose(det.depth_m, 0.164)
    assert det.wall_normal_cam.shape == (3,)


def test_from_outputs_defaults_visible_when_the_model_has_no_such_head():
    """Without a ``visible`` head, the confidence threshold is the only gate."""
    outputs = _outputs()
    del outputs["visible"]
    assert PerceptionDetection.from_outputs(outputs).visible is True


def test_present_in_ignores_the_optional_visible_key():
    outputs = _outputs()
    del outputs["visible"]
    assert PerceptionDetection.present_in(outputs)
    assert not PerceptionDetection.present_in({"right_drill_tip_eef_9d": 0.5})
    assert not PerceptionDetection.present_in(None)


def test_present_in_rejects_a_partial_perception_head():
    outputs = _outputs()
    del outputs["target_depth_m"]
    assert not PerceptionDetection.present_in(outputs)


def test_from_reply_finds_a_detector_only_models_head_in_the_action_dict():
    det = PerceptionDetection.from_reply(_outputs(), {})
    assert det is not None and np.allclose(det.uv_px, [400.0, 300.0])


def test_from_reply_finds_an_action_models_head_in_info():
    """The same model can serve the perception role: its actions sit in the
    action dict and the detection in info, and the client takes the detection."""
    actions = {"right_drill_tip_eef_9d": np.zeros((1, 30, 9))}
    det = PerceptionDetection.from_reply(actions, _outputs())
    assert det is not None and np.allclose(det.uv_px, [400.0, 300.0])


def test_from_reply_returns_none_when_no_head_is_present():
    actions = {"right_drill_tip_eef_9d": np.zeros((1, 30, 9))}
    assert PerceptionDetection.from_reply(actions, {}) is None
    assert PerceptionDetection.from_reply(None, None) is None


def test_lifter_lifts_every_detection_when_no_confidence_floor_is_set():
    """``min_visibility`` defaults to 0.0, which gates nothing."""
    lifter = TipTargetLifter(
        tf_buffer=_NoTransforms(),
        camera_frame="hand_camera_optical_frame",
        tip_frame="drill_tip_link",
        intrinsics=_INTR,
    )
    disowned = PerceptionDetection.from_outputs(
        _outputs(visibility=np.array([0.0]), visible=np.array([False]))
    )
    assert not hasattr(lifter, "gated_out")
    # Reaches the (absent) camera->tip TF, i.e. past any confidence gate.
    with _captured_logs("g1_control.vla.tip_target") as messages:
        assert lifter.lift(disowned) is None
    assert any("TF not available" in m for m in messages)
    assert not any("gated out" in m for m in messages)


def test_lifter_drops_a_detection_under_the_confidence_floor():
    """With a floor set, confidence is judged before the camera geometry."""
    lifter = TipTargetLifter(
        tf_buffer=_NoTransforms(),
        camera_frame="hand_camera_optical_frame",
        tip_frame="drill_tip_link",
        intrinsics=_INTR,
        min_visibility=0.9,
    )
    unsure = PerceptionDetection.from_outputs(_outputs(visibility=np.array([0.42])))
    # The gate logs at DEBUG.
    with _captured_logs("g1_control.vla.tip_target", logging.DEBUG) as messages:
        assert lifter.lift(unsure) is None
    assert any("gated out" in m for m in messages)
    assert not any("TF not available" in m for m in messages)


def test_lifter_floor_is_inclusive_at_the_threshold():
    """A detection exactly at the floor is kept: the gate is `<`, not `<=`."""
    lifter = TipTargetLifter(
        tf_buffer=_NoTransforms(),
        camera_frame="hand_camera_optical_frame",
        tip_frame="drill_tip_link",
        intrinsics=_INTR,
        min_visibility=0.9,
    )
    at_floor = PerceptionDetection.from_outputs(_outputs(visibility=np.array([0.9])))
    with _captured_logs("g1_control.vla.tip_target", logging.DEBUG) as messages:
        assert lifter.lift(at_floor) is None  # blocked by the absent TF, not the floor
    assert not any("gated out" in m for m in messages)


def test_detection_slot_counts_every_detection():
    """The count tells "inferring" from "not inferring" at a glance."""
    slot = DetectionSlot()
    assert slot.count == 0 and slot.latest() is None

    weak = PerceptionDetection.from_outputs(_outputs(visibility=np.array([0.18])))
    slot.publish(weak)
    slot.publish(weak)
    # Counted even though both would be gated out downstream.
    assert slot.count == 2 and slot.latest() is weak

    slot.reset()
    assert slot.count == 0 and slot.latest() is None


class _FixedTransform:
    """A TF buffer holding one transform; rotation given as a matrix."""

    def __init__(self, rotation, translation):
        self._rotation = np.asarray(rotation, dtype=float)
        self._translation = np.asarray(translation, dtype=float)

    def lookup_transform(self, *_args, **_kwargs):
        import pinocchio as pin

        quaternion = pin.Quaternion(self._rotation)
        return type(
            "TF",
            (),
            {
                "transform": type(
                    "T",
                    (),
                    {
                        "translation": type(
                            "V",
                            (),
                            dict(zip("xyz", self._translation)),
                        )(),
                        "rotation": type(
                            "Q",
                            (),
                            {
                                "x": quaternion.x,
                                "y": quaternion.y,
                                "z": quaternion.z,
                                "w": quaternion.w,
                            },
                        )(),
                    },
                )()
            },
        )()


def _wrist_lifter():
    """A lifter whose camera sits 140 mm behind the tip, looking along the bit.

    T(tip <- cam): optical +z = tip +x, +x = tip +y, +y = tip +z (the wrist
    mount, straightened).
    """
    rotation = np.column_stack(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
    return TipTargetLifter(
        tf_buffer=_FixedTransform(rotation, [-0.14, 0.0, 0.0]),
        camera_frame="hand_camera_optical_frame",
        tip_frame="drill_tip_link",
        intrinsics=CameraIntrinsics(
            fx=690.0, fy=690.0, cx=360.0, cy=360.0, width=720, height=720
        ),
    )


def test_projecting_a_tip_frame_point_is_the_pose_projection_without_a_pose():
    """The two must agree exactly — one is the other's translation."""
    lifter = _wrist_lifter()
    point = [0.005, 0.002, -0.001]
    by_point = lifter.project_tip_point(point)
    by_pose = lifter.project_tip_pose(np.concatenate([point, [1, 0, 0], [0, 1, 0]]))
    assert np.allclose(by_point.uv_px, by_pose.uv_px)
    assert by_point.depth_m == by_pose.depth_m


def test_the_pixel_scale_a_lateral_miss_is_judged_on():
    """~0.21 mm/px at the wrist geometry: a 2 mm lateral miss is ~9.5 px.

    Both points share a station along the drill axis, so no parallax enters.
    """
    lifter = _wrist_lifter()
    target = lifter.project_tip_point([0.005, 0.002, 0.0])
    entry = lifter.project_tip_point([0.005, 0.0, 0.0])
    assert np.allclose(entry.uv_px, [360.0, 360.0])
    assert abs(float(np.linalg.norm(target.uv_px - entry.uv_px)) - 9.52) < 0.05
    # 1 px ~ depth/fx, so the scale holds only at this fixed depth.
    assert abs(entry.depth_m - 0.145) < 1e-9


def test_a_point_behind_the_lens_projects_to_a_negative_depth_not_a_guess():
    """A point behind the lens yields negative depth (the planner drops the pixel)."""
    lifter = _wrist_lifter()
    assert lifter.project_tip_point([-0.2, 0.0, 0.0]).depth_m < 0.0
