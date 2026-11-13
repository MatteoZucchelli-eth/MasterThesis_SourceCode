# MIT License
"""Tests for the camera gate: does the episode's target land in a recorded frame?

Projection over the prepared intrinsics is tested directly; the gate decision
(plan, wait, skip) is tested through AutoCollectHandler.
"""

from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

import pinocchio as pin  # noqa: E402

from g1_common.camera import PinholeIntrinsics  # noqa: E402

from g1_control.utils.target_visibility import (  # noqa: E402
    BODY_TO_OPTICAL,
    CameraView,
    RECORDED_CAMERAS,
    RecordedCamera,
    TargetVisibility,
    VisibilityReport,
)

from g1_definitions.definitions import AutoCollectPhase  # noqa: E402

_VIEWS = tuple(camera.view for camera in RECORDED_CAMERAS)

# Wrist camera after the 1280x720 -> 720x720 centre crop (cx shifted by 280 px).
_WRIST = PinholeIntrinsics(
    fx=690.1127251872208,
    fy=690.1127251872208,
    cx=359.83226287869536,
    cy=369.7137298773904,
    width=720,
    height=720,
)


def _view(name="wrist_view", intrinsics=_WRIST):
    camera = next(c for c in RECORDED_CAMERAS if c.view == name)
    return CameraView(camera, intrinsics, declared_frame=f"{name}_optical_frame")


def _at(x, y, z):
    """A target sitting at an optical-frame point, orientation irrelevant here."""
    return pin.SE3(np.eye(3), np.array([float(x), float(y), float(z)]))


# ── Projection ────────────────────────────────────────────────────────────────


def test_a_target_on_the_optical_axis_lands_on_the_principal_point():
    check = _view().sees(_at(0.0, 0.0, 0.5))
    assert check.visible
    assert check.uv_px == pytest.approx([_WRIST.cx, _WRIST.cy])
    assert check.depth_m == pytest.approx(0.5)


def test_a_target_off_the_side_of_the_crop_is_not_visible():
    """In the sensor's 1280 px field of view but outside the 720 px crop (u ~ 774)."""
    check = _view().sees(_at(0.30, 0.0, 0.5))
    assert check.uv_px[0] > _WRIST.width - 1
    assert not check.visible


def test_a_target_behind_the_camera_is_never_visible():
    """A point behind the lens has no pixel (NaN), even if it projects to (cx, cy)."""
    check = _view().sees(_at(0.0, 0.0, -0.5))
    assert not check.visible
    assert np.isnan(check.uv_px).all()
    assert "behind the camera" in check.describe()


def test_the_margin_demands_clearance_from_the_border():
    """A centre 5 mm inside the edge is in frame; the square around it is not."""
    depth = 0.5
    # x projecting onto the right border, pulled back 5 mm.
    edge_x = (_WRIST.width - 1 - _WRIST.cx) * depth / _WRIST.fx
    x = edge_x - 0.005
    assert _view().sees(_at(x, 0.0, depth), margin_m=0.0).visible
    assert not _view().sees(_at(x, 0.0, depth), margin_m=0.01).visible


@pytest.mark.parametrize("depth", [0.4, 0.5, 1.0, 2.0])
def test_the_margin_buys_the_same_clearance_at_every_range(depth):
    """The metric margin equals the clearance from the frame edge at any depth."""
    view, margin_m = _view(), 0.01
    edge_x = (_WRIST.width - 1 - _WRIST.cx) * depth / _WRIST.fx

    lo, hi = 0.0, edge_x
    for _ in range(60):
        mid = (lo + hi) / 2
        if view.sees(_at(mid, 0.0, depth), margin_m).visible:
            lo = mid
        else:
            hi = mid

    assert edge_x - lo == pytest.approx(margin_m, abs=1e-6)


def test_the_report_is_an_or_over_the_views():
    """One camera with the square in frame is the whole requirement."""
    inside = _view("ego_view").sees(_at(0.0, 0.0, 0.5))
    outside = _view().sees(_at(0.30, 0.0, 0.5))
    assert VisibilityReport((inside, outside)).visible
    assert not VisibilityReport((outside, outside)).visible
    assert "ego_view" in VisibilityReport((inside, outside)).summary


# ── CameraInfo adoption ───────────────────────────────────────────────────────


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def _record(self, severity):
        def _log(msg, *_a, **_k):
            self.messages.append((severity, str(msg)))

        return _log

    def __getattr__(self, name):
        if name in ("debug", "info", "warn", "warning", "error"):
            return self._record("warning" if name == "warn" else name)
        raise AttributeError(name)


class _FakeNode:
    def __init__(self):
        self.logger = _FakeLogger()
        self.subscriptions_made = []

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions_made.append(topic)
        return SimpleNamespace(topic=topic, callback=callback)

    def get_logger(self):
        return self.logger


def _camera_info(width=1280, height=720, frame_id="hand_camera_optical_frame"):
    from sensor_msgs.msg import CameraInfo

    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width, msg.height = width, height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    # Sensor K: principal point shifted by the crop offset.
    cx = _WRIST.cx + (width - min(width, height)) / 2.0
    msg.k = [_WRIST.fx, 0.0, cx, 0.0, _WRIST.fy, _WRIST.cy, 0.0, 0.0, 1.0]
    return msg


class _FakeRobotModel:
    """Places camera frames without a URDF.

    ``poses`` maps a URDF frame name to its pelvis-frame SE3; absent names are
    absent from the model.
    """

    def __init__(self, poses):
        self.poses = dict(poses)
        self.cached_q = None

    def has_frame(self, name):
        return name in self.poses

    def cache_forward_kinematics(self, q, auto_clip=True):
        self.cached_q = np.asarray(q, dtype=float)

    def frame_placement(self, name):
        return self.poses[name]


def _model(**overrides):
    """A model carrying both camera mounts, identity unless overridden."""
    poses = {camera.urdf_frame: pin.SE3.Identity() for camera in RECORDED_CAMERAS}
    poses.update(overrides)
    return _FakeRobotModel(poses)


def _feed(visibility, view, msg):
    """Hand ``msg`` to the CameraInfo callback that camera's subscription carries."""
    visibility._subs[view].callback(msg)


def _feed_all(visibility):
    for view in _VIEWS:
        _feed(visibility, view, _camera_info())


def test_the_lens_comes_from_the_stream_that_published_it():
    """Intrinsics are read off the CameraInfo and adjusted for the crop."""
    node = _FakeNode()
    visibility = TargetVisibility(node, _model())
    assert set(node.subscriptions_made) == {c.info_topic for c in RECORDED_CAMERAS}
    assert visibility.pending() == list(_VIEWS)

    _feed(visibility, "wrist_view", _camera_info())
    view = visibility._views["wrist_view"]
    assert view.intrinsics.width == 720 and view.intrinsics.height == 720
    assert view.intrinsics.cx == pytest.approx(_WRIST.cx)
    assert view.intrinsics.cy == pytest.approx(_WRIST.cy)
    assert view.declared_frame == "hand_camera_optical_frame"
    assert visibility.pending() == ["ego_view"]


def test_the_head_is_projected_through_its_body_frame_not_its_declared_one():
    """The head goes via d435_link plus the published body-to-optical rotation."""
    ego = next(c for c in RECORDED_CAMERAS if c.view == "ego_view")
    wrist = next(c for c in RECORDED_CAMERAS if c.view == "wrist_view")
    assert ego.urdf_frame == "d435_link"
    assert wrist.urdf_frame == "hand_camera_optical_frame"

    published = pin.Quaternion(0.5, -0.5, 0.5, -0.5).normalized().matrix()
    np.testing.assert_allclose(BODY_TO_OPTICAL, published, atol=1e-15)
    # The wrist frame is already optical in the URDF.
    np.testing.assert_allclose(wrist.optical_rotation, np.eye(3), atol=0.0)


def test_the_mount_rotation_is_composed_onto_the_fk_pose():
    """A body-frame placement comes back turned into the optical convention."""
    ego = next(c for c in RECORDED_CAMERAS if c.view == "ego_view")
    body = pin.SE3(np.eye(3), np.array([0.05, 0.02, 0.43]))
    placement = ego.optical_placement(_model(d435_link=body))
    np.testing.assert_allclose(placement.rotation, BODY_TO_OPTICAL, atol=1e-15)
    np.testing.assert_allclose(placement.translation, body.translation, atol=0.0)


def test_a_camera_with_no_mount_frame_is_dropped_rather_than_waited_on():
    """A camera without a URDF mount frame is dropped, not left pending forever."""
    node = _FakeNode()
    model = _model()
    del model.poses["hand_camera_optical_frame"]
    visibility = TargetVisibility(node, model)
    assert [c.view for c in visibility.cameras] == ["ego_view"]
    assert any("hand_camera_optical_frame" in msg for _, msg in node.logger.messages)

    _feed(visibility, "ego_view", _camera_info())
    report = visibility.check(np.zeros(3), _at(0.0, 0.0, 0.5))
    assert report is not None
    assert [check.view for check in report.checks] == ["ego_view"]


def test_a_model_with_no_camera_frames_at_all_leaves_nothing_to_check():
    node = _FakeNode()
    visibility = TargetVisibility(node, _FakeRobotModel({}))
    assert visibility.cameras == ()
    assert visibility.check(np.zeros(3), _at(0.0, 0.0, 0.5)) is None
    assert "no camera has a mount frame" in visibility.waiting_on()


def test_a_check_is_undecidable_until_every_camera_has_described_itself():
    """The check returns None until every camera has a CameraInfo."""
    visibility = TargetVisibility(_FakeNode(), _model())
    assert visibility.check(np.zeros(3), _at(0.0, 0.0, 0.5)) is None
    _feed(visibility, "wrist_view", _camera_info())
    assert visibility.check(np.zeros(3), _at(0.0, 0.0, 0.5)) is None
    assert "no CameraInfo yet" in visibility.waiting_on()
    assert "/camera/d435/color/camera_info" in visibility.waiting_on()


def test_the_target_is_projected_into_every_view_at_the_given_configuration():
    """Both cameras are placed by one FK pass at the configuration handed in."""
    model = _model()
    visibility = TargetVisibility(_FakeNode(), model)
    _feed_all(visibility)

    q = np.array([0.1, -0.2, 0.3])
    report = visibility.check(q, _at(0.0, 0.0, 0.5))
    assert report is not None
    np.testing.assert_allclose(model.cached_q, q)
    assert {check.view for check in report.checks} == set(_VIEWS)


def test_a_target_the_wrist_can_see_and_the_head_cannot_still_passes():
    """The head is rotated away, so only the wrist has the square in frame."""
    # Wrist at origin looking +z; head 1 m aside facing +x.
    facing_x = pin.SE3(BODY_TO_OPTICAL.T, np.array([0.0, 1.0, 0.0]))
    visibility = TargetVisibility(_FakeNode(), _model(d435_link=facing_x))
    _feed_all(visibility)

    report = visibility.check(np.zeros(3), _at(0.0, 0.0, 0.5))
    assert report is not None and report.visible
    by_view = {check.view: check for check in report.checks}
    assert by_view["wrist_view"].visible
    assert not by_view["ego_view"].visible


# ── The gate around it ────────────────────────────────────────────────────────


class _StubVisibility:
    """Stands in for the projector so the gate's decisions can be driven."""

    def __init__(self, report):
        self.report = report
        self.cameras = RECORDED_CAMERAS

    def check(self, *_a, **_k):
        return self.report

    def waiting_on(self):
        return "no CameraInfo yet on /camera/hand/color/camera_info"


def _handler_class():
    """AutoCollectHandler, or a skip when optional package deps are missing."""
    try:
        from g1_control.states.auto_collect import AutoCollectHandler
    except ModuleNotFoundError as exc:
        pytest.skip(f"g1_control.states unavailable: {exc}")
    return AutoCollectHandler


def _gate(report, *, loop):
    """A handler carrying only what the camera gate reads."""
    handler = object.__new__(_handler_class())
    handler._ctx = SimpleNamespace(
        full_q=np.zeros(3), drill_target_frame="red_square_target"
    )
    handler._logger = _FakeLogger()
    handler._visibility = _StubVisibility(report)
    handler._visibility_margin_m = 0.01
    handler._visibility_skips = 0
    handler._loop = loop
    handler._phase = AutoCollectPhase.TRAJECTORY_GENERATION
    handler._capture_active = False
    handler._clock = lambda: 0.0
    handler._phase_pub = SimpleNamespace(publish=lambda _msg: None)
    handler.resets = []
    handler._reset_request_pub = SimpleNamespace(
        publish=lambda msg: handler.resets.append(msg)
    )
    return handler


def _report(visible):
    point = _at(0.0, 0.0, 0.5) if visible else _at(0.30, 0.0, 0.5)
    return VisibilityReport((_view().sees(point),))


_TARGET = _at(0.0, 0.0, 0.5)


def test_a_visible_target_clears_the_gate_and_clears_the_counter():
    gate = _gate(_report(True), loop=True)
    gate._visibility_skips = 4
    assert gate._target_in_shot(_TARGET) is True
    assert gate._visibility_skips == 0
    assert not gate.resets, "a passing configuration must not reset the scene"
    assert gate._phase is AutoCollectPhase.TRAJECTORY_GENERATION


def test_an_off_camera_target_is_skipped_and_costs_a_scene_not_an_episode():
    gate = _gate(_report(False), loop=True)
    assert gate._target_in_shot(_TARGET) is False
    assert gate._visibility_skips == 1
    assert len(gate.resets) == 1, "loop mode must ask for a fresh scene"
    assert gate._phase is AutoCollectPhase.HOLDING, "nothing may move without a plan"
    warnings = [msg for severity, msg in gate._logger.messages if severity == "warning"]
    assert any("skipping this configuration" in msg for msg in warnings)
    assert any("red_square_target" in msg for msg in warnings)


def test_a_skip_outside_loop_mode_parks_without_touching_the_sim():
    """Interactively there is no scene to re-roll — say so and hold."""
    gate = _gate(_report(False), loop=False)
    assert gate._target_in_shot(_TARGET) is False
    assert not gate.resets
    assert gate._phase is AutoCollectPhase.HOLDING


def test_repeated_skips_escalate_to_an_error():
    """A scene where nothing is ever in shot must not hide in the warnings."""
    gate = _gate(_report(False), loop=True)
    from g1_control.states.auto_collect import _VISIBILITY_ALARM_SKIPS

    for _ in range(_VISIBILITY_ALARM_SKIPS - 1):
        gate._target_in_shot(_TARGET)
    assert not [msg for severity, msg in gate._logger.messages if severity == "error"]
    gate._target_in_shot(_TARGET)
    assert [msg for severity, msg in gate._logger.messages if severity == "error"]


def test_an_undecidable_check_waits_instead_of_skipping():
    """No CameraInfo yet is not a verdict — hold the phase and retry next tick."""
    gate = _gate(None, loop=True)
    assert gate._target_in_shot(_TARGET) is False
    assert gate._visibility_skips == 0, "counted a wait as a bad configuration"
    assert not gate.resets, "re-rolled the scene over the controller's own startup"
    assert gate._phase is AutoCollectPhase.TRAJECTORY_GENERATION


def test_the_gate_is_a_no_op_when_the_check_is_disabled():
    gate = _gate(_report(False), loop=True)
    gate._visibility = None
    assert gate._target_in_shot(_TARGET) is True
    assert not gate.resets
