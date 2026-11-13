# MIT License
"""VLA planner phase and routing tests (no ROS, stubbed context)."""

import numpy as np
import pinocchio as pin

from g1_control.states.vla_planner import FinalizingPhase, freeze_arm


class _Commander:
    """Just the commander members the freeze path touches."""

    def __init__(self):
        self.commanded_upper_body = None

    def command_upper_body(self, target):
        self.commanded_upper_body = np.asarray(target, dtype=float)

    def reset(self):
        self.commanded_upper_body = None


class _Actions:
    def __init__(self, waypoint):
        self.has_first_chunk = True
        self._waypoint = np.asarray(waypoint, dtype=float)

    def get(self, _key):
        return self._waypoint


class _NoTf:
    """A TF buffer that never resolves (robot: no ``{colour}_{shape}_target`` frame)."""

    def lookup_transform(self, *_args, **_kwargs):
        from tf2_ros import TransformException

        raise TransformException("no such frame in this test")


class _TargetTf:
    """A TF buffer holding one target at ``position`` with identity rotation, as Isaac
    publishes it."""

    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def lookup_transform(self, _source, _target, _time):
        vec = type("V", (), dict(zip("xyz", map(float, self.position))))()
        rot = type("R", (), {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})()
        return type(
            "T",
            (),
            {"transform": type("X", (), {"translation": vec, "rotation": rot})()},
        )()


class _Ctx:
    def __init__(self, measured):
        self.robot_state = type("S", (), {"upper_body_target_chunk": object()})()
        self.targets = type("T", (), {"left_9d": object()})()
        self.full_q = np.asarray(measured, dtype=float)
        self.upper_body_indices = list(range(len(measured)))
        # FinalizingPhase reads both to log the plan against sim ground truth.
        self.tf_buffer = _NoTf()
        self.drill_target_frame = "red_square_target"


def test_freeze_keeps_a_pose_the_commander_is_already_holding():
    """A stop entered from another stop must not move on a stale waypoint."""
    ctx = _Ctx([9.0, 9.0])
    commander = _Commander()
    commander.command_upper_body([1.0, 1.0])  # e.g. the finalize end pose
    freeze_arm(
        ctx,
        commander,
        _Actions([5.0, 5.0]),
        live_upper_body_command=lambda: np.array([3.0, 4.0]),
    )
    np.testing.assert_allclose(commander.commanded_upper_body, [1.0, 1.0])
    assert ctx.robot_state.upper_body_target_chunk is None


def test_freeze_takes_the_commanded_waypoint_not_the_measured_pose():
    """The measured pose would bake gravity droop into the hold."""
    ctx = _Ctx([9.0, 9.0])
    commander = _Commander()
    freeze_arm(ctx, commander, _Actions([1.0, 1.0]))
    np.testing.assert_allclose(commander.commanded_upper_body, [1.0, 1.0])


def test_freeze_prefers_the_live_interpolator_command_to_the_chunk_pointer():
    """The live motor command is the only continuous target at handover; the chunk index
    runs on its own clock."""
    ctx = _Ctx([9.0, 9.0])
    commander = _Commander()
    freeze_arm(
        ctx,
        commander,
        _Actions([1.0, 1.0]),
        live_upper_body_command=lambda: np.array([3.0, 4.0]),
    )
    np.testing.assert_allclose(commander.commanded_upper_body, [3.0, 4.0])


def test_freeze_falls_back_to_the_measured_pose_before_any_chunk():
    ctx = _Ctx([7.0, 7.0])
    commander = _Commander()
    actions = _Actions([1.0, 1.0])
    actions.has_first_chunk = False
    freeze_arm(ctx, commander, actions)
    np.testing.assert_allclose(commander.commanded_upper_body, [7.0, 7.0])


def test_finalize_starts_at_the_held_command_not_the_drooped_measurement():
    """Keeping integral torque is bumpless only if the position command is too."""

    class _Model:
        def cache_forward_kinematics(self, q, auto_clip=False):
            self.q = np.asarray(q, dtype=float)

        def frame_placement(self, _frame):
            return pin.SE3(np.eye(3), np.array([self.q[0], 0.0, 0.0]))

    ctx = _Ctx([9.0, 9.0])  # measured FK is deliberately far from the command
    ctx.robot_model = _Model()
    commander = _Commander()
    commander.command_upper_body([1.0, 1.0])
    phase = FinalizingPhase.__new__(FinalizingPhase)
    phase._ctx = ctx
    phase._commander = commander

    np.testing.assert_allclose(phase._trajectory_start_pose().translation, [1, 0, 0])

    commander.reset()
    np.testing.assert_allclose(phase._trajectory_start_pose().translation, [9, 0, 0])


# ── Shared fixtures for the handler-level tests below ────────────────────────

import pytest  # noqa: E402

from g1_control.states.vla_planner import VlaPlannerHandler  # noqa: E402
from g1_control.vla.handover import HandoverEvidence  # noqa: E402
from g1_control.vla.perception import CameraIntrinsics, FrameProjection  # noqa: E402
from g1_control.vla.planner import FinalizeResidual, VlaPlannerConfig  # noqa: E402

_INTR = CameraIntrinsics(fx=690.0, fy=690.0, cx=360.0, cy=360.0, width=720, height=720)


def _pose_9d(position):
    return np.concatenate([np.asarray(position, float), [1, 0, 0], [0, 1, 0]])


def _anchored(position, seq=1):
    return (_pose_9d(position), _pose_9d([0.0, 0.0, 0.0]), 0.0, seq)


class _Projector:
    """Puts every estimate at a healthy on-axis point unless told otherwise."""

    def __init__(self, point_cam=(0.0, 0.0, 0.25)):
        self.point_cam = np.asarray(point_cam, float)

    def project_tip_pose(self, _pred_tip_9d):
        return FrameProjection.of(self.point_cam, _INTR)


def _fire(gate, evidence_for_seq, seqs=range(1, 6)):
    """Run the gate past its debounce; return whether it ended up fired."""
    for seq in seqs:
        gate.update(evidence_for_seq(seq))
    return gate.fired


def test_an_unknown_depth_source_is_refused_rather_than_defaulted():
    """A typo must not silently select the detector's range."""
    cfg = VlaPlannerConfig(depth_source="corase")
    with pytest.raises(ValueError, match="depth_source"):
        VlaPlannerHandler._depth_donor(cfg, _Projector())


def test_the_donor_reads_the_extrinsic_the_lift_used():
    """The donor shares the lifter's extrinsic instead of doing its own TF lookup."""

    class _Lifter(_Projector):
        def extrinsic_tip_cam(self):
            return np.eye(4)

    cfg = VlaPlannerConfig(depth_source="stereo")
    donor = VlaPlannerHandler._depth_donor(cfg, _Lifter(), lambda: None)
    assert donor is not None
    np.testing.assert_allclose(donor._extrinsic_provider(), np.eye(4))
    assert VlaPlannerHandler._depth_donor(VlaPlannerConfig(), _Lifter()) is None


def test_stereo_selects_the_measured_wall_donor():
    from g1_control.vla.planner import StereoDepthDonor

    class _Lifter(_Projector):
        def extrinsic_tip_cam(self):
            return np.eye(4)

    cfg = VlaPlannerConfig(depth_source="stereo")
    donor = VlaPlannerHandler._depth_donor(cfg, _Lifter(), lambda: None)
    assert isinstance(donor, StereoDepthDonor)


def test_stereo_without_a_plane_source_is_refused_not_degraded_to_fine():
    """A missing plane source is a wiring error, not a silent fallback."""

    class _Lifter(_Projector):
        def extrinsic_tip_cam(self):
            return np.eye(4)

    cfg = VlaPlannerConfig(depth_source="stereo")
    with pytest.raises(ValueError, match="no wall-plane source"):
        VlaPlannerHandler._depth_donor(cfg, _Lifter())


# ── The measured wall, for the stereo depth donor ────────────────────────────

from g1_common.plane import Plane  # noqa: E402
from g1_control.vla.depth_donor import StereoDepthDonor, WallPlaneSlot  # noqa: E402


def _facing_wall(z_m: float) -> Plane:
    return Plane(
        point=np.array([0.0, 0.0, z_m]),
        normal=np.array([0.0, 0.0, -1.0]),
        inliers=5000,
        sampled=19200,
        rms_m=0.002,
    )


def _wall_donor(plane, max_shift_m: float = 0.06):
    slot = WallPlaneSlot()
    if plane is not None:
        slot.publish(plane, frame_id="pelvis")
    return StereoDepthDonor(
        lambda: np.eye(4), slot.latest, max_age_s=2.0, max_shift_m=max_shift_m
    )


# ── ACCUMULATING waits for the tip to arrive ─────────────────────────────────
#
# Sampling must not start while the arm is still converging on the frozen command.

from g1_control.states.vla_planner import AccumulatingPhase  # noqa: E402
from g1_control.utils.target_correction import TargetCorrection  # noqa: E402
from g1_control.vla.planner import TargetPoseAccumulator  # noqa: E402


class _MovingModel:
    """FK whose tip converges on the commanded pose.

    ``errors_mm`` advances once per (measured, commanded) FK pair; the last value
    repeats.
    """

    def __init__(self, errors_mm):
        self.errors_mm = list(errors_mm)
        self._pending = None

    def cache_forward_kinematics(self, q, auto_clip=False):
        # Commanded q carries the sentinel set by _SettleCtx.
        self._pending = 0.0 if q[0] == 99.0 else self.errors_mm[0] * 1e-3
        if q[0] == 99.0 and len(self.errors_mm) > 1:
            self.errors_mm.pop(0)

    def frame_placement(self, _frame):
        return type("P", (), {"translation": np.array([self._pending, 0.0, 0.0])})()


class _SettleCtx(_Ctx):
    def __init__(self, model):
        super().__init__([0.0, 0.0])
        self.robot_model = model
        self.full_q = np.array([0.0, 0.0])
        self.upper_body_indices = [0, 1]


class _NoDetections:
    def get_tip_target_anchored(self):
        return None

    def get_last_detection(self):
        return None


class _Detecting:
    """A source with a fresh estimate every tick."""

    def __init__(self):
        self.seq = 0

    def get_tip_target_anchored(self):
        self.seq += 1
        pose = np.concatenate([[0.1, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
        anchor = np.concatenate([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
        return (pose, anchor, 0.0, self.seq)

    def get_last_detection(self):
        return None


def _accumulating(model, clock, config=None, client=None, donor=None):
    ctx = _SettleCtx(model)
    commander = _Commander()
    commander.command_upper_body([99.0, 0.0])  # sentinel: the commanded pose
    return AccumulatingPhase(
        ctx,
        type(
            "L",
            (),
            {
                "info": lambda *a, **k: None,
                "warn": lambda *a, **k: None,
                "debug": lambda *a, **k: None,
            },
        )(),
        client or _NoDetections(),
        _Actions([99.0, 0.0]),
        TargetPoseAccumulator(),
        donor,
        commander,
        config or VlaPlannerConfig(),
        clock,
    )


def test_accumulating_does_not_sample_until_the_tip_has_arrived():
    """No detection enters the batch until the tip is at the commanded pose."""
    cfg = VlaPlannerConfig()
    t = [0.0]
    # 20 mm out, then 10, then inside the 3 mm tolerance.
    phase = _accumulating(
        _MovingModel([20.0, 10.0, 1.0]), lambda: t[0], cfg, client=_Detecting()
    )
    phase.on_enter()
    # Below the settle floor: the tip is not even measured, let alone sampled.
    assert phase.update() is None
    assert not phase._settled and phase._accumulator.count == 0
    # Past the floor, the tolerance starts being consulted: 20 mm, then 10 mm.
    for tick, _ in enumerate([20.0, 10.0]):
        t[0] = cfg.accumulate_settle_min_s + 0.1 * tick
        assert phase.update() is None
        assert not phase._settled and phase._accumulator.count == 0
    # ...then 1 mm, inside the 3 mm tolerance.
    t[0] = cfg.accumulate_settle_min_s + 0.2
    phase.update()
    assert phase._settled, "should start sampling once the tip is within tolerance"
    assert phase._accumulator.count == 1, "and the first sample lands that tick"


def test_a_tip_that_looks_settled_immediately_still_waits_out_the_floor():
    """Being within tolerance on the entry tick means the controller has not reacted
    yet."""
    cfg = VlaPlannerConfig()
    t = [0.0]
    phase = _accumulating(_MovingModel([1.4]), lambda: t[0], cfg, client=_Detecting())
    phase.on_enter()
    phase.update()
    assert not phase._settled, "within tolerance on the entry tick means nothing"
    assert phase._accumulator.count == 0
    t[0] = cfg.accumulate_settle_min_s
    phase.update()
    assert phase._settled


def test_the_sampling_budget_starts_when_the_tip_settles_not_on_entry():
    """A slow settle must not consume the accumulate budget."""
    cfg = VlaPlannerConfig()
    t = [0.0]
    phase = _accumulating(_MovingModel([1.0]), lambda: t[0], cfg)
    phase.on_enter()
    phase.update()
    t[0] = 4.0  # past the floor, still under the settle timeout
    phase.update()
    assert phase._settled
    # The accumulate timeout must be measured from here, not from on_enter.
    assert phase._started == pytest.approx(4.0)


def test_a_tip_that_never_arrives_accumulates_anyway_after_the_timeout():
    """Detections are anchored per snapshot, so a standing offset must not stall the
    run."""
    cfg = VlaPlannerConfig()
    t = [0.0]
    phase = _accumulating(_MovingModel([20.0]), lambda: t[0], cfg)
    phase.on_enter()
    phase.update()
    assert not phase._settled
    t[0] = cfg.accumulate_settle_timeout_s
    phase.update()
    assert phase._settled


def test_a_ready_batch_cannot_start_motion_outside_the_tracking_tolerance():
    """Sampling may time out into an anchored batch; phase motion may not."""
    cfg = VlaPlannerConfig(accumulate_min_samples=1)
    t = [0.0]
    # Settle at 1 mm, drift to 5 mm when the batch is ready, then reconverge.
    phase = _accumulating(
        _MovingModel([1.0, 5.0, 1.0, 1.0]), lambda: t[0], cfg, client=_Detecting()
    )
    phase.on_enter()
    t[0] = cfg.accumulate_settle_min_s
    assert phase.update() is None

    t[0] += 0.1
    assert phase.update() is None  # inside once: start the continuous hold
    t[0] += cfg.accumulate_settle_hold_s + 0.1
    requested = phase.update()
    assert requested is not None and requested.name == "FINALIZING"


def test_a_model_that_cannot_do_fk_does_not_stall_the_phase():
    """A wait with no measurable end is a stall, not a safeguard."""

    class _NoFk:
        pass

    cfg = VlaPlannerConfig()
    t = [0.0]
    phase = _accumulating(_NoFk(), lambda: t[0], cfg)
    phase.on_enter()
    phase.update()
    t[0] = cfg.accumulate_settle_min_s
    phase.update()
    assert phase._settled


# ── ACCUMULATING at the measured wall's range ────────────────────────────────
#
# Donor geometry is covered in test_vla_planner.py; these check the wiring.


class _DetectingAt:
    """A fine source whose target sits at a fixed tip-frame position."""

    def __init__(self, position):
        self._position = np.asarray(position, dtype=float)
        self.seq = 0

    def get_tip_target_anchored(self):
        self.seq += 1
        pose = np.concatenate([self._position, [1, 0, 0], [0, 1, 0]])
        anchor = np.concatenate([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
        return (pose, anchor, 0.0, self.seq)

    def get_last_detection(self):
        return None


def _settled_donor_phase(client, donor, t):
    """An accumulating phase past its settle wait, sampling on every tick."""
    cfg = VlaPlannerConfig()
    phase = _accumulating(
        _MovingModel([1.0]),
        lambda: t[0],
        cfg,
        client=client,
        donor=donor,
    )
    phase.on_enter()
    phase.update()
    t[0] = cfg.accumulate_settle_min_s
    return phase


def test_accumulating_samples_at_the_donated_range():
    """The batch lands on the measured wall along the detector's ray."""
    t = [0.0]
    # Detector: 100 mm out, off-axis. Measured wall: 140 mm.
    fine = _DetectingAt([0.02, 0.01, 0.10])
    donor = _wall_donor(_facing_wall(0.14))
    phase = _settled_donor_phase(fine, donor, t)
    phase.update()
    assert phase._accumulator.count == 1
    np.testing.assert_allclose(
        phase._accumulator.position(), [0.028, 0.014, 0.14], atol=1e-12
    )


def test_accumulating_takes_nothing_when_the_range_cannot_be_donated():
    """No silent fallback to the detector's range."""
    t = [0.0]
    donor = _wall_donor(None)  # no plane published: nothing can be built
    phase = _settled_donor_phase(_DetectingAt([0.02, 0.01, 0.10]), donor, t)
    phase.update()
    assert phase._accumulator.count == 0
    assert donor.skipped == 1
    # The diagnosis names the donor, not the detector.
    diagnosis = phase._no_detection_diagnosis()
    assert "measured wall's range" in diagnosis
    assert "wall_plane node" in diagnosis


def test_accumulating_without_a_donor_takes_the_estimate_as_published():
    """Without a donor the estimate is used as published."""
    t = [0.0]
    phase = _settled_donor_phase(_DetectingAt([0.02, 0.01, 0.10]), None, t)
    phase.update()
    np.testing.assert_allclose(
        phase._accumulator.position(), [0.02, 0.01, 0.10], atol=1e-12
    )


def test_a_new_round_does_not_inherit_the_previous_batchs_range():
    """Re-entering the phase resets the donor's counters and freshness."""
    t = [0.0]
    fine = _DetectingAt([0.02, 0.01, 0.10])
    donor = _wall_donor(_facing_wall(0.14))
    phase = _settled_donor_phase(fine, donor, t)
    phase.update()
    assert donor.fused == 1
    phase.on_enter()
    assert donor.fused == 0 and donor.skipped == 0


# ── FINALIZING ignores predictions that cannot be of the planned target ──────


class _Recording:
    """Stands in for the live estimator, recording what the gate let through."""

    def __init__(self):
        self.seen = []

    def update(self, anchored, _now):
        self.seen.append(anchored)

    def age(self, _now):
        return 0.0

    def pose_9d(self):
        return None

    def reset(self):
        self.seen.clear()


class _OneEstimate:
    def __init__(self, position):
        self.position = np.asarray(position, float)

    def get_tip_target_anchored(self):
        pose = np.concatenate([self.position, [1, 0, 0], [0, 1, 0]])
        anchor = np.concatenate([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
        return (pose, anchor, 0.0, 1)


def _quiet_logger():
    return type("L", (), {"info": lambda *a, **k: None})()


def _tracker_with(planned, config=None, donor=None, label=""):
    """A LiveTargetTracker with its estimator swapped for a recorder, anchored by
    hand."""
    from g1_control.vla.planner import LiveTargetTracker

    config = config or VlaPlannerConfig()
    tracker = LiveTargetTracker(
        correction=TargetCorrection(
            max_rate_m_s=config.finalize_correction_rate_mm_s * 1e-3,
            max_pos_m=config.finalize_correction_max_mm * 1e-3,
            max_rate_rad_s=np.deg2rad(config.finalize_correction_rate_deg_s),
            max_ang_rad=np.deg2rad(config.finalize_correction_max_deg),
        ),
        donor=donor,
        ema_alpha=config.finalize_correction_ema_alpha,
        gate_m=config.finalize_correction_gate_mm * 1e-3,
        max_age_s=config.finalize_correction_max_age_s,
        logger=_quiet_logger(),
        label=label,
    )
    tracker._estimator = _Recording()
    tracker._planned_target = np.asarray(planned, float)
    return tracker


def _finalizing_with(
    live_position,
    planned=(0.5, 0.0, 0.1),
    config=None,
    donor=None,
    shadow=False,
):
    """A FinalizingPhase wired to recording tracker(s); only ``_update_correction`` is
    exercised."""
    from g1_control.states.vla_planner import FinalizingPhase

    config = config or VlaPlannerConfig()
    phase = FinalizingPhase.__new__(FinalizingPhase)
    phase._cfg = config
    phase._logger = _quiet_logger()
    phase._fine_client = _OneEstimate(live_position)
    phase._donor = donor
    phase._clock = lambda: 0.0
    phase._tracker = _tracker_with(planned, config, donor)
    phase._shadow = (
        _tracker_with(planned, config, None, label="fine-only") if shadow else None
    )
    return phase


def _seen(tracker):
    """What the gate let through to the estimator."""
    return tracker._estimator.seen


def test_a_prediction_near_the_plan_reaches_the_correction():
    phase = _finalizing_with(live_position=(0.51, 0.0, 0.1))  # 10 mm out
    phase._update_correction(0.0)
    assert len(_seen(phase._tracker)) == 1
    admitted = _seen(phase._tracker)[0]
    assert admitted is not None
    np.testing.assert_allclose(admitted[0][:3], [0.51, 0.0, 0.1])
    assert phase._tracker.gated_out == 0


def test_a_far_mode_prediction_never_reaches_the_correction():
    """Target out of frame: the head's far-mode prediction is gated out."""
    phase = _finalizing_with(live_position=(0.80, 0.0, 0.1))  # 300 mm out
    phase._update_correction(0.0)
    assert _seen(phase._tracker) == [None], "garbage must not enter the EMA"
    assert phase._tracker.gated_out == 1


def test_the_gate_is_measured_from_the_plan_not_the_running_estimate():
    """A run of bad readings must not drag the gate along."""
    phase = _finalizing_with(live_position=(0.80, 0.0, 0.1))
    for tick in range(5):
        phase._update_correction(float(tick))
    assert _seen(phase._tracker) == [None] * 5
    assert phase._tracker.gated_out == 5
    np.testing.assert_allclose(phase._tracker._planned_target, [0.5, 0.0, 0.1])


def test_a_zero_gate_disables_the_filter():
    cfg = VlaPlannerConfig()
    cfg.finalize_correction_gate_mm = 0.0
    phase = _finalizing_with(live_position=(0.80, 0.0, 0.1), config=cfg)
    phase._update_correction(0.0)
    assert _seen(phase._tracker) != [None] and phase._tracker.gated_out == 0


def test_the_correction_tracks_at_the_range_the_plan_was_built_at():
    """The correction uses the same donated range as the plan."""
    phase = _finalizing_with(
        live_position=(0.0, 0.0, 0.10),  # detector: 100 mm down the axis
        planned=(0.0, 0.0, 0.14),
        donor=_wall_donor(_facing_wall(0.14)),  # measured wall: 140 mm
    )
    phase._update_correction(0.0)
    admitted = _seen(phase._tracker)[0]
    assert admitted is not None, "the reading agrees with the plan — 0 mm out"
    np.testing.assert_allclose(admitted[0][:3], [0.0, 0.0, 0.14], atol=1e-12)
    # Without the donor the same reading is taken 40 mm short.
    bare = _finalizing_with(live_position=(0.0, 0.0, 0.10), planned=(0.0, 0.0, 0.14))
    bare._update_correction(0.0)
    np.testing.assert_allclose(_seen(bare._tracker)[0][0][:3], [0.0, 0.0, 0.10])


def test_the_correction_holds_when_the_range_cannot_be_donated():
    """With no range to donate, the correction keeps its last value."""
    donor = _wall_donor(None)
    phase = _finalizing_with(
        live_position=(0.0, 0.0, 0.10),
        planned=(0.0, 0.0, 0.14),
        donor=donor,
    )
    phase._update_correction(0.0)
    assert _seen(phase._tracker) == [None]
    assert donor.skipped == 1


# ── The live target, and its fine-only shadow ────────────────────────────────
#
# Live target: plan-time estimate plus the limited correction. Shadow: the same
# tracker on the detector's own depth, for comparison only.


def _live_tracker(planned, donor=None, config=None):
    """A tracker anchored through ``reset``, i.e. with a real estimator."""
    from g1_control.vla.planner import LiveTargetTracker

    config = config or VlaPlannerConfig()
    tracker = LiveTargetTracker(
        correction=TargetCorrection(
            max_rate_m_s=1e9,  # limits are glitch protection, not the subject
            max_pos_m=1.0,
            max_rate_rad_s=1e9,
            max_ang_rad=np.pi,
        ),
        donor=donor,
        ema_alpha=1.0,  # no smoothing, so one prediction moves it fully
        gate_m=config.finalize_correction_gate_mm * 1e-3,
        max_age_s=config.finalize_correction_max_age_s,
        logger=_quiet_logger(),
    )
    tracker.reset(_pose_9d(planned), last_seq=0)
    return tracker


def test_the_live_target_is_unset_until_a_plan_anchors_it():
    """No plan, no live target."""
    from g1_control.vla.planner import LiveTargetTracker

    bare = LiveTargetTracker(
        correction=TargetCorrection(
            max_rate_m_s=1.0, max_pos_m=1.0, max_rate_rad_s=1.0, max_ang_rad=1.0
        ),
        donor=None,
        ema_alpha=1.0,
        gate_m=0.05,
        max_age_s=3.0,
    )
    assert bare.corrected_target is None
    assert bare.corrected_target_9d is None


def test_the_live_target_starts_on_the_plan_and_then_tracks():
    """Before any fresh prediction it sits exactly on the planned estimate."""
    tracker = _live_tracker(planned=(0.5, 0.0, 0.1))
    np.testing.assert_allclose(
        tracker.corrected_target.translation, [0.5, 0.0, 0.1], atol=1e-12
    )

    # 10 mm along x, inside the gate, no smoothing. Two ticks: the first after
    # an anchor has no dt, so TargetCorrection leaves the offset alone.
    fine = _OneEstimate([0.51, 0.0, 0.1]).get_tip_target_anchored()
    tracker.update(1.0, fine)
    tracker.update(2.0, fine)
    np.testing.assert_allclose(
        tracker.corrected_target.translation, [0.51, 0.0, 0.1], atol=1e-9
    )
    np.testing.assert_allclose(
        tracker.corrected_target_9d[:3], [0.51, 0.0, 0.1], atol=1e-9
    )


def test_the_live_target_holds_when_every_prediction_is_gated_out():
    """It holds its last good value."""
    tracker = _live_tracker(planned=(0.5, 0.0, 0.1))
    for tick in range(1, 6):
        tracker.update(
            float(tick), _OneEstimate([0.9, 0.0, 0.1]).get_tip_target_anchored()
        )
    np.testing.assert_allclose(
        tracker.corrected_target.translation, [0.5, 0.0, 0.1], atol=1e-12
    )
    assert tracker.gated_out == 5


def test_the_shadow_and_the_steered_target_differ_by_the_donation():
    """Only the depth source differs, so the separation is the donation."""
    donor = _wall_donor(_facing_wall(0.14))
    steered = _live_tracker(planned=(0.0, 0.0, 0.14), donor=donor)
    shadow = _live_tracker(planned=(0.0, 0.0, 0.14), donor=None)

    fine = _OneEstimate([0.0, 0.0, 0.10]).get_tip_target_anchored()
    for now in (1.0, 2.0):  # the first tick after an anchor has no dt
        steered.update(now, fine)
        shadow.update(now, fine)

    # Donated: rebuilt at the wall's 140 mm, goal unchanged. Fine-only: 100 mm,
    # goal moves 40 mm in.
    np.testing.assert_allclose(
        steered.corrected_target.translation, [0.0, 0.0, 0.14], atol=1e-9
    )
    np.testing.assert_allclose(
        shadow.corrected_target.translation, [0.0, 0.0, 0.10], atol=1e-9
    )
    separation = np.linalg.norm(
        steered.corrected_target.translation - shadow.corrected_target.translation
    )
    assert separation == pytest.approx(0.04, abs=1e-9)


def test_the_shadow_is_never_the_one_applied():
    """The shadow is diagnostic only and never steers."""
    phase = _finalizing_with(
        live_position=(0.0, 0.0, 0.10),
        planned=(0.0, 0.0, 0.14),
        donor=_wall_donor(_facing_wall(0.14)),
        shadow=True,
    )
    phase._update_correction(0.0)
    assert phase._shadow is not None
    assert len(_seen(phase._tracker)) == 1 and len(_seen(phase._shadow)) == 1
    import pinocchio as pin_

    identity = pin_.SE3.Identity()
    np.testing.assert_allclose(
        phase._tracker.apply(identity).translation,
        phase._tracker._correction.apply(identity).translation,
    )


def test_both_trackers_see_the_same_prediction_on_the_same_tick():
    """The client is read once per tick and the tuple shared."""
    phase = _finalizing_with(
        live_position=(0.51, 0.0, 0.1),
        planned=(0.5, 0.0, 0.1),
        donor=_wall_donor(_facing_wall(0.1)),
        shadow=True,
    )
    calls = []
    inner = phase._fine_client.get_tip_target_anchored

    def counted():
        calls.append(1)
        return inner()

    phase._fine_client.get_tip_target_anchored = counted
    phase._update_correction(0.0)
    assert len(calls) == 1, "one read per tick, shared by every tracker"


def test_no_shadow_is_built_when_there_is_no_donor_to_bypass():
    """Without a donor the steering tracker is already fine-only."""
    from g1_control.states.vla_planner import FinalizingPhase

    cfg = VlaPlannerConfig()
    cfg.finalize_shadow_fine_only = True
    phase = FinalizingPhase.__new__(FinalizingPhase)
    phase._cfg = cfg
    phase._donor = None
    phase._tracker = _tracker_with((0.5, 0.0, 0.1), cfg, None)
    phase._shadow = (
        _tracker_with((0.5, 0.0, 0.1), cfg, None)
        if phase._donor is not None and cfg.finalize_shadow_fine_only
        else None
    )
    assert phase._shadow is None
    assert phase.shadow_target_9d is None


# ── Refinement is measured, not scheduled ────────────────────────────────────

from g1_definitions.definitions import VlaPlannerPhase as _Phase  # noqa: E402


class _RouteHarness:
    """The handler's ``_route`` with everything else stubbed out."""

    def __init__(self, residual_mm=None, config=None, along_mm=0.0):
        """``residual_mm`` is the lateral target-to-tip miss; ``along_mm`` the axial
        one."""
        self.handler = VlaPlannerHandler.__new__(VlaPlannerHandler)
        self.handler._cfg = config or VlaPlannerConfig()
        self.handler._logger = type(
            "L",
            (),
            {
                "info": lambda *a, **k: None,
                "warn": lambda *a, **k: None,
                "error": lambda *a, **k: None,
            },
        )()
        self.handler._restarts = 0
        self.handler._at_standoff = False
        self.handler._success_time = None
        self.handler._clock = lambda: 0.0
        self.handler._accumulator = type("A", (), {"reset": lambda self: None})()

        class _Motion:
            completed = True

        self.handler._phases = {
            _Phase.APPROACHING: _Motion(),
            _Phase.STANDOFF: _Motion(),
            _Phase.FINALIZING: _Motion(),
            _Phase.VISUAL_SERVOING: _Motion(),
        }
        cfg = self.handler._cfg
        self.handler._residual = lambda: (
            None
            if residual_mm is None
            else FinalizeResidual.of(
                [
                    along_mm * 1e-3 + cfg.finalize_wall_offset_m,
                    residual_mm * 1e-3,
                    0.0,
                ],
                cfg.finalize_wall_offset_m,
            )
        )

    def finalize_lands(self):
        self.handler._phase = _Phase.FINALIZING
        return self.handler._route(_Phase.HOLDING)

    def initial_measurement_converges(self):
        self.handler._phase = _Phase.ACCUMULATING
        return self.handler._route(_Phase.FINALIZING)

    def standoff_lands(self):
        self.handler._phase = _Phase.STANDOFF
        return self.handler._route(_Phase.ACCUMULATING)

    def measurement_converges(self):
        self.handler._phase = _Phase.ACCUMULATING
        return self.handler._route(_Phase.FINALIZING)


def test_the_push_ends_the_run():
    """The push's landing is the result; nothing re-measures or re-inserts."""
    h = _RouteHarness(residual_mm=1.0)
    assert h.finalize_lands() is _Phase.HOLDING
    assert h.handler._success_time is not None
    assert not h.handler._at_standoff


def test_initial_estimate_stops_and_reaccumulates_at_standoff_before_push():
    h = _RouteHarness(residual_mm=1.0)
    assert h.initial_measurement_converges() is _Phase.STANDOFF
    assert h.standoff_lands() is _Phase.ACCUMULATING
    assert h.handler._at_standoff
    assert h.measurement_converges() is _Phase.FINALIZING


def test_enabled_visual_servo_runs_after_standoff_measurement():
    cfg = VlaPlannerConfig(visual_servo_enabled=True)
    h = _RouteHarness(residual_mm=1.0, config=cfg)
    assert h.initial_measurement_converges() is _Phase.STANDOFF
    assert h.standoff_lands() is _Phase.ACCUMULATING
    assert h.measurement_converges() is _Phase.VISUAL_SERVOING
    h.handler._phase = _Phase.VISUAL_SERVOING
    assert h.handler._route(_Phase.FINALIZING) is _Phase.FINALIZING


# ── A nonsense measurement aborts the run instead of correcting to it ────────


def test_a_batch_past_the_handover_threshold_aborts_rather_than_pushing():
    """A 320 mm miss at the knot is the head's far mode, not the target."""
    cfg = VlaPlannerConfig()
    h = _RouteHarness(residual_mm=319.8, config=cfg)
    assert 0.3198 > cfg.approach_distance
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.RECOVERING
    assert h.handler._restarts == 1
    assert h.handler._success_time is None


def test_an_abort_resets_the_attempt_rather_than_continuing_it():
    h = _RouteHarness(residual_mm=319.8)
    h.standoff_lands()
    assert h.handler._at_standoff
    h.measurement_converges()
    assert not h.handler._at_standoff, "the retry is a fresh attempt"


def test_restarts_are_bounded():
    cfg = VlaPlannerConfig()
    cfg.abort_max_restarts = 1
    h = _RouteHarness(residual_mm=319.8, config=cfg)
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.RECOVERING
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.HOLDING, "must not cycle forever"


def test_zero_restarts_gives_up_without_pushing_at_the_bad_estimate():
    cfg = VlaPlannerConfig()
    cfg.abort_max_restarts = 0
    h = _RouteHarness(residual_mm=319.8, config=cfg)
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.HOLDING


def test_a_large_but_believable_miss_still_gets_its_push():
    """A miss just under the handover distance is still pushed."""
    cfg = VlaPlannerConfig()
    h = _RouteHarness(residual_mm=cfg.approach_distance * 1e3 - 1.0, config=cfg)
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.FINALIZING
    assert h.handler._restarts == 0


def test_an_approach_that_times_out_restarts_from_the_default_pose():
    """A stalled approach is usually in a pose that cannot see the target."""
    h = _RouteHarness(residual_mm=1.0)
    h.handler._phase = _Phase.APPROACHING
    assert h.handler._route(_Phase.RECOVERING) is _Phase.RECOVERING
    assert h.handler._restarts == 1


def test_both_failures_share_one_restart_budget():
    cfg = VlaPlannerConfig()
    cfg.abort_max_restarts = 1
    h = _RouteHarness(residual_mm=319.8, config=cfg)
    # Timed-out approach spends the budget; the far-mode batch has none left.
    h.handler._phase = _Phase.APPROACHING
    assert h.handler._route(_Phase.RECOVERING) is _Phase.RECOVERING
    h.standoff_lands()
    assert h.measurement_converges() is _Phase.HOLDING


# ── The measurement itself: pelvis-frame batch -> miss about the drill axis ───


class _TipPlacement:
    def __init__(self, rotation, translation):
        self.rotation = np.asarray(rotation, float)
        self.translation = np.asarray(translation, float)


class _PlacedModel:
    """FK that reports one fixed tip placement, rotation included."""

    def __init__(self, placement):
        self._placement = placement

    def cache_forward_kinematics(self, _q, auto_clip=False):
        pass

    def frame_placement(self, _frame):
        return self._placement


class _RecordingProjector:
    """Records the tip-frame points it is asked to project, and declines."""

    def __init__(self):
        self.points = []

    def project_tip_point(self, point):
        self.points.append(np.asarray(point, float).copy())
        return None


def _residual_handler(placement, position):
    handler = VlaPlannerHandler.__new__(VlaPlannerHandler)
    handler._cfg = VlaPlannerConfig()
    handler._accumulator = type("A", (), {"position": lambda self: position})()
    handler._ctx = type(
        "C", (), {"robot_model": _PlacedModel(placement), "full_q": np.zeros(3)}
    )()
    handler._tip_projector = _RecordingProjector()
    return handler


def test_the_miss_is_split_in_the_tip_frame_not_the_pelvis_frame():
    """The along/lateral split is taken in the tip frame.

    Bit along pelvis -y, target 25 mm ahead of the tip: zero lateral miss.
    """
    rotation = np.column_stack(([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    handler = _residual_handler(
        _TipPlacement(rotation, [0.4, 0.1, 0.9]), np.array([0.4, 0.075, 0.9])
    )
    residual = handler._residual()
    assert abs(residual.lateral_m) < 1e-12
    cfg = VlaPlannerConfig()
    assert abs(residual.along_m - (0.025 - cfg.finalize_wall_offset_m)) < 1e-12
    assert abs(residual.distance_m - 0.025) < 1e-12


def test_a_miss_across_the_bit_is_lateral_wherever_the_bit_points():
    rotation = np.column_stack(([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    handler = _residual_handler(
        _TipPlacement(rotation, [0.4, 0.1, 0.9]), np.array([0.404, 0.095, 0.9])
    )
    residual = handler._residual()
    assert abs(residual.lateral_m - 0.004) < 1e-12
    cfg = VlaPlannerConfig()
    assert abs(residual.along_m - (0.005 - cfg.finalize_wall_offset_m)) < 1e-12


def test_no_batch_means_no_decision_rather_than_a_zero_residual():
    handler = _residual_handler(_TipPlacement(np.eye(3), np.zeros(3)), None)
    assert handler._residual() is None


# ── The split maneuver ───────────────────────────────────────────────────────
#
# Finalize is two one-segment legs (tip -> standoff, standoff -> aim) with a
# stop and re-accumulation between them; ``stop_at_standoff`` selects the leg.


class _TipModel:
    """FK that reports whatever pose the test placed the tip at."""

    def __init__(self, translation):
        self.translation = np.asarray(translation, float)

    def cache_forward_kinematics(self, q, auto_clip=False):
        pass

    def frame_placement(self, _frame):
        return pin.SE3(np.eye(3), self.translation.copy())


class _Batch:
    """A converged accumulation, reduced to what planning reads off it."""

    def __init__(self, estimate_9d):
        self._estimate = np.asarray(estimate_9d, float)
        self.last_seq = 7
        self.count = 30

    def pose_9d(self):
        return self._estimate.copy()

    def position_spread(self):
        return 0.001

    def rotation_spread_deg(self):
        return 0.1


class _AnchorRecorder:
    """Stands in for a live tracker, recording the anchor each leg was planned on."""

    donor = None

    def __init__(self):
        self.anchors = []
        self.updates = []

    def reset(self, estimate_9d, seq):
        self.anchors.append((np.asarray(estimate_9d, float).copy(), seq))

    def update(self, now, fine):
        self.updates.append((now, fine))

    def apply(self, pose):
        return pose


def _planned_leg(
    *,
    stop_at_standoff,
    tip,
    target=(0.5, 0.0, 0.1),
    standoff_m=0.05,
    rise_m=0.0,
    transit_s=6.0,
    insert_s=12.0,
    config=None,
    truth=None,
    lines=None,
):
    """A ``FinalizingPhase`` assembled just far enough to run ``on_enter``.

    Identity estimate rotation: wall normal +x, up +z. The standoff is
    ``standoff_m`` short in x and ``rise_m`` up in z; the aim is
    ``finalize_wall_offset_m`` short.
    """
    from g1_control.states.vla_planner import FinalizingPhase

    phase = FinalizingPhase.__new__(FinalizingPhase)
    phase._cfg = config or VlaPlannerConfig()
    phase._logger = (
        _quiet_logger()
        if lines is None
        else type("L", (), {"info": lambda _self, line: lines.append(line)})()
    )
    phase._clock = lambda: 0.0
    phase._stop_at_standoff = stop_at_standoff
    phase._standoff_m = standoff_m
    phase._rise_m = rise_m
    phase._transit_s = transit_s
    phase._insert_s = insert_s
    phase._accumulator = _Batch(_pose_9d(target))
    phase._tracker = _AnchorRecorder()
    phase._shadow = None
    phase._fine_client = _OneEstimate(target)
    phase._donor = None
    ctx = _Ctx([0.0, 0.0])
    ctx.robot_model = _TipModel(tip)
    if truth is not None:
        ctx.tf_buffer = _TargetTf(truth)
    phase._ctx = ctx
    phase._commander = _Commander()
    phase._publish_viz = lambda knots: None  # RViz only; no ROS in this test
    phase.on_enter()
    return phase


def _knot_x(target_x, back_off_m, config=None):
    """Where a knot lands on the +x normal: ``back_off_m`` short of the target."""
    return target_x - back_off_m


def test_the_transit_stops_at_the_standoff_instead_of_flying_through_it():
    """The leg ends at the knot rather than passing through it at speed."""
    phase = _planned_leg(stop_at_standoff=True, tip=(0.2, 0.0, 0.1))
    assert len(phase._trajectory.knots) == 2, "one leg, so one segment"
    standoff_x = _knot_x(0.5, 0.05)
    np.testing.assert_allclose(phase._trajectory.knots[0], [0.2, 0.0, 0.1])
    np.testing.assert_allclose(phase._trajectory.knots[-1], [standoff_x, 0.0, 0.1])
    np.testing.assert_allclose(
        phase._trajectory.pose_at(1.0).translation, [standoff_x, 0.0, 0.1], atol=1e-9
    )
    assert phase._trajectory.duration == pytest.approx(6.0)
    assert phase._trajectory.knots[-1][0] < _knot_x(
        0.5, phase._cfg.finalize_wall_offset_m
    ), "the transit stops short of the wall; only the push goes in"


def test_the_push_starts_where_the_arm_actually_stopped_not_at_the_planned_knot():
    """The push's first knot is the measured held pose, not the planned knot."""
    stopped_at = (0.4507, 0.0012, 0.0994)
    phase = _planned_leg(stop_at_standoff=False, tip=stopped_at)
    aim_x = _knot_x(0.5, phase._cfg.finalize_wall_offset_m)
    assert len(phase._trajectory.knots) == 2
    np.testing.assert_allclose(phase._trajectory.knots[0], stopped_at)
    np.testing.assert_allclose(phase._trajectory.knots[-1], [aim_x, 0.0, 0.1])
    assert phase._trajectory.duration == pytest.approx(12.0)


def test_a_raised_standoff_tilts_the_push_but_not_the_bit():
    """The wall normal comes from the estimate, not the path, so an oblique push keeps
    the bit square."""
    phase = _planned_leg(
        stop_at_standoff=False, tip=(0.45, 0.0, 0.105), standoff_m=0.05, rise_m=0.005
    )
    start, end = phase._trajectory.knots[0], phase._trajectory.knots[-1]
    travel = (end - start) / np.linalg.norm(end - start)

    np.testing.assert_allclose(phase._wall_normal, [1.0, 0.0, 0.0], atol=1e-12)
    assert travel[2] < 0.0, "the tip comes down onto the target"
    assert np.dot(travel, phase._wall_normal) < 0.999, "obliquely"
    # Fully aligned: bit along the normal, not the travel direction.
    np.testing.assert_allclose(
        phase._aligned_rotation(phase._align_duration_s)[:, 0],
        [1.0, 0.0, 0.0],
        atol=1e-9,
    )


def test_the_rise_costs_no_range():
    """The knot is lifted in the wall plane, so push length stays the configured
    standoff."""
    for rise_m in (0.0, 0.005, 0.02):
        phase = _planned_leg(
            stop_at_standoff=True, tip=(0.2, 0.0, 0.1), standoff_m=0.05, rise_m=rise_m
        )
        knot = phase._trajectory.knots[-1]
        assert float(0.5 - knot[0]) == pytest.approx(0.05)
        assert float(knot[2] - 0.1) == pytest.approx(rise_m)


def test_the_push_is_a_straight_line():
    """A bowed push would drag the bit across the hole it is entering."""
    phase = _planned_leg(stop_at_standoff=False, tip=(0.45, 0.02, 0.1))
    start, end = phase._trajectory.knots[0], phase._trajectory.knots[-1]
    for u in np.linspace(0.0, 1.0, 21):
        point = phase._trajectory.pose_at(float(u)).translation
        along = np.dot(point - start, end - start) / np.dot(end - start, end - start)
        np.testing.assert_allclose(point, start + along * (end - start), atol=1e-9)


def test_visual_servo_push_preserves_the_servoed_wall_plane_position():
    cfg = VlaPlannerConfig(visual_servo_enabled=True)
    phase = _planned_leg(
        stop_at_standoff=False,
        tip=(0.45, 0.012, 0.094),
        target=(0.5, 0.0, 0.1),
        config=cfg,
    )
    start, end = phase._trajectory.knots
    delta = end - start
    np.testing.assert_allclose(delta[1:], [0.0, 0.0], atol=1e-12)
    assert delta[0] > 0.0


def test_the_push_is_planned_on_the_batch_taken_at_the_standoff():
    """The push and correction are anchored to the standoff batch."""
    phase = _planned_leg(
        stop_at_standoff=False, tip=(0.45, 0.0, 0.1), target=(0.5, 0.003, 0.1)
    )
    anchored, seq = phase._tracker.anchors[-1]
    np.testing.assert_allclose(anchored[:3], [0.5, 0.003, 0.1])
    assert seq == 7
    np.testing.assert_allclose(
        phase._trajectory.knots[-1],
        [_knot_x(0.5, phase._cfg.finalize_wall_offset_m), 0.003, 0.1],
    )


def test_each_leg_finishes_its_alignment_exactly_when_it_lands():
    """The alignment blend reaches 1.0 exactly at the end of each leg."""
    from g1_control.utils.tip_trajectory import aligned_drill_rotation

    for stop_at_standoff, leg_s in ((True, 6.0), (False, 12.0)):
        phase = _planned_leg(stop_at_standoff=stop_at_standoff, tip=(0.2, 0.1, 0.1))
        assert phase._align_duration_s == pytest.approx(leg_s)
        np.testing.assert_allclose(
            phase._aligned_rotation(0.0), phase._rotation0, atol=1e-12
        )
        np.testing.assert_allclose(
            phase._aligned_rotation(leg_s),
            aligned_drill_rotation(phase._wall_normal, phase._rotation0, 1.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            phase._aligned_rotation(leg_s)[:, 0], phase._wall_normal, atol=1e-9
        )


def test_the_transit_reports_no_final_error_because_it_is_not_the_end():
    """Only the push logs final errors."""
    phase = _planned_leg(stop_at_standoff=True, tip=(0.2, 0.0, 0.1))
    logged = []
    phase._log_final_errors = lambda: logged.append(True)
    phase._log_correction = lambda: None
    phase._commander.command_tip_pose = lambda pose: None
    phase._clock = lambda: 6.0
    assert phase.update() is _Phase.ACCUMULATING
    assert phase.completed
    assert not logged
    assert not phase._tracker.updates, "disabled correction must not run at all"


def test_the_planned_standoff_is_logged_against_the_sims_true_one():
    """The planned knot vs. ground truth isolates estimate error from tracking error."""
    lines = []
    _planned_leg(
        stop_at_standoff=True,
        tip=(0.2, 0.0, 0.1),
        target=(0.5, 0.0, 0.1),  # what perception believes
        truth=(0.51, 0.002, 0.097),  # where Isaac actually put the target
        standoff_m=0.05,
        lines=lines,
    )
    planned = [line for line in lines if "standoff planned at" in line]
    assert len(planned) == 1, lines
    # 10 mm short along the normal, 2 mm right of it, 3 mm above it: 10.6 mm.
    assert "10.6 mm out" in planned[0]
    assert "normal -10.0, lateral -2.0, up +3.0 mm" in planned[0]
    assert "red_square_target" in planned[0]


def test_the_push_leg_does_not_log_a_standoff_it_never_flies_to():
    """The push leg does not log a planned standoff."""
    lines = []
    _planned_leg(
        stop_at_standoff=False,
        tip=(0.45, 0.0, 0.1),
        truth=(0.51, 0.0, 0.1),
        lines=lines,
    )
    assert not [line for line in lines if "standoff planned at" in line], lines


def test_the_standoff_arrival_is_logged_against_plan_and_truth():
    """Arrival is logged against both the commanded and the true standoff."""
    lines = []
    phase = _planned_leg(
        stop_at_standoff=True,
        tip=(0.2, 0.0, 0.1),
        target=(0.5, 0.0, 0.1),
        truth=(0.51, 0.0, 0.1),
        standoff_m=0.05,
        lines=lines,
    )
    # The transit has flown; the tip stops 4 mm short of the knot it was sent to.
    phase._ctx.robot_model = _TipModel((0.446, 0.0, 0.1))
    phase._log_correction = lambda: None
    phase._commander.command_tip_pose = lambda pose: None
    phase._clock = lambda: 6.0
    assert phase.update() is _Phase.ACCUMULATING

    arrival = [line for line in lines if "standoff reached at" in line]
    assert len(arrival) == 1, lines
    assert "4.0 mm from the commanded standoff" in arrival[0]
    # True knot 0.46, tip 0.446: 10 mm estimate + 4 mm tracking error.
    assert "14.0 mm from the true standoff" in arrival[0]
    assert "normal -14.0, lateral +0.0, up +0.0 mm" in arrival[0]


def test_no_ground_truth_frame_leaves_the_standoff_logs_on_the_plan_alone():
    """Without a target frame (robot), logging still works on the plan alone."""
    lines = []
    phase = _planned_leg(
        stop_at_standoff=True, tip=(0.2, 0.0, 0.1), lines=lines
    )  # _Ctx's default buffer resolves nothing
    phase._ctx.robot_model = _TipModel((0.45, 0.0, 0.1))
    phase._log_correction = lambda: None
    phase._commander.command_tip_pose = lambda pose: None
    phase._clock = lambda: 6.0
    assert phase.update() is _Phase.ACCUMULATING

    assert not [line for line in lines if "standoff planned at" in line]
    arrival = [line for line in lines if "standoff reached at" in line]
    assert len(arrival) == 1 and "true standoff" not in arrival[0], lines


def test_a_batch_at_the_standoff_that_cannot_be_the_target_is_not_pushed_at():
    """A standoff batch beyond the handover distance aborts instead of pushing."""
    cfg = VlaPlannerConfig()
    h = _RouteHarness(residual_mm=cfg.approach_distance * 1e3 + 10.0, config=cfg)
    assert h.initial_measurement_converges() is _Phase.STANDOFF
    assert h.standoff_lands() is _Phase.ACCUMULATING
    assert h.measurement_converges() is _Phase.RECOVERING
    assert h.handler._restarts == 1
    assert not h.handler._at_standoff, "a restart is a fresh attempt"


def test_a_standoff_batch_with_no_measurable_residual_still_pushes():
    """No measurable residual means no abort."""
    h = _RouteHarness(residual_mm=None)
    assert h.initial_measurement_converges() is _Phase.STANDOFF
    assert h.standoff_lands() is _Phase.ACCUMULATING
    assert h.measurement_converges() is _Phase.FINALIZING


# ── The approach gate ────────────────────────────────────────────────────────


def _vanilla_gate(config=None):
    """The gate built for a driver with no perception head to corroborate."""
    return VlaPlannerHandler._approach_gate(config or VlaPlannerConfig())


def _blind_evidence(fine_m, age_s=0.0, seq=1):
    """What a tick looks like: the detector's estimate and nothing else."""
    return HandoverEvidence(
        fine=_anchored([fine_m, 0.0, 0.0], seq), fine_age_s=age_s, now=0.0
    )


def test_a_blind_driver_hands_over_on_the_detectors_confidence_alone():
    """Visibility is enforced upstream by TipTargetLifter, so a fresh fine estimate
    suffices."""
    gate = _vanilla_gate()
    assert _fire(gate, lambda s: _blind_evidence(0.05, seq=s))


def test_the_blind_gate_refuses_a_stale_estimate():
    """Only the estimate's age distinguishes a stopped detector."""
    gate = _vanilla_gate()
    assert not _fire(gate, lambda s: _blind_evidence(0.05, age_s=99.0, seq=s))
    assert "old" in gate.reason


def test_the_blind_gate_debounces_on_the_detectors_own_sequence():
    cfg = VlaPlannerConfig()
    cfg.approach_consecutive = 2
    gate = _vanilla_gate(cfg)
    assert not gate.update(_blind_evidence(0.05, seq=4))
    assert not gate.update(_blind_evidence(0.05, seq=4))  # the same reading twice
    assert gate.update(_blind_evidence(0.05, seq=5))
