# MIT License
"""Keep the Isaac camera prims on the poses the URDF was calibrated to.

Isaac renders from two prims in :attr:`IsaacSimCfg.robot_asset` while TF comes
from the URDF, so :func:`sync` copies URDF -> USD at sim startup:

    /g1_29dof_rev_1_0/torso_link/d435_link/camera                     (head)
    /g1_29dof_rev_1_0/hand_drill_hilti/hand_cam_link_01/camera_hand   (wrist)

- Written: the mount prims in :data:`CAMERA_MOUNTS`. Their camera children
  (USD camera convention rotation) are preserved.
- Checked only: every other prim named after a URDF link. Drifted articulation
  links block the write; reference frames (``mid360_link``, IMUs) are reported.
- ``drill_tip_link`` has no prim counterpart; a bit swap must be applied by hand
  (see docs/pages/hardware.rst, "Changing the drill bit").
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

try:  # Available under Isaac and anywhere x86_64 with `pip install usd-core`.
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:  # pragma: no cover - the Jetson has no aarch64 usd-core
    Gf = Usd = UsdGeom = UsdPhysics = None

DEFAULT_URDF = Path(
    "/workspace/src/g1_state_publisher/urdf/g1_29_dof_with_hand_drill.urdf.xml"
)

ROBOT_ROOT = "/g1_29dof_rev_1_0"
ROOT_LINK = "pelvis"

# prim -> the URDF frame it is the same frame as. Adding a camera is one row.
CAMERA_MOUNTS = (
    (f"{ROBOT_ROOT}/torso_link/d435_link", "d435_link"),
    (f"{ROBOT_ROOT}/hand_drill_hilti/hand_cam_link_01", "hand_camera_optical_frame"),
)

# Loose enough for float32 prim data and URDF importer residue (~1e-4 rad).
TOL_M = 1e-3
TOL_DEG = 0.05


@dataclass
class FrameDelta:
    """How far one prim is from the URDF frame it mirrors."""

    prim: str
    urdf_frame: str
    translation_m: float
    rotation_deg: float

    @property
    def agrees(self) -> bool:
        return self.translation_m <= TOL_M and self.rotation_deg <= TOL_DEG

    def __str__(self) -> str:
        return (
            f"{self.prim} vs URDF {self.urdf_frame}: "
            f"{self.translation_m * 1000:.3f} mm / {self.rotation_deg:.4f} deg"
        )


@dataclass
class SyncReport:
    """What :func:`sync` found, and what it did about it."""

    lines: list[str] = field(default_factory=list)
    """The full human-readable report, one line each."""

    written: list[FrameDelta] = field(default_factory=list)
    """Camera mounts moved onto the URDF pose (empty on a dry run)."""

    stale: list[FrameDelta] = field(default_factory=list)
    """Camera mounts that disagree and were left alone (dry run, or blocked)."""

    drifted: list[FrameDelta] = field(default_factory=list)
    """Non-camera frames that disagree. Reported only, never written."""

    blocked: str | None = None
    """Why nothing was written, if something stopped it."""

    @property
    def cameras_in_sync(self) -> bool:
        """True when the camera prims match the URDF and nothing blocked the write.

        :attr:`drifted` is report-only and does not count.
        """
        return not (self.written or self.stale or self.blocked)


def sync(
    stage_path: Path | str,
    urdf_path: Path | str = DEFAULT_URDF,
    dry_run: bool = False,
) -> SyncReport:
    """Put the camera mount prims where the URDF says, and report everything else.

    Must run before Isaac references the asset; later edits need a stage reload.
    """
    if Gf is None:
        raise RuntimeError(
            "pxr not importable. Isaac's python.sh alone is NOT enough -- USD is "
            "a kit extension there and is only on the path once an app has "
            "started -- so run standalone through scripts/isaac_usd_python.sh "
            "(what `make sync-cameras` does), or in any environment with "
            "usd-core installed. Called from inside a running sim, where Isaac "
            "has already started, the plain import works."
        )

    report = SyncReport()
    frames = urdf_frames(urdf_path)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"could not open stage {str(stage_path)!r}")

    layer = stage.GetEditTarget().GetLayer()
    report.lines.append(f"urdf  {urdf_path}")
    report.lines.append(f"stage {stage_path}")
    report.lines.append(f"layer {layer.identifier} <- edits land here")

    mounts = {path for path, _ in CAMERA_MOUNTS}
    articulation, reference = compare_shared_frames(stage, frames, skip=mounts)

    worst = max(
        articulation,
        key=lambda d: max(d.translation_m / TOL_M, d.rotation_deg / TOL_DEG),
        default=None,
    )
    if worst is None:
        report.blocked = f"no prim under {ROBOT_ROOT} matches a URDF link"
    elif not worst.agrees:
        report.blocked = f"the stage is not this URDF at zero joint position: {worst}"
    report.lines.append(
        f"articulation: {len(articulation)} shared links, worst {worst}"
        if worst is not None
        else "articulation: nothing shared with the URDF"
    )

    report.drifted = [delta for delta in reference if not delta.agrees]
    for delta in report.drifted:
        report.lines.append(f"DRIFTED (reported, not written): {delta}")

    if report.blocked:
        report.lines.append(f"REFUSING TO WRITE: {report.blocked}")

    pending = []
    for prim_path, frame_name in CAMERA_MOUNTS:
        if frame_name not in frames:
            raise RuntimeError(f"URDF has no frame {frame_name!r}")
        prim = _prim(stage, prim_path)
        parent = prim_frame(stage, prim.GetParent().GetPath())
        # p_parent = p_frame * (frame -> root) * (root -> parent)
        wanted = frames[frame_name] * parent.GetInverse()
        current = UsdGeom.Xformable(prim).GetLocalTransformation(
            Usd.TimeCode.Default()
        )
        delta = FrameDelta(prim_path, frame_name, *pose_delta(wanted, current))

        report.lines.append(f"{prim_path} == URDF {frame_name}")
        report.lines.append(f"  joint: {describe_holding_joint(stage, prim_path)}")
        report.lines.append(f"  now:   {format_pose(current)}")
        report.lines.append(f"  urdf:  {format_pose(wanted)}")
        identical = delta.translation_m <= 1e-9 and delta.rotation_deg <= 1e-6
        report.lines.append(
            f"  delta: {delta.translation_m * 1000:.3f} mm / "
            f"{delta.rotation_deg:.4f} deg" + ("  (in sync)" if identical else "")
        )
        if identical:
            continue
        pending.append((prim, wanted, delta))

    if not pending:
        report.lines.append("cameras already match the URDF, nothing written")
        return report
    if dry_run or report.blocked:
        report.stale = [delta for _, _, delta in pending]
        report.lines.append(f"{len(pending)} camera prim(s) left unchanged")
        return report

    for prim, wanted, _ in pending:
        author_pose(prim, wanted)
    layer.Save()

    # Read the composed value back: composition may route the opinion to a weaker layer.
    for prim, wanted, _ in pending:
        composed = UsdGeom.Xformable(prim).GetLocalTransformation(
            Usd.TimeCode.Default()
        )
        translation, rotation = pose_delta(wanted, composed)
        if translation > 1e-6 or rotation > 1e-4:
            raise RuntimeError(
                f"{prim.GetPath()} still composes to {format_pose(composed)} "
                f"({translation * 1000:.3f} mm / {rotation:.4f} deg off) -- a "
                "stronger opinion is winning; find it before trusting the asset"
            )

    report.written = [delta for _, _, delta in pending]
    report.lines.append(
        f"wrote {len(pending)} camera prim(s) to {stage_path}, verified composed"
    )
    return report


# ------------------------------------------------------------------ URDF side


def urdf_frames(urdf_path: Path | str) -> dict:
    """Forward kinematics of every URDF link at zero joint position.

    Returns link name -> Gf.Matrix4d from that link into :data:`ROOT_LINK`.
    """
    root = ET.parse(str(urdf_path)).getroot()

    children: dict[str, list] = {}
    for joint in root.findall("joint"):
        # <origin> is optional in URDF, and identity when absent.
        origin = joint.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        local = rpy_xyz_matrix(
            [float(v) for v in rpy.split()], [float(v) for v in xyz.split()]
        )
        children.setdefault(joint.find("parent").get("link"), []).append(
            (joint.find("child").get("link"), local)
        )

    frames = {ROOT_LINK: Gf.Matrix4d(1.0)}
    stack = [ROOT_LINK]
    while stack:
        parent = stack.pop()
        for child, local in children.get(parent, ()):
            # Row-vector convention throughout, as USD uses: p_parent = p_child * M.
            frames[child] = local * frames[parent]
            stack.append(child)
    return frames


def rpy_xyz_matrix(rpy, xyz):
    """A URDF ``<origin>`` as a Gf.Matrix4d, row-vector convention.

    URDF fixes R = Rz(yaw) Ry(pitch) Rx(roll) applied to column vectors, i.e.
    roll first; transposed for row vectors that is Mx * My * Mz.
    """
    roll, pitch, yaw = rpy
    matrix = Gf.Matrix4d(1.0)
    for axis, angle in (
        (Gf.Vec3d(1, 0, 0), roll),
        (Gf.Vec3d(0, 1, 0), pitch),
        (Gf.Vec3d(0, 0, 1), yaw),
    ):
        matrix = matrix * Gf.Matrix4d(1.0).SetRotate(
            Gf.Rotation(axis, math.degrees(angle))
        )
    matrix.SetTranslateOnly(Gf.Vec3d(*xyz))
    return matrix


# ------------------------------------------------------------------- USD side


def _prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"no prim at {str(path)!r} in the stage")
    return prim


def prim_frame(stage, path):
    """A prim's transform relative to the robot root prim, row-vector convention."""
    time = Usd.TimeCode.Default()
    to_world = UsdGeom.Xformable(_prim(stage, path)).ComputeLocalToWorldTransform(time)
    root = UsdGeom.Xformable(_prim(stage, ROBOT_ROOT))
    return to_world * root.ComputeLocalToWorldTransform(time).GetInverse()


def pose_delta(a, b) -> tuple[float, float]:
    """(translation metres, rotation degrees) between two Gf.Matrix4d poses."""
    translation = (a.ExtractTranslation() - b.ExtractTranslation()).GetLength()
    relative = a.ExtractRotationQuat() * b.ExtractRotationQuat().GetInverse()
    real = max(-1.0, min(1.0, abs(relative.GetReal())))
    return translation, math.degrees(2.0 * math.acos(real))


def compare_shared_frames(stage, frames, skip=()):
    """Every prim named after a URDF link, split into articulation vs reference.

    Articulation links are direct children of the robot root; deeper matches are
    reference frames.
    """
    articulation, reference = [], []
    for prim in Usd.PrimRange(_prim(stage, ROBOT_ROOT)):
        name = prim.GetName()
        path = prim.GetPath().pathString
        if name not in frames or path in skip or not UsdGeom.Xformable(prim):
            continue
        delta = FrameDelta(
            path, name, *pose_delta(prim_frame(stage, path), frames[name])
        )
        top_level = prim.GetParent().GetPath().pathString == ROBOT_ROOT
        (articulation if top_level else reference).append(delta)
    return articulation, reference


def describe_holding_joint(stage, mount_path) -> str:
    """Whether a PhysicsFixedJoint would put the mount's body where it is authored.

    The joint holds ``hand_drill_hilti`` at runtime, so it must agree with the xform.
    """
    body_path = _prim(stage, mount_path).GetParent().GetPath()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1 or body1[0] != body_path:
            continue

        held = prim_frame(stage, body0[0]).GetInverse()
        authored = prim_frame(stage, body_path) * held
        by_joint = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
        by_joint.SetTranslateOnly(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        translation, degrees = pose_delta(authored, by_joint)
        agrees = translation <= TOL_M and degrees <= TOL_DEG
        return (
            f"{prim.GetName()} holds {body_path.name} to {body0[0].name}, "
            f"{translation * 1000:.3f} mm / {degrees:.4f} deg from the authored "
            f"xform [{'ok' if agrees else 'MISMATCH'}]"
        )
    return "rigidly parented, no fixed joint involved"


def format_pose(matrix) -> str:
    t = matrix.ExtractTranslation()
    q = matrix.ExtractRotationQuat()
    i = q.GetImaginary()
    return (
        f"translate ({t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f})  "
        f"orient ({q.GetReal():.10f}, {i[0]:.10f}, {i[1]:.10f}, {i[2]:.10f})"
    )


def author_pose(prim, matrix) -> None:
    """Write translate + orient, leaving the rest of the xformOpOrder alone."""
    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    unexpected = set(ops) - {"xformOp:translate", "xformOp:orient", "xformOp:scale"}
    if unexpected:
        raise RuntimeError(
            f"{prim.GetPath()} carries {sorted(unexpected)}; a stray op can shadow "
            "translate/orient, so resolve it by hand rather than writing under it"
        )
    scale = ops.get("xformOp:scale")
    if scale is not None and not Gf.IsClose(scale.Get(), Gf.Vec3d(1, 1, 1), 1e-9):
        raise RuntimeError(
            f"{prim.GetPath()} is scaled {scale.Get()}; refusing to write"
        )

    translate_op = ops.get("xformOp:translate") or xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(matrix.ExtractTranslation()))

    orient_op = ops.get("xformOp:orient") or xformable.AddOrientOp()
    quat = matrix.ExtractRotationQuat()
    # Match the op's authored precision: mixing Quatf/Quatd raises.
    if isinstance(orient_op.GetAttr().Get(), Gf.Quatf):
        quat = Gf.Quatf(quat)
    orient_op.Set(quat)
