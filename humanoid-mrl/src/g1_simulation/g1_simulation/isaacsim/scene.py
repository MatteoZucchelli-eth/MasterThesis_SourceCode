# MIT License
"""Isaac scene setup: room, tatami floor, wall objects, robot collision tweaks.

The wall carries all nine ``(colour, shape)`` cells (``WALL_OBJECTS``). The
taxonomy lives in :mod:`g1_definitions.wall_objects`; this module renders it.
"""
from __future__ import annotations

import logging
import math

from g1_definitions import wall_objects

from g1_simulation.isaacsim import calibration_tag

# Target square dimensions (UsdGeom.Cube scale = full side length).
TARGET_DEPTH = 0.001  # X: depth into the wall
TARGET_SIDE = 0.01  # Y and Z: face size

# Target frame lies on the plaque's visible (-X) face.
TARGET_X = 3.309 - TARGET_DEPTH / 2.0

# Paper-target / steel-bit contact. Both sides share these values under the
# average combine rule, so the pair coefficient equals them exactly.
CONTACT_STATIC_FRICTION = 1.0
CONTACT_DYNAMIC_FRICTION = 0.8
CONTACT_RESTITUTION = 0.0
PAPER_PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterials/TargetPaper"
STEEL_PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterials/DrillBitSteel"

# Room X offset, independent of the targets.
_ENV_OFFSET_X = 0.0

# Colour source for EVAL_COLOURS; these prims are not built at runtime.
# Red is measured off the real square (median sRGB (115, 17, 42), linearised).
TARGETS: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "red_target": ("/World/TargetBoard/RedTarget", (0.1714, 0.0056, 0.0232), -0.27),
    # Green/blue scaled to red's Rec.709 luminance (Y = 0.0421) so brightness
    # is not a colour cue.
    "green_target": ("/World/TargetBoard/GreenTarget", (0.0075, 0.0559, 0.0075), -0.17),
    "blue_target": ("/World/TargetBoard/BlueTarget", (0.0184, 0.0367, 0.1653), -0.07),
}

# ---- Tatami floor ---------------------------------------------------------
# Matte cover over Simple_Room's specular wooden floor.
TATAMI_MAT_SIZE = (1.82, 0.91)  # standard Japanese mat, long axis along +X
TATAMI_GRID = (3, 5)  # mats along (X, Y) -> 5.46 x 4.55 m, wall to behind robot
TATAMI_CENTRE_XY = (0.58, 0.0)
"""Area centre, offset forward so the mat reaches the target wall."""

TATAMI_SEAM = 0.014  # gap between mats; the base slab shows through as a seam
TATAMI_TOP_Z = 0.0015
"""Mat surface height above the room floor.

Minimal lift that avoids z-fighting; a realistic thickness would shift the
robot's stance relative to the dataset poses."""

TATAMI_SEAM_DEPTH = 0.0007
"""Seam floor (base slab top face) depth below the mat surface.

Must stay under :data:`TATAMI_TOP_Z`, else the seams show the wooden floor."""

TATAMI_COLOUR = (0.05, 0.05, 0.055)
TATAMI_SEAM_COLOUR = (0.025, 0.025, 0.03)
"""Near-black charcoal (~1/8 of the room floor's albedo).

Floor pixels sit in the bottom few codes, so the drill shadow and seam grid are
effectively invisible in recorded frames."""
TATAMI_ROUGHNESS = 0.92  # near-Lambertian: no view-dependent highlight

_LEG_LINK_NAMES = (
    "hip_pitch_link",
    "hip_roll_link",
    "hip_yaw_link",
    "knee_link",
    "ankle_pitch_link",
    "ankle_roll_link",
)


def add_cube(stage, prim_path, *, pos, scale, color, roughness=0.5, collision=True):
    """Create a USD cube prim with a diffuse material and optional box collision."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3d(*scale))

    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    _bind_diffuse(stage, cube.GetPrim(), prim_path + "/Mat", color, roughness)


def _bind_diffuse(stage, prim, mat_path, color, roughness):
    """Give ``prim`` a matte UsdPreviewSurface of ``color``.

    Shared by all shape builders so every object is lit identically.
    """
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _bind_physics_material(stage, prim, mat_path):
    """Bind a non-bouncing contact material to ``prim`` (``physics`` purpose)."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, mat_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(CONTACT_STATIC_FRICTION)
    physics.CreateDynamicFrictionAttr().Set(CONTACT_DYNAMIC_FRICTION)
    physics.CreateRestitutionAttr().Set(CONTACT_RESTITUTION)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set(PhysxSchema.Tokens.average)
    physx.CreateRestitutionCombineModeAttr().Set(PhysxSchema.Tokens.average)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _wall_polygon(stage, prim_path, pos, depth, points_yz, color, roughness):
    """Extrude a flat polygon into a wall plaque, front face toward the robot.

    ``points_yz`` is the outline in the wall plane, CCW as seen from the robot;
    the winding makes the visible face's normal point along -X.
    """
    from pxr import Gf, UsdGeom

    n = len(points_yz)
    front = [Gf.Vec3f(0.0, y, z) for y, z in points_yz]
    back = [Gf.Vec3f(depth, y, z) for y, z in points_yz]

    counts = [n, n] + [4] * n
    indices = list(range(n))[::-1] + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        indices.extend([i, j, n + j, n + i])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(front + back)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    ys = [y for y, _ in points_yz]
    zs = [z for _, z in points_yz]
    mesh.GetExtentAttr().Set(
        [Gf.Vec3f(0.0, min(ys), min(zs)), Gf.Vec3f(depth, max(ys), max(zs))]
    )
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
    _bind_diffuse(stage, mesh.GetPrim(), prim_path + "/Mat", color, roughness)
    return mesh.GetPrim()


def _area_matched_polygon(sides, size, phase):
    """Regular ``sides``-gon whose area equals a square of side ``size``.

    Equal area keeps pixel count from being a shape cue. Exact for the authored
    n-gon (area = (n/2) R^2 sin(2 pi / n)).
    """
    radius = math.sqrt(2.0 * size * size / (sides * math.sin(2.0 * math.pi / sides)))
    return [
        (
            radius * math.cos(phase + 2.0 * math.pi * i / sides),
            radius * math.sin(phase + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


# Shape name -> (size -> (y, z) outline, CCW from the robot). The square's phase
# makes it axis-aligned with side ``size``.
SHAPE_OUTLINES = {
    "square": lambda size: _area_matched_polygon(4, size, math.pi / 4.0),
    "circle": lambda size: _area_matched_polygon(32, size, 0.0),
    "triangle": lambda size: _area_matched_polygon(3, size, math.pi / 2.0),
}

SHAPE_NAMES = wall_objects.SHAPES

# Colour name -> linear diffuse.
EVAL_COLOURS = {
    "red": TARGETS["red_target"][1],
    "green": TARGETS["green_target"][1],
    "blue": TARGETS["blue_target"][1],
}

# Cells withheld from training: a Latin-square diagonal, so every colour and
# shape is still seen in training and only the combination is novel.
EVAL_HELD_OUT = wall_objects.HELD_OUT


DISTRACTOR_ROOT = "/World/EvalShapes"


def shape_prim_path(colour_name: str, shape: str, root_path: str = DISTRACTOR_ROOT):
    """Stable prim path for one grid cell, so the resetter can find it later."""
    return f"{root_path}/{colour_name.capitalize()}{shape.capitalize()}"


WALL_OBJECTS: dict[tuple[str, str], tuple[str, str, tuple[float, float, float]]] = {
    (colour, shape): (
        shape_prim_path(colour, shape),
        wall_objects.object_frame(colour, shape),
        rgb,
    )
    for colour, rgb in EVAL_COLOURS.items()
    for shape in SHAPE_NAMES
}
"""Every ``(colour, shape)`` cell -> ``(prim_path, tf_frame, rgb)``.

The instruction must name both attributes to identify one cell.
"""

TARGETABLE_CELLS = wall_objects.TARGETABLE_CELLS
"""Cells an episode may command: the grid minus :data:`EVAL_HELD_OUT`.

Held-out cells are still built and shown as distractors; they are never the target.
"""


def add_wall_shape(
    stage,
    prim_path: str,
    *,
    colour_rgb: tuple[float, float, float],
    shape: str,
    size: float,
    pos: tuple[float, float, float],
    depth: float = TARGET_DEPTH,
    roughness: float = 0.5,
):
    """Build one wall plaque and return its prim.

    Single builder for all wall shapes so geometry and material stay identical.
    """
    from pxr import UsdPhysics

    prim = _wall_polygon(
        stage,
        prim_path,
        pos=pos,
        depth=depth,
        points_yz=SHAPE_OUTLINES[shape](size),
        color=colour_rgb,
        roughness=roughness,
    )
    UsdPhysics.CollisionAPI.Apply(prim)
    _bind_physics_material(stage, prim, PAPER_PHYSICS_MATERIAL_PATH)
    return prim


def add_wall_objects(stage, *, size: float = TARGET_SIDE, depth: float = TARGET_DEPTH):
    """Build all nine colour x shape wall objects, initially hidden.

    The resetter only moves and shows/hides these prims per episode.
    """
    from pxr import UsdGeom

    for (colour, shape), (prim_path, _, rgb) in WALL_OBJECTS.items():
        prim = add_wall_shape(
            stage,
            prim_path,
            colour_rgb=rgb,
            shape=shape,
            size=size,
            pos=(TARGET_X, 0.0, 0.0),
            depth=depth,
        )
        UsdGeom.Imageable(prim).MakeInvisible()

    logging.info(
        "add_wall_objects: %d cells built (hidden until the first reset); "
        "%d targetable, held out: %s",
        len(WALL_OBJECTS),
        len(TARGETABLE_CELLS),
        ", ".join(f"{c} {s}" for c, s in EVAL_HELD_OUT),
    )


def _flat_quads(stage, prim_path, x, quads, color, roughness=0.9):
    """Build axis-aligned wall quads as one mesh."""
    from pxr import Gf, UsdGeom

    points, counts, indices = [], [], []
    for y0, y1, z0, z1 in quads:
        base = len(points)
        points += [
            Gf.Vec3f(x, y0, z0),
            Gf.Vec3f(x, y0, z1),
            Gf.Vec3f(x, y1, z1),
            Gf.Vec3f(x, y1, z0),
        ]
        counts.append(4)
        indices += [base, base + 1, base + 2, base + 3]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetExtentAttr().Set(
        [
            Gf.Vec3f(x, min(p[1] for p in points), min(p[2] for p in points)),
            Gf.Vec3f(x, max(p[1] for p in points), max(p[2] for p in points)),
        ]
    )
    _bind_diffuse(stage, mesh.GetPrim(), prim_path + "/Mat", color, roughness)
    return mesh.GetPrim()


def add_calibration_tag(stage):
    """Put the head-camera calibration tag on the wall."""
    _flat_quads(
        stage,
        calibration_tag.PRIM_PATH + "/Backing",
        TARGET_X - calibration_tag.BACKING_INSET_M,
        [calibration_tag.backing_quad()],
        (0.92, 0.92, 0.92),
    )
    _flat_quads(
        stage,
        calibration_tag.PRIM_PATH + "/Pattern",
        TARGET_X - calibration_tag.PATTERN_INSET_M,
        calibration_tag.black_cell_quads(),
        (0.02, 0.02, 0.02),
    )
    logging.info(
        "add_calibration_tag: %s id %d, %.0f mm black square at y=%.3f z=%.3f, "
        "published as '%s'.",
        calibration_tag.TAG_FAMILY,
        calibration_tag.TAG_ID,
        calibration_tag.TAG_SIZE_M * 1e3,
        *calibration_tag.TAG_CENTRE_YZ,
        calibration_tag.TAG_FRAME,
    )
    return calibration_tag.PRIM_PATH


def add_tatami_floor(
    stage, root_path: str = "/World/Tatami", *, collision: bool = False
) -> None:
    """Lay an opaque matte-grey tatami over the room floor in front of the robot.

    A seam-coloured base slab plus individual mats inset by half a seam, so the
    seams are the base showing through. ``collision=True`` (walk mode) makes the
    mats a contact surface at ``TATAMI_TOP_Z``; otherwise the ground plane at
    z = -0.02 is the only floor.
    """
    cols, rows = TATAMI_GRID
    mat_x, mat_y = TATAMI_MAT_SIZE
    centre_x, centre_y = TATAMI_CENTRE_XY
    area_x, area_y = cols * mat_x, rows * mat_y

    # Base slab top sits one seam depth below the mats; clamped to stay above
    # the room floor.
    seam_depth = min(TATAMI_SEAM_DEPTH, TATAMI_TOP_Z / 2.0)
    base_thickness = 0.01
    base_top = TATAMI_TOP_Z - seam_depth
    add_cube(
        stage,
        f"{root_path}/Base",
        pos=(centre_x, centre_y, base_top - base_thickness / 2.0),
        scale=(area_x, area_y, base_thickness),
        color=TATAMI_SEAM_COLOUR,
        roughness=TATAMI_ROUGHNESS,
        collision=collision,
    )

    # Mats, inset by half a seam on each side.
    mat_thickness = 0.006
    origin_x = centre_x - area_x / 2.0
    origin_y = centre_y - area_y / 2.0
    for col in range(cols):
        for row in range(rows):
            add_cube(
                stage,
                f"{root_path}/Mat_{col}_{row}",
                pos=(
                    origin_x + (col + 0.5) * mat_x,
                    origin_y + (row + 0.5) * mat_y,
                    TATAMI_TOP_Z - mat_thickness / 2.0,
                ),
                scale=(mat_x - TATAMI_SEAM, mat_y - TATAMI_SEAM, mat_thickness),
                color=TATAMI_COLOUR,
                roughness=TATAMI_ROUGHNESS,
                collision=collision,
            )

    logging.info(
        "add_tatami_floor: %dx%d mats covering %.2fx%.2f m centred at "
        "(%.2f, %.2f), surface at z=%.3f, collision=%s.",
        cols,
        rows,
        area_x,
        area_y,
        centre_x,
        centre_y,
        TATAMI_TOP_Z,
        collision,
    )


def disable_leg_collisions(robot_path: str = "/World/robot") -> None:
    """Disable collision geometry on every leg link of the loaded robot USD.

    With the pelvis pinned, leg-floor contacts only inject jitter into the arm
    joints. Links are payloaded and instanceable, so the payload is loaded and
    the subtree de-instanced before authoring overrides. Warns if nothing is
    disabled.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        logging.warning(f"disable_leg_collisions: {robot_path} not found; skipping.")
        return

    robot_prim.Load(Usd.LoadWithDescendants)

    # De-instance repeatedly: un-instancing exposes children that may be instanceable.
    while True:
        made_editable = False
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                made_editable = True
        if not made_editable:
            break

    matched = disabled = 0
    samples: list[str] = []
    for prim in Usd.PrimRange(robot_prim):
        path = str(prim.GetPath())
        if not any(name in path for name in _LEG_LINK_NAMES):
            continue
        matched += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        disabled += 1
        if len(samples) < 3:
            samples.append(path)

    if disabled:
        logging.info(
            f"disable_leg_collisions: disabled {disabled} leg colliders "
            f"({matched} leg prims matched), e.g. {samples}. Pelvis is pinned, so "
            "the legs never need to touch the floor."
        )
    else:
        logging.warning(
            "disable_leg_collisions: matched %d leg prims but disabled 0 colliders — "
            "the leg collision meshes were not found (still behind a payload, still "
            "instanced, or the link names differ). The feet will still hit the floor. "
            "Log the prim paths under %s and update _LEG_LINK_NAMES to match.",
            matched,
            robot_path,
        )


DRILL_BIT_COLLIDER_CUTOFF_X = -0.010
"""Colliders on the drill ahead of this X (drill body frame) count as "the bit".

Geometry-based rather than name-based, since hull names are regenerated. Bit
colliders span +0.0007..+0.0451; the next collider back ends at -0.0266.
"""

DRILL_TIP_X_IN_BODY = 0.1287 + 0.0444 - 0.2112549984204769 + 0.0833
"""``drill_tip_link`` X in the drill body frame, for logging only.

Mirrors g1_29_dof_with_hand_drill.urdf.xml: chuck face 0.1287 + bit 0.0444,
shifted by the 0.2112550 / 0.0833 offset between the two drill body frames.
"""


DRILL_BIT_HIGH_CONTRAST_COLOR = (0.13, 0.14, 0.16)
"""Dark gunmetal for the bit when the high-contrast override is on."""


def set_drill_bit_high_contrast(robot_path: str = "/World/robot") -> None:
    """Repaint the drill bit matte dark grey so it reads against a pale wall.

    Debug view only: the fully metallic bit mirrors the wall, but the cameras
    record the repaint, so do not collect data with it on.
    """
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        logging.warning(
            "set_drill_bit_high_contrast: %s not found; skipping.", robot_path
        )
        return

    # Load payloads and de-instance so overrides land on real prims.
    robot_prim.Load(Usd.LoadWithDescendants)
    while True:
        made_editable = False
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                made_editable = True
        if not made_editable:
            break

    bit_prim = next(
        (p for p in Usd.PrimRange(robot_prim) if p.GetName() == "drill_bit"), None
    )
    if bit_prim is None:
        logging.warning(
            "set_drill_bit_high_contrast: no drill_bit prim under %s.", robot_path
        )
        return

    repainted = []
    for prim in Usd.PrimRange(bit_prim):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*DRILL_BIT_HIGH_CONTRAST_COLOR)
        )
        repainted.append(str(prim.GetPath()))

    # displayColor is the viewport fallback when the material is ignored.
    for prim in Usd.PrimRange(bit_prim):
        attr = prim.GetAttribute("primvars:displayColor")
        if attr and attr.HasAuthoredValue():
            attr.Set([Gf.Vec3f(*DRILL_BIT_HIGH_CONTRAST_COLOR)])

    if repainted:
        logging.warning(
            "set_drill_bit_high_contrast: repainted %d bit shader(s) %s -- the "
            "cameras see this, so do not collect a dataset with it on.",
            len(repainted),
            DRILL_BIT_HIGH_CONTRAST_COLOR,
        )
    else:
        logging.warning(
            "set_drill_bit_high_contrast: found no UsdPreviewSurface under the "
            "bit; it keeps its authored look."
        )


def set_drill_bit_physics_material(robot_path: str = "/World/robot") -> None:
    """Bind the explicit steel contact material to every drill-bit collider."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        logging.warning(
            "set_drill_bit_physics_material: %s not found; skipping.", robot_path
        )
        return

    # Load payloads and de-instance so bindings land on real prims.
    robot_prim.Load(Usd.LoadWithDescendants)
    while True:
        made_editable = False
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                made_editable = True
        if not made_editable:
            break

    drill_prim = next(
        (p for p in Usd.PrimRange(robot_prim) if p.GetName() == "hand_drill_hilti"),
        None,
    )
    if drill_prim is None:
        logging.warning(
            "set_drill_bit_physics_material: no hand_drill_hilti prim under %s.",
            robot_path,
        )
        return

    cache = UsdGeom.BBoxCache(
        0,
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.guide,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
        ],
        useExtentsHint=False,
    )
    bound = []
    for prim in Usd.PrimRange(drill_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        box = cache.ComputeRelativeBound(prim, drill_prim).ComputeAlignedRange()
        if box.IsEmpty() or box.GetMax()[0] <= DRILL_BIT_COLLIDER_CUTOFF_X:
            continue
        _bind_physics_material(stage, prim, STEEL_PHYSICS_MATERIAL_PATH)
        bound.append(str(prim.GetPath()))

    if bound:
        logging.info(
            "set_drill_bit_physics_material: bound steel material to %d bit "
            "collider(s), static friction=%.2f, dynamic friction=%.2f.",
            len(bound),
            CONTACT_STATIC_FRICTION,
            CONTACT_DYNAMIC_FRICTION,
        )
    else:
        logging.warning(
            "set_drill_bit_physics_material: found no collider ahead of x=%.3f; "
            "the bit is still using its authored/default material.",
            DRILL_BIT_COLLIDER_CUTOFF_X,
        )


def disable_drill_bit_collision(robot_path: str = "/World/robot") -> None:
    """Turn off the drill-bit colliders so the tip can rest on the wall.

    Prevents the position-controlled arm from loading against the wall when the
    commanded ``drill_tip_link`` pose is at the surface. Disables every collider
    forward of ``DRILL_BIT_COLLIDER_CUTOFF_X``; the next collider back is ~72 mm
    behind the tip. The bit then passes through everything and contact cannot
    serve as a surface signal.
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        logging.warning(
            f"disable_drill_bit_collision: {robot_path} not found; skipping."
        )
        return

    robot_prim.Load(Usd.LoadWithDescendants)
    while True:
        made_editable = False
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                made_editable = True
        if not made_editable:
            break

    drill_prim = next(
        (p for p in Usd.PrimRange(robot_prim) if p.GetName() == "hand_drill_hilti"),
        None,
    )
    if drill_prim is None:
        logging.warning(
            "disable_drill_bit_collision: no hand_drill_hilti prim under %s; the "
            "drill bit will still hit the wall.",
            robot_path,
        )
        return

    # Bounds in the drill body frame. All purposes are required: the hulls are
    # purpose="guide" and a default_-only cache returns empty ranges.
    cache = UsdGeom.BBoxCache(
        0,
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.guide,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
        ],
        useExtentsHint=False,
    )
    # Count already-disabled colliders so a repeat call reports as a no-op.
    disabled, already_off, remaining_front = [], 0, None
    for prim in Usd.PrimRange(drill_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        box = cache.ComputeRelativeBound(prim, drill_prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        front = box.GetMax()[0]
        enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        is_off = bool(enabled_attr) and enabled_attr.Get() is False

        if front > DRILL_BIT_COLLIDER_CUTOFF_X:
            if is_off:
                already_off += 1
            else:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled.append((prim.GetName(), front))
        elif not is_off and (remaining_front is None or front > remaining_front):
            remaining_front = front

    if not disabled:
        if already_off:
            logging.info(
                "disable_drill_bit_collision: already applied (%d bit collider(s) "
                "were off); nothing to do.",
                already_off,
            )
        else:
            logging.warning(
                "disable_drill_bit_collision: found no collider at all ahead of "
                "x=%.3f on the drill. The bit will still load against the wall. "
                "Check whether the collision decomposition changed shape.",
                DRILL_BIT_COLLIDER_CUTOFF_X,
            )
        return

    clearance_mm = (
        (DRILL_TIP_X_IN_BODY - remaining_front) * 1e3
        if remaining_front is not None
        else float("inf")
    )
    logging.info(
        "disable_drill_bit_collision: disabled %d bit collider(s) %s. Overshoot "
        "before the drill next touches anything is now %.1f mm (was %.2f mm), "
        "against a 5 mm settle tolerance.",
        len(disabled),
        [f"{name} @ x={front:+.4f}" for name, front in disabled],
        clearance_mm,
        (DRILL_TIP_X_IN_BODY - max(f for _, f in disabled)) * 1e3,
    )


def setup_scene(
    world,
    add_reference_to_stage,
    *,
    floor_collision: bool = False,
    calibration: bool = False,
) -> None:
    """Premade Simple_Room environment + the colour x shape wall objects.

    ``floor_collision`` makes the tatami a contact surface (walk mode); see
    :func:`add_tatami_floor`.
    """
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    # Physics ground, slightly below the room floor to avoid z-fighting.
    world.scene.add_default_ground_plane(z_position=-0.02)

    stage = omni.usd.get_context().get_stage()

    # ---- Premade room ------------------------------------------------------
    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:  # older Isaac Sim layout
        from omni.isaac.nucleus import get_assets_root_path

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Could not resolve the Isaac assets root. Check your Isaac Sim asset "
            "configuration (Nucleus connection or local asset cache)."
        )

    env_root = "/World/Environment"
    add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd",
        path=env_root,
    )

    # +90° yaw maps the room's -Y wooden wall to the front (+X).
    env_yaw_deg = 90.0

    env_xf = UsdGeom.Xformable(stage.GetPrimAtPath(env_root))
    env_xf.ClearXformOpOrder()
    # [translate, rotateZ] => point' = translate(rotateZ(point)).
    env_xf.AddTranslateOp().Set(Gf.Vec3d(_ENV_OFFSET_X, 0.0, 0.0))
    env_xf.AddRotateZOp().Set(env_yaw_deg)

    # Deactivate the centre table and props where the robot stands.
    prune_name_substr = ("table", "tablecloth", "prop")
    env_prim = stage.GetPrimAtPath(env_root)
    if env_prim and env_prim.IsValid():
        for prim in Usd.PrimRange(env_prim):
            name = prim.GetName().lower()
            if any(s in name for s in prune_name_substr):
                prim.SetActive(False)

    # ---- Tatami floor (world frame) ----------------------------------------
    add_tatami_floor(stage, collision=floor_collision)

    # ---- Wall objects ------------------------------------------------------
    # Built hidden; RobotPoseResetter places and shows them per episode.
    if calibration:
        add_calibration_tag(stage)
    else:
        add_wall_objects(stage)
