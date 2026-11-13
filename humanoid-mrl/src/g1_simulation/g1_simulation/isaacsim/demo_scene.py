# MIT License
"""Fixed demo wall layout: four groups along the wall, placed by one key.

Places geometry only; the robot is not moved or locked. Groups, in increasing
world Y:

1. two 10 cm squares, red and green
2. two 1 cm squares, red and green (trained size)
3. a red square and a red circle
4. the six trained targets (:data:`wall_objects.TARGETABLE_CELLS`) in a 3x2 grid

Groups 1-3 are demo-only prims under :data:`DEMO_ROOT` (trained prims are never
rescaled). Group 4 moves the real :data:`scene.WALL_OBJECTS` prims, so their
``{colour}_{shape}_target`` TF frames stay valid; held-out cells are hidden.
"""
from __future__ import annotations

import logging

from g1_definitions import wall_objects

from g1_simulation.isaacsim import scene
from g1_simulation.isaacsim.carb_keys import is_key_press

DEMO_ROOT = "/World/DemoShapes"

DEMO_Z = 0.88
"""Height [m] of every group centre, inside ``IsaacSimCfg.target_z_range``."""

DEMO_GROUP_SPACING = 0.80
"""Centre-to-centre distance [m] between groups; keeps one group per ego view (~0.36 m
of wall)."""

_TRAINED_GRID_DY = 0.08
_TRAINED_GRID_DZ = 0.07
"""In-group spacing [m] for the trained-target 3x2 lattice, roughly training-scene
density."""


def _square_pair(size: float):
    """Red and green squares of ``size`` straddling the group centre; gap scales with
    size (min 6 cm)."""
    gap = max(2.5 * size, 0.06)
    return (
        ("red", "square", size, -gap / 2.0, 0.0),
        ("green", "square", size, +gap / 2.0, 0.0),
    )


def _trained_grid():
    """The six trained cells at ``TARGET_SIDE``, laid out 3 wide by 2 high."""
    placements = []
    for i, (colour, shape) in enumerate(wall_objects.TARGETABLE_CELLS):
        col, row = i % 3, i // 3
        placements.append(
            (
                colour,
                shape,
                scene.TARGET_SIDE,
                (col - 1) * _TRAINED_GRID_DY,
                (0.5 - row) * _TRAINED_GRID_DZ,
            )
        )
    return tuple(placements)


GROUPS = (
    {"name": "10 cm squares", "use_cells": False, "objects": _square_pair(0.10)},
    {"name": "1 cm squares", "use_cells": False, "objects": _square_pair(0.01)},
    {
        "name": "red square + red circle",
        "use_cells": False,
        "objects": (
            ("red", "square", scene.TARGET_SIDE, -0.03, 0.0),
            ("red", "circle", scene.TARGET_SIDE, +0.03, 0.0),
        ),
    },
    {"name": "six trained targets", "use_cells": True, "objects": _trained_grid()},
)
"""The layout, in wall order. ``objects`` entries are
``(colour, shape, size, dy, dz)`` offsets from the group centre; ``use_cells``
uses the real WALL_OBJECTS prims instead of demo-only ones.
"""


def group_centre_y(index: int) -> float:
    """World Y of group ``index``, with the whole layout centred on y = 0."""
    return (index - (len(GROUPS) - 1) / 2.0) * DEMO_GROUP_SPACING


class DemoScene:
    """Places the fixed demo layout on the wall on a keypress. Default key: Z.

    Objects only: no robot teleport, no BaseLock, nothing published over ROS.
    Re-pressing re-places the layout (no toggle); the clear keys (``R`` reset,
    ``V`` eval scenario) hide the demo prims.
    """

    def __init__(
        self,
        key: str = "Z",
        clear_keys: tuple[str, ...] = ("R", "V"),
    ):
        self._key = key.upper()
        self._clear_keys = tuple(k.upper() for k in clear_keys)

        # (prim_path, y, z) for demo-only prims; cell -> (y, z) for borrowed trained
        # cells.
        self._demo_prims: list[tuple[str, float, float]] = []
        self._cell_placements: dict[tuple[str, str], tuple[float, float]] = {}
        self._resolve_layout()
        self._build()

        span = self._span_y()
        logging.info(
            "DemoScene ready — press '%s' to place the demo wall layout "
            "(%d groups, %d objects, y from %.2f to %.2f m at z=%.2f m). "
            "%s clears it.",
            self._key,
            len(GROUPS),
            len(self._demo_prims) + len(self._cell_placements),
            span[0],
            span[1],
            DEMO_Z,
            "/".join(self._clear_keys),
        )

    # ---- layout ----------------------------------------------------------

    def _resolve_layout(self) -> None:
        for index, group in enumerate(GROUPS):
            centre_y = group_centre_y(index)
            for colour, shape, size, dy, dz in group["objects"]:
                y, z = centre_y + dy, DEMO_Z + dz
                if group["use_cells"]:
                    cell = (colour, shape)
                    if cell not in scene.WALL_OBJECTS:
                        raise KeyError(
                            f"DemoScene: group '{group['name']}' names a cell the "
                            f"scene does not build: {cell}."
                        )
                    self._cell_placements[cell] = (y, z)
                else:
                    path = self._prim_path(index, colour, shape, size)
                    self._demo_prims.append((path, y, z))

    @staticmethod
    def _prim_path(index: int, colour: str, shape: str, size: float) -> str:
        """Stable prim path for one demo object, e.g. ``G0_RedSquare_100mm``."""
        return (
            f"{DEMO_ROOT}/G{index}_{colour.capitalize()}{shape.capitalize()}"
            f"_{round(size * 1000)}mm"
        )

    def _span_y(self) -> tuple[float, float]:
        ys = [y for _, y, _ in self._demo_prims] + [
            y for y, _ in self._cell_placements.values()
        ]
        return (min(ys), max(ys))

    def _build(self) -> None:
        """Author the demo-only prims at their final poses, hidden."""
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        for index, group in enumerate(GROUPS):
            if group["use_cells"]:
                continue
            centre_y = group_centre_y(index)
            for colour, shape, size, dy, dz in group["objects"]:
                prim = scene.add_wall_shape(
                    stage,
                    self._prim_path(index, colour, shape, size),
                    colour_rgb=scene.EVAL_COLOURS[colour],
                    shape=shape,
                    size=size,
                    pos=(scene.TARGET_X, centre_y + dy, DEMO_Z + dz),
                )
                UsdGeom.Imageable(prim).MakeInvisible()

    # ---- keyboard --------------------------------------------------------

    def on_keyboard_input(self, e) -> None:
        if is_key_press(e, self._key):
            self.show()
        elif any(is_key_press(e, k) for k in self._clear_keys):
            self.hide()

    # ---- placement -------------------------------------------------------

    def show(self) -> None:
        """Put the demo layout on the wall: demo prims visible, cells re-placed."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        for prim_path, _, _ in self._demo_prims:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logging.warning(f"DemoScene: prim {prim_path} not found")
                continue
            UsdGeom.Imageable(prim).MakeVisible()

        # Trained cells are shared with the resetter/eval sequencer, so move them too.
        for cell, (prim_path, _, _) in scene.WALL_OBJECTS.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logging.warning(f"DemoScene: prim {prim_path} not found")
                continue
            imageable = UsdGeom.Imageable(prim)
            if cell not in self._cell_placements:
                # Held-out cell.
                imageable.MakeInvisible()
                continue
            imageable.MakeVisible()
            y, z = self._cell_placements[cell]
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(scene.TARGET_X, y, z))
                    break

        logging.info(
            "DemoScene: placed %d groups (%s) at z=%.2f m, centres y=%s.",
            len(GROUPS),
            ", ".join(g["name"] for g in GROUPS),
            DEMO_Z,
            [round(group_centre_y(i), 2) for i in range(len(GROUPS))],
        )

    def hide(self) -> None:
        """Hide the demo-only prims; trained cells are left to the reset/eval that
        follows."""
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        for prim_path, _, _ in self._demo_prims:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
