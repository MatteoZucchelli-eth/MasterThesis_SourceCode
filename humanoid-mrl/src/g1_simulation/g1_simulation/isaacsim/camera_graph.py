# MIT License
"""OmniGraph builder for publishing an Isaac USD camera over ROS2."""

from __future__ import annotations

import json
import logging
import os

import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from pxr import UsdGeom

# Re-exported from camera_specs (importable outside Isaac) for consumers of this module.
from g1_simulation.isaacsim.camera_specs import (  # noqa: F401
    HAND_CAMERA_CENTRE_PX,
    HAND_CAMERA_FX_PX,
    HAND_CAMERA_SIZE_PX,
    HEAD_CAMERA_CENTRE_PX,
    HEAD_CAMERA_FX_PX,
    HEAD_CAMERA_SIZE_PX,
)

_LATEST_FRAME_QOS = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "bestEffort",
        "durability": "volatile",
        # Isaac rejects the whole profile if any key is missing, and then never
        # creates the publisher (logs "Missing key: ..." per frame). Check
        # ``ros2 topic info <rgb topic>`` shows 1 publisher.
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
)
"""Publisher QoS for the RGB stream: keep the newest frame, never retransmit.

A queued stream lags sim time without bound, so the recorder would pair frames
with stale state; dropping an overtaken frame is always preferable.
"""


QOS_PROFILE_ENV = "G1_CAMERA_QOS_PROFILE"
"""Env override for the published QoS profile. Unset uses _LATEST_FRAME_QOS.

``default`` publishes with Isaac's own QoS (keeping ``queueSize``); a JSON
string replaces the profile. Avoids an image rebuild to change the QoS.
"""


HEAD_DEPTH_TOPIC = "/camera/d435/depth/image_rect_raw"
"""Topic the simulated head depth stream publishes on.

Same name as realsense2_camera's rectified D435 depth, so consumers are
identical in sim and on the robot. A rendered pinhole is rectified by
construction.
"""

DEPTH_FRAME_SKIP = 1
"""Rendered frames skipped between published depth frames — 1 gives 15 Hz.

Matches the robot's 15 Hz D435 depth and halves the bandwidth of a 1.23 MB
32FC1 frame.
"""

DEPTH_TOPIC_ENV = "G1_HEAD_DEPTH_TOPIC"
"""Env override for the head depth topic. Empty string disables the stream
(e.g. on headless collection workers, where nothing consumes it).
"""

DEPTH_FRAME_SKIP_ENV = "G1_HEAD_DEPTH_FRAME_SKIP"
"""Env override for :data:`DEPTH_FRAME_SKIP`."""


def head_depth_topic() -> str:
    """The head depth topic to publish on; empty means do not publish at all."""
    override = os.environ.get(DEPTH_TOPIC_ENV)
    if override is None:
        return HEAD_DEPTH_TOPIC
    return override.strip()


def head_depth_frame_skip() -> int:
    """Frames to skip between depth publishes, from the env or the default."""
    raw = os.environ.get(DEPTH_FRAME_SKIP_ENV, "").strip()
    if not raw:
        return DEPTH_FRAME_SKIP
    try:
        return max(0, int(raw))
    except ValueError:
        logging.warning(
            "Camera depth: %s=%r is not an integer; using the default %d",
            DEPTH_FRAME_SKIP_ENV,
            raw,
            DEPTH_FRAME_SKIP,
        )
        return DEPTH_FRAME_SKIP


def _set_latest_frame_qos(node_path: str) -> None:
    """Point a ROS2 publisher node at the latest-frame-only QoS.

    Unknown input names are logged and skipped. A profile Isaac rejects fails
    silently later (no publisher is created), so after changes check
    ``ros2 topic info <rgb topic>`` for a publisher count of 1.
    """
    profile = os.environ.get(QOS_PROFILE_ENV, "").strip()
    if profile.lower() == "default":
        logging.warning(
            "Camera QoS: %s=default — publishing with Isaac's own QoS. The image "
            "stream can queue under load; watch the recorder's staleness error.",
            QOS_PROFILE_ENV,
        )
        attrs = (("inputs:queueSize", 1),)
    else:
        attrs = (
            ("inputs:qosProfile", profile or _LATEST_FRAME_QOS),
            ("inputs:queueSize", 1),
        )
    for attr_name, value in attrs:
        path = f"{node_path}.{attr_name}"
        try:
            og.Controller.set(og.Controller.attribute(path), value)
        except Exception as exc:  # noqa: BLE001 - any failure means unsupported
            logging.warning(
                "Camera QoS: could not set %s (%s). This Isaac release keeps the "
                "node's default queue, so the image stream can back up under "
                "load — watch the recorder's staleness error.",
                path,
                exc,
            )


def _define_child_camera(parent_path: str, child_name: str) -> str:
    """Define a fresh camera as a child of ``parent_path`` and return its path.

    The child has no transform ops, so it shares the parent's world pose. Used
    for the wrist camera, whose asset prim ignores lens writes; a freshly
    defined prim has no composition history and honours them. Non-lens
    attributes are copied from the parent. A lens that never got written
    renders as an extreme telephoto (USD default focalLength 50).
    """
    stage = omni.usd.get_context().get_stage()
    parent = UsdGeom.Camera(stage.GetPrimAtPath(parent_path))
    child_path = f"{parent_path}/{child_name}"
    child = UsdGeom.Camera.Define(stage, child_path)

    # Copy everything except the lens (aperture is needed: focal/aperture is the
    # normalised focal).
    copied = []
    for name in (
        "HorizontalAperture",
        "VerticalAperture",
        "ClippingRange",
        "ClippingPlanes",
        "Projection",
        "FocusDistance",
        "FStop",
        "StereoRole",
        "Exposure",
    ):
        try:
            source = getattr(parent, f"Get{name}Attr")()
            value = source.Get()
        except Exception:  # noqa: BLE001 - attribute absent in this USD version
            continue
        if value is None:
            continue
        try:
            getattr(child, f"Create{name}Attr")().Set(value)
            copied.append(f"{name}={value}")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Camera %s: could not copy %s (%s)", child_path, name, exc)

    # Cleared explicitly: Define on an existing prim keeps its xform ops.
    child.ClearXformOpOrder()
    logging.info(
        "Camera %s: defined as a child of %s with an identity transform; copied %s",
        child_path,
        parent_path,
        ", ".join(copied) or "nothing",
    )
    return child_path


def _set_principal_point(
    camera_prim_path: str,
    centre_px: tuple[float, float],
    width_px: int,
    height_px: int,
) -> tuple[float, float]:
    """Offset the rendered optical centre, and return where it actually landed.

    Expressed as USD aperture offsets (unit-agnostic, as in
    :func:`_set_focal_length`). The value is read back after writing; on any
    failure the image centre is returned, since that is what gets rendered.

    Signs, verified from rendered images (image v points down, USD +Y up):

        horizontalApertureOffset = -(cx - width/2)  * per_px
        verticalApertureOffset   =  (cy - height/2) * per_px

    Re-verify visually after changes (e.g. track the rigidly mounted drill): the
    published camera_info is built from the same numbers and cannot catch a
    sign error.
    """
    cx_px, cy_px = centre_px
    try:
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
        aperture = camera.GetHorizontalApertureAttr().Get()
        if not aperture:
            raise ValueError(f"horizontalAperture is {aperture!r}")
        per_px = aperture / float(width_px)
        h_attr = camera.GetHorizontalApertureOffsetAttr()
        v_attr = camera.GetVerticalApertureOffsetAttr()
        h_attr.Set(-(cx_px - width_px / 2.0) * per_px)
        v_attr.Set((cy_px - height_px / 2.0) * per_px)

        # Read back in pixels, comparable to the published cx/cy.
        h_read, v_read = h_attr.Get(), v_attr.Get()
        cx_read = width_px / 2.0 - h_read / per_px
        cy_read = height_px / 2.0 + v_read / per_px
        logging.info(
            "Camera %s: principal point requested (%.3f, %.3f) px, prim now "
            "reports (%.3f, %.3f) px [aperture offsets %+.6f / %+.6f, "
            "horizontalAperture %.4f, %.6f aperture units per px]",
            camera_prim_path,
            cx_px,
            cy_px,
            cx_read,
            cy_read,
            h_read,
            v_read,
            aperture,
            per_px,
        )
        if abs(cx_read - cx_px) > 0.01 or abs(cy_read - cy_px) > 0.01:
            logging.warning(
                "Camera %s: the write did not stick -- asked for (%.3f, %.3f) "
                "px, prim reports (%.3f, %.3f). Publishing the prim's value, "
                "since that is what gets rendered.",
                camera_prim_path,
                cx_px,
                cy_px,
                cx_read,
                cy_read,
            )
        return (cx_read, cy_read)
    except Exception as exc:  # noqa: BLE001 - any failure leaves it centred
        logging.warning(
            "Camera %s: could not set the aperture offsets (%s). The rendered "
            "optical centre stays at the image centre, so that is what will be "
            "published -- a constant back-projection bias against the real "
            "camera, but an honest one.",
            camera_prim_path,
            exc,
        )
        return (width_px / 2.0, height_px / 2.0)


def _set_focal_length(camera_prim_path: str, fx_px: float, width_px: int) -> None:
    """Give a USD camera the real sensor's focal length, in USD's own units.

    Sets focalLength = (fx / width) * horizontalAperture, so asset units cancel.
    verticalAperture is left alone: Isaac derives the vertical FOV from the
    render product's aspect. Failures warn rather than raise.
    """
    target_norm = fx_px / float(width_px)
    try:
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
        aperture_attr = camera.GetHorizontalApertureAttr()
        focal_attr = camera.GetFocalLengthAttr()
        aperture = aperture_attr.Get()
        if not aperture:
            raise ValueError(f"horizontalAperture is {aperture!r}")
        before = focal_attr.Get()
        focal_attr.Set(target_norm * aperture)
        logging.info(
            "Camera %s: focalLength %.4f -> %.4f (horizontalAperture %.4f), "
            "normalised focal %.6f -> %.6f, i.e. fx = %.2f px at %d px wide",
            camera_prim_path,
            before if before is not None else float("nan"),
            target_norm * aperture,
            aperture,
            (before / aperture) if before else float("nan"),
            target_norm,
            fx_px,
            width_px,
        )
    except Exception as exc:  # noqa: BLE001 - any failure leaves the asset's lens
        logging.warning(
            "Camera %s: could not set focalLength (%s). The prim keeps whatever "
            "lens the USD asset was authored with, which is NOT known to match "
            "the real sensor -- check camera_info before trusting any frame.",
            camera_prim_path,
            exc,
        )


def _depth_publisher_graph(depth_topic: str, frame_id: str):
    """Graph fragments ``(nodes, connections, values)`` for a depth publisher.

    Must be spliced into the same ``og.Controller.edit`` as the rest of the
    graph: a later edit does not resolve bare node names from an earlier one.

    Bound to the RGB render product, so depth shares the colour lens, pose and
    ``camera_info`` (the robot equivalent is the aligned D435 stream).

    Isaac's ``depth`` type is ``distance_to_image_plane`` (z along the optical
    axis), as g1_common.depth expects. A ray-length annotator would bow a flat
    wall by up to ~12% at the frame edge; wall_plane_node's fit rms growing
    from mm to cm is the symptom.
    """
    nodes = [("cameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper")]
    connections = [
        ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
        (
            "getRenderProduct.outputs:renderProductPath",
            "cameraHelperDepth.inputs:renderProductPath",
        ),
    ]
    values = [
        ("cameraHelperDepth.inputs:frameId", frame_id),
        ("cameraHelperDepth.inputs:topicName", depth_topic),
        ("cameraHelperDepth.inputs:type", "depth"),
    ]
    return nodes, connections, values


def _configure_depth_publisher(
    graph_path: str, depth_topic: str, frame_id: str, frame_skip: int
) -> None:
    """Apply the depth helper's QoS and publish rate after the graph is built,
    so an unknown input name warns rather than aborts."""
    node_path = f"{graph_path}/cameraHelperDepth"
    _set_latest_frame_qos(node_path)
    if frame_skip > 0:
        try:
            og.Controller.set(
                og.Controller.attribute(f"{node_path}.inputs:frameSkipCount"),
                int(frame_skip),
            )
        except Exception as exc:  # noqa: BLE001 - unsupported input name
            logging.warning(
                "Camera depth: could not set frameSkipCount (%s). Depth will "
                "publish at the full render rate, which is ~4x the bytes of "
                "the RGB stream beside it — watch the recorder's staleness "
                "error, or disable depth with %s=''.",
                exc,
                DEPTH_TOPIC_ENV,
            )
    logging.info(
        "Camera depth: publishing '%s' (frame_id=%s) as 32FC1 distance to the "
        "image plane, every %d render(s), off the same render product as the "
        "RGB stream. Disable with %s=''.",
        depth_topic,
        frame_id,
        frame_skip + 1,
        DEPTH_TOPIC_ENV,
    )


def build_camera_graph(
    graph_path: str,
    camera_prim_path: str,
    rgb_topic: str,
    frame_id: str,
    width: int,
    height: int,
    fx_px: float,
    centre_px: tuple[float, float],
    child_camera_name: str | None = None,
    depth_topic: str = "",
    depth_frame_skip: int = 0,
) -> tuple[float, float]:
    """Build an OmniGraph publishing a USD camera's RGB (and optionally depth)
    over ROS2; return the principal point actually rendered.

    Used for both head (d435) and wrist (hand_cam) cameras. camera_info is not
    published here: Isaac's CameraInfoHelper pins cx, cy to the image centre, so
    ros_bridge publishes it using the returned centre.

    The render product is the sensor's native aspect; the policy's square crop
    is cut downstream by ``g1_common.camera.build_frame_preparer``, as on the
    robot. ``fx_px`` and ``centre_px`` come from the hardware camera_info.

    Depth is published only when ``depth_topic`` is non-empty (head camera,
    see :func:`head_depth_topic`).
    """
    # Lens set before the render product exists, so the first frame is correct.
    # A camera whose prim ignores its lens is rendered through a child prim.
    if child_camera_name:
        camera_prim_path = _define_child_camera(camera_prim_path, child_camera_name)

    _set_focal_length(camera_prim_path, fx_px, width)
    rendered_centre = _set_principal_point(camera_prim_path, centre_px, width, height)

    nodes = [
        ("OnTick", "omni.graph.action.OnTick"),
        ("getRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
        ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
    ]
    connections = [
        ("OnTick.outputs:tick", "getRenderProduct.inputs:execIn"),
        ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
        (
            "getRenderProduct.outputs:renderProductPath",
            "setCamera.inputs:renderProductPath",
        ),
        ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
        (
            "getRenderProduct.outputs:renderProductPath",
            "cameraHelperRgb.inputs:renderProductPath",
        ),
    ]
    values = [
        ("cameraHelperRgb.inputs:frameId", frame_id),
        ("cameraHelperRgb.inputs:topicName", rgb_topic),
        ("cameraHelperRgb.inputs:type", "rgb"),
        ("getRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
        ("getRenderProduct.inputs:width", width),
        ("getRenderProduct.inputs:height", height),
        ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
    ]
    # Same edit as the rest of the graph; see _depth_publisher_graph.
    if depth_topic:
        depth_nodes, depth_connections, depth_values = _depth_publisher_graph(
            depth_topic, frame_id
        )
        nodes += depth_nodes
        connections += depth_connections
        values += depth_values

    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: nodes,
            keys.CONNECT: connections,
            keys.SET_VALUES: values,
        },
    )
    # After the edit so unsupported inputs only warn; before evaluate so the
    # first tick uses them.
    _set_latest_frame_qos(f"{graph_path}/cameraHelperRgb")
    if depth_topic:
        _configure_depth_publisher(graph_path, depth_topic, frame_id, depth_frame_skip)
    og.Controller.evaluate_sync(graph)
    return rendered_centre
