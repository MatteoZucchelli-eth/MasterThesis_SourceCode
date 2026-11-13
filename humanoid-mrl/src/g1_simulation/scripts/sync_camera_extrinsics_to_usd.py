#!/usr/bin/env python3
# MIT License
"""Command-line front end for g1_simulation.isaacsim.camera_extrinsics.

The sim runs :func:`camera_extrinsics.sync` itself at startup
(``IsaacSimCfg.sync_camera_extrinsics``), so this exists for the times you want
the answer without bringing Isaac up: after a calibration run, from CI, or to
see the numbers before trusting them.

    python3 src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py --dry-run
    python3 src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py
    python3 src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py --check

``--check`` writes nothing and exits non-zero if the USD and the URDF disagree.

In the sim container, `make sync-cameras` is this, and `make sync-cameras
ARGS=--check` is the check. It needs pxr, which will not install on the Jetson
(no aarch64 usd-core wheel).

Isaac's own python cannot import pxr on its own -- USD is a kit extension there,
and kit only puts it on the path once an app has started -- so go through
scripts/isaac_usd_python.sh, which adds it:

    docker compose -f docker/docker-compose.yaml run --rm --no-deps \
        --entrypoint /workspace/scripts/isaac_usd_python.sh simulation-isaacsim \
        /workspace/src/g1_simulation/scripts/sync_camera_extrinsics_to_usd.py

Anywhere x86_64 with `pip install usd-core` it runs directly, no Isaac needed.

Isaac holds the stage in memory, so a running sim keeps its existing cameras
until the stage is reloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Importable without the package installed, which is the point of a script.
sys.path.insert(0, str(REPO_ROOT / "src/g1_simulation"))

from g1_simulation.isaacsim import camera_extrinsics  # noqa: E402

DEFAULT_URDF = (
    REPO_ROOT / "src/g1_state_publisher/urdf/g1_29_dof_with_hand_drill.urdf.xml"
)
DEFAULT_STAGE = (
    REPO_ROOT
    / "src/g1_simulation/assets/isaacsim/g1/g1_29dof_rev_1_0_w_hilti_drill.usd"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument(
        "--dry-run", action="store_true", help="report the deltas and write nothing"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="like --dry-run, but exit non-zero if anything is out of sync",
    )
    args = parser.parse_args()

    try:
        report = camera_extrinsics.sync(
            args.stage, args.urdf, dry_run=args.dry_run or args.check
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print("\n".join(report.lines))
    if args.check and not report.cameras_in_sync:
        print("\n--check: the camera prims do not match the URDF", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
