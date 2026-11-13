# MIT License
"""Write the printable tag the head camera calibration is measured against.

    ros2 run g1_sensors head_camera_calibration_tag --size-m 0.12

Print at 100%, verify one cell with a ruler, and glue to the rigid mocap plate
with the mocap body defined on the printed corners (any offset adds directly to
the measured ``R hh'``). Set ``tag_size_m`` to the measured black-square side
shown in the caption.
"""

from __future__ import annotations

import argparse

import cv2
from g1_common.apriltag import cells_per_side, DEFAULT_FAMILY, printable_tag


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument(
        "--size-m",
        type=float,
        default=0.12,
        # ~145 px in the head d435 frame at 0.5 m, within its ~0.52 m field of view.
        help="side of the BLACK square [m] -- what tag_size_m must be set to",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", default="head_calibration_tag.png")
    args = parser.parse_args(argv)

    image, pitch_m = printable_tag(
        args.tag_id, args.family, size_m=args.size_m, dpi=args.dpi
    )
    side_m = pitch_m * cells_per_side(args.family)
    # Caption below the sheet to keep the quiet zone clear.
    caption = (
        f"{args.family} id {args.tag_id}   black square {side_m * 1e3:.2f} mm"
        f"   cell {pitch_m * 1e3:.2f} mm   tag_size_m={side_m:.5f}"
    )
    height = int(0.06 * image.shape[0])
    canvas = cv2.copyMakeBorder(image, 0, height, 0, 0, cv2.BORDER_CONSTANT, value=255)
    scale = image.shape[1] / 1600.0
    cv2.putText(
        canvas,
        caption,
        (int(10 * scale), image.shape[0] + int(height * 0.6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        0,
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )
    cv2.imwrite(args.output, canvas)
    print(
        f"wrote {args.output}: {caption}\n"
        f"print at 100% (no fit-to-page), measure one cell, and pass "
        f"tag_size_m:={side_m:.5f} to head_camera_calibration."
    )


if __name__ == "__main__":
    main()
