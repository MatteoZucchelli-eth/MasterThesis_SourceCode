# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recover the wrist-camera geometry from the dataset itself, rather than assuming it.

Two constants come out of this, both properties of the rig and not of any model:

  focal length   the heatmap sidecar carries both `depth_z` (Z) and `ray_dist`
                 (sqrt(X^2+Y^2+Z^2)) for the same labelled pixel, so
                 (ray/Z)^2 - 1 = ((u-cx)^2 + (v-cy)^2) / f^2  solves for f.
  camera->tip    the sidecar's (u, v, depth_z) give the target in the camera frame;
  extrinsics     `next.tip_target_9d` gives the same point in the drill-tip frame.
                 Kabsch on the two point sets gives the rigid transform -- and it IS
                 rigid, because the wrist camera and the drill tip share a link.

The rotation is what `tip_target_cam_axes` in the model config holds. The tip-target
head splits its position loss into a lateral part and a depth part, and "depth" means
along the CAMERA's optical axis, which is not any axis of the label's own frame: this
script is what establishes that the optical axis sits at [0.9037, -0.0012, -0.4281] in
the tip frame, i.e. that tip-frame z is not depth. RE-RUN IT and update
`tip_target_cam_axes` if the wrist camera is ever remounted or re-calibrated, otherwise
that loss silently splits along the wrong axes.

Measured on g1_dataset (44,725 point pairs): focal 388.09 px at the raw 720px camera,
rigid fit rms residual 1.18 mm.

Usage:
    python scripts/recover_camera_tip_geometry.py --dataset-path ./g1_dataset
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def recover_focal(labels: pd.DataFrame, cx: float, cy: float) -> float:
    """Solve (ray/Z)^2 - 1 = r_px^2 / f^2 for f (least squares through the origin)."""
    r2 = (labels["u"] - cx) ** 2 + (labels["v"] - cy) ** 2
    lhs = (labels["ray_dist"] / labels["depth_z"]) ** 2 - 1.0
    ok = (lhs > 1e-9) & (r2 > 1.0)
    inv_f2 = (lhs[ok] * r2[ok]).sum() / (r2[ok] ** 2).sum()
    return float(1.0 / np.sqrt(inv_f2))


def recover_extrinsics(cam_pts: np.ndarray, tip_pts: np.ndarray):
    """Kabsch: rigid (R, t) with tip ~= cam @ R.T + t. Returns (R, t, rms_residual_m)."""
    cm, tm = cam_pts.mean(0), tip_pts.mean(0)
    u, _, vt = np.linalg.svd((cam_pts - cm).T @ (tip_pts - tm))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    trans = tm - rot @ cm
    resid = np.linalg.norm(tip_pts - (cam_pts @ rot.T + trans), axis=1)
    return rot, trans, float(np.sqrt((resid**2).mean()))


def load_dataset_geometry(dataset_path: Path, max_episodes: int, cx: float, cy: float):
    label_files = sorted(
        glob.glob(str(dataset_path / "heatmap_labels" / "chunk-*" / "episode_*.parquet"))
    )
    if not label_files:
        raise FileNotFoundError(f"No heatmap_labels/ parquets under {dataset_path}")
    step = max(1, len(label_files) // max_episodes)
    sample = label_files[::step][:max_episodes]

    cols = [
        "frame_index",
        "u",
        "v",
        "depth_z",
        "ray_dist",
        "tip_target_dist",
        "label_ok",
        "visible",
    ]
    labels, cam_pts, tip_pts, train_dist = [], [], [], []
    for label_path in sample:
        lab = pd.read_parquet(label_path, columns=cols)
        valid = lab[lab["label_ok"]].dropna(subset=["tip_target_dist"])
        train_dist.append(valid["tip_target_dist"])
        vis = valid[valid["visible"]].dropna(subset=["u", "v", "depth_z", "ray_dist"])
        if vis.empty:
            continue
        labels.append(vis)
        data_path = Path(str(label_path).replace("heatmap_labels/", "data/"))
        if not data_path.exists():
            continue
        dat = pd.read_parquet(data_path, columns=["next.tip_target_9d"]).reset_index(
            names="frame_index"
        )
        merged = vis.merge(dat, on="frame_index")
        if merged.empty:
            continue
        tip = np.stack(merged["next.tip_target_9d"].to_numpy())[:, :3]
        keep = np.isfinite(tip).all(axis=1)
        z = merged["depth_z"].to_numpy()[keep]
        cam_pts.append(
            np.stack(
                [
                    (merged["u"].to_numpy()[keep] - cx) * z,
                    (merged["v"].to_numpy()[keep] - cy) * z,
                    z,
                ],
                axis=1,
            )
        )
        tip_pts.append(tip[keep])
    return (
        pd.concat(labels, ignore_index=True),
        np.concatenate(cam_pts),
        np.concatenate(tip_pts),
        pd.concat(train_dist, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument(
        "--camera-size",
        type=int,
        default=720,
        help="Raw wrist frame edge in px; the principal point is assumed to be its centre.",
    )
    args = parser.parse_args()

    cx = cy = args.camera_size / 2.0
    labels, cam_pts_scaled, tip_pts, _ = load_dataset_geometry(
        args.dataset_path, args.max_episodes, cx, cy
    )
    focal = recover_focal(labels, cx, cy)
    # cam_pts were built with X = (u-cx)*Z, needing the /f that was unknown until now.
    cam_pts = cam_pts_scaled.copy()
    cam_pts[:, 0] /= focal
    cam_pts[:, 1] /= focal
    rot, trans, resid = recover_extrinsics(cam_pts, tip_pts)

    print("=== recovered geometry (from the dataset, not assumed) ===")
    print(
        f"  focal length          {focal:.2f} px @ {args.camera_size}px camera "
        f"(FOV {2 * np.degrees(np.arctan(cx / focal)):.1f} deg)"
    )
    print(f"  1 camera px           {1000 / focal:.3f} mm per metre of depth")
    print(
        f"  camera->tip fit       rigid, rms residual {resid * 1000:.2f} mm "
        f"over {len(cam_pts)} points"
    )
    print(f"  optical axis in tip frame  {np.round(rot[:, 2], 4)}")
    print()
    print("Paste into Gr00tN1d7Config.tip_target_cam_axes (rows = camera X, Y, Z-in-tip):")
    print("    [")
    for row in rot.T:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")
    print("    ]")
    depth_axis = rot[:, 2]
    print(
        f"\n  -> depth = {depth_axis[0]:.4f}*x + {depth_axis[1]:.4f}*y + {depth_axis[2]:.4f}*z "
        "in the tip frame."
    )


if __name__ == "__main__":
    main()
