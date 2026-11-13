#!/usr/bin/env python3
"""Pre-compute GR00T visual embeddings for every frame in the g1_dataset.

This script must be run in an environment where:
  - The GR00T server is reachable (default localhost:5555)
  - ffmpeg is on PATH (used by gr00t.utils.video_utils)
  - gr00t and g1_definitions packages are importable

The easiest way to satisfy all three is to run it inside the GR00T Docker
container while the server is also running there:

    docker exec -it <container> \\
        python /workspace/scripts/compute_dataset_embeddings.py \\
            --dataset /workspace/g1_dataset

Output: <dataset>/visual_embeddings.npy   — float32 (N, hidden_size)
        <dataset>/visual_embeddings_meta.npy — int32 (N, 2) columns [episode_idx, frame_idx]
        (rows match parquet files iterated in alphabetical order, then frames within each episode)

With --subsample N only every N-th frame is embedded; the meta file tracks which
frames were actually stored so the scorer can look them up correctly.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gr00t.policy.server_client import PolicyClient
from gr00t.utils.video_utils import get_all_frames

STATE_KEY_SIZES: dict[str, int] = {
    "left_palm_eef_9d": 9,
    "right_palm_eef_9d": 9,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
}

_idx = 0
STATE_SLICES: dict[str, tuple[int, int]] = {}
for _k, _n in STATE_KEY_SIZES.items():
    STATE_SLICES[_k] = (_idx, _idx + _n)
    _idx += _n


def _build_obs(
    frames: np.ndarray,
    frame_idx: int,
    state_row: np.ndarray,
    task: str,
    video_keys: list,
    state_keys: list,
    language_keys: list,
    video_horizon: int,
    state_horizon: int,
) -> dict:
    # Temporal window: use actual past frames, padding with the first frame
    indices = [max(0, frame_idx - (video_horizon - 1 - j)) for j in range(video_horizon)]
    img_seq = np.stack([frames[i] for i in indices], axis=0)  # (horizon, H, W, C)

    state_row = state_row.astype(np.float32)
    state_map = {
        key: state_row[start:end]
        for key, (start, end) in STATE_SLICES.items()
    }

    obs = {"video": {}, "state": {}, "language": {}}
    for key in language_keys:
        obs["language"][key] = [[task]]
    for key in video_keys:
        obs["video"][key] = img_seq[np.newaxis, ...]  # (1, horizon, H, W, C)
    for key in state_keys:
        if key in state_map:
            seq = np.stack([state_map[key]] * state_horizon, axis=0)
        else:
            size = STATE_KEY_SIZES.get(key, 1)
            seq = np.zeros((state_horizon, size), dtype=np.float32)
        obs["state"][key] = seq[np.newaxis, ...]  # (1, horizon, D)

    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute GR00T visual embeddings")
    parser.add_argument("--dataset", default="g1_dataset", help="Path to g1_dataset root")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Use every N-th frame (1 = all frames, 5 = 20%%, etc.)",
    )
    parser.add_argument(
        "--out-embeddings",
        default=None,
        help="Output path for embeddings .npy (default: <dataset>/visual_embeddings.npy)",
    )
    parser.add_argument(
        "--out-meta",
        default=None,
        help="Output path for meta .npy (default: <dataset>/visual_embeddings_meta.npy)",
    )
    args = parser.parse_args()

    out_embeddings = args.out_embeddings or str(Path(args.dataset) / "visual_embeddings.npy")
    out_meta = args.out_meta or str(Path(args.dataset) / "visual_embeddings_meta.npy")

    print(f"Connecting to GR00T server at {args.host}:{args.port} ...")
    policy = PolicyClient(host=args.host, port=args.port, timeout_ms=30_000, strict=False)
    mc = policy.get_modality_config()
    video_horizon = len(mc["video"].delta_indices)
    state_horizon = len(mc["state"].delta_indices)
    video_keys = mc["video"].modality_keys
    state_keys = mc["state"].modality_keys
    language_keys = mc["language"].modality_keys
    print(f"Modality config: video_horizon={video_horizon}  state_horizon={state_horizon}")

    parquet_files = sorted(
        glob.glob(str(Path(args.dataset) / "data" / "**" / "*.parquet"), recursive=True)
    )
    print(f"Found {len(parquet_files)} episodes  (subsample={args.subsample})")

    all_embeddings: list[np.ndarray] = []
    all_meta: list[tuple[int, int]] = []  # (episode_idx, frame_idx)
    total_processed = 0

    for pf in parquet_files:
        ep_idx = int(Path(pf).stem.split("_")[1])
        video_path = str(
            Path(args.dataset)
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{ep_idx:06d}.mp4"
        )

        df = pd.read_parquet(pf)
        n_frames = len(df)

        try:
            frames, _ = get_all_frames(video_path, video_backend="ffmpeg")
        except Exception as e:
            print(f"  [episode {ep_idx:03d}] SKIP — could not read video: {e}", file=sys.stderr)
            continue

        if len(frames) != n_frames:
            print(
                f"  [episode {ep_idx:03d}] WARNING: {len(frames)} video frames "
                f"!= {n_frames} parquet rows — truncating to min"
            )
            n_frames = min(len(frames), n_frames)

        sampled = range(0, n_frames, args.subsample)
        for frame_idx in sampled:
            state_row = np.array(df.iloc[frame_idx]["observation.state"], dtype=np.float32)
            task = str(df.iloc[frame_idx]["annotation.human.action.task_description"])

            obs = _build_obs(
                frames,
                frame_idx,
                state_row,
                task,
                video_keys,
                state_keys,
                language_keys,
                video_horizon,
                state_horizon,
            )
            try:
                # Backbone features are opt-in: the server omits them unless asked
                # (they are ~4 MB per response and nothing else needs them).
                _, info = policy.get_action(
                    observation=obs, options={"return_backbone_features": True}
                )
            except Exception as e:
                print(
                    f"  [episode {ep_idx:03d} frame {frame_idx}] inference error: {e}",
                    file=sys.stderr,
                )
                continue

            backbone_features = info.get("backbone_features")  # (1, seq_len, hidden_size)
            image_mask = info.get("image_mask")  # (1, seq_len) bool
            if backbone_features is None or image_mask is None:
                print(
                    "ERROR: server returned no 'backbone_features'/'image_mask' in info "
                    "despite options={'return_backbone_features': True}. Make sure a "
                    "current Gr00tPolicy is running (restart the server after pulling).",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Mean-pool the raw VLM latent over image-token positions -> (hidden_size,).
            feat = backbone_features[0][image_mask[0]].mean(axis=0)
            all_embeddings.append(feat.astype(np.float32))  # (hidden_size,)
            all_meta.append((ep_idx, frame_idx))
            total_processed += 1

        print(
            f"  [episode {ep_idx:03d}] {len(sampled)} frames embedded  "
            f"(total so far: {total_processed})"
        )

    embeddings = np.stack(all_embeddings, axis=0)  # (N, hidden_size)
    meta = np.array(all_meta, dtype=np.int32)  # (N, 2)

    np.save(out_embeddings, embeddings)
    np.save(out_meta, meta)

    print(f"\nDone. Saved {len(embeddings)} embeddings ({embeddings.shape[1]}-dim)")
    print(f"  embeddings → '{out_embeddings}'")
    print(f"  meta       → '{out_meta}'")


if __name__ == "__main__":
    main()
