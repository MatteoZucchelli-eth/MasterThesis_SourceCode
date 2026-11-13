# MIT License
"""LeRobot v2 dataset writer: directory scaffolding, meta files, video, parquet.

Pure Python — no ROS. The recorder node buffers frames and hands each finished
episode to :meth:`LeRobotDatasetWriter.save_episode`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from g1_definitions import definitions

logger = logging.getLogger(__name__)

# Fallback frame size for info.json before any frame has been seen; the real
# (height, width) is filled in from the first saved frame of each camera.
_DEFAULT_IMG_HW: tuple[int, int] = (720, 720)

# Episodes per chunk-<n> directory; must match chunks_size and the path
# templates in info.json.
_CHUNKS_SIZE: int = 1000


def _chunk_dir(episode_index: int) -> str:
    return f"chunk-{episode_index // _CHUNKS_SIZE:03d}"


def video_column(cam_key: str) -> str:
    """Dataset column / folder name for a camera's video stream."""
    return f"observation.images.{cam_key}"


def _modality_json(camera_keys: list[str]) -> dict:
    def to_slices(slices: dict) -> dict:
        return {key: {"start": s, "end": e} for key, (s, e) in slices.items()}

    return {
        "state": to_slices(definitions.STATE_SLICES),
        "action": to_slices(definitions.ACTION_SLICES),
        "video": {key: {"original_key": video_column(key)} for key in camera_keys},
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }


class LeRobotDatasetWriter:
    """Appends episodes to a GR00T-compatible LeRobot v2 dataset on disk.

    Each frame dict must carry: ``observation.state`` and ``action`` (float32
    vectors), ``tip_target_9d`` (float32[9]), ``tip_target_dist`` (float),
    ``frames`` ({camera_key: rgb image}) and ``task`` (str).
    """

    def __init__(self, root: Path, fps: int, camera_keys: list[str]) -> None:
        self._root = Path(root)
        self._fps = fps
        self._camera_keys = list(camera_keys)

        # Actual (height, width) of each camera, discovered on first save and
        # written into info.json so the declared shape matches the video.
        self._video_shapes: dict[str, tuple[int, int]] = {}

        self.episode_index: int = self._count_existing_episodes()
        self.global_frame_index: int = self._count_existing_frames()
        self._init_structure()

    # ── Dataset initialisation ────────────────────────────────────────────────

    def _init_structure(self) -> None:
        """Create directory tree and write static meta files."""
        root = self._root

        (root / "meta").mkdir(parents=True, exist_ok=True)
        chunk = _chunk_dir(self.episode_index)
        for cam_key in self._camera_keys:
            (root / "videos" / chunk / video_column(cam_key)).mkdir(
                parents=True, exist_ok=True
            )
        (root / "data" / chunk).mkdir(parents=True, exist_ok=True)

        # modality.json – always overwrite to keep it in sync with this writer.
        with open(root / "meta" / "modality.json", "w") as fh:
            json.dump(_modality_json(self._camera_keys), fh, indent=2)

        # info.json – create once; updated incrementally after each episode.
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            info = {
                "robot_type": "g1",
                "embodiment": "G1_REAL",
                "fps": self._fps,
                "codebase_version": "v2.0",
                "chunks_size": _CHUNKS_SIZE,
                "total_chunks": 1,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "total_episodes": self.episode_index,
                "total_frames": self.global_frame_index,
                "total_tasks": 0,
                "total_videos": self.episode_index * len(self._camera_keys),
                "splits": {"train": f"0:{self.episode_index}"},
                "state_dim": definitions.STATE_DIM,
                "action_dim": definitions.ACTION_DIM,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [definitions.STATE_DIM],
                    },
                    "action": {"dtype": "float32", "shape": [definitions.ACTION_DIM]},
                    **{
                        video_column(cam_key): {
                            "dtype": "video",
                            "shape": [3, *_DEFAULT_IMG_HW],
                            "names": ["channels", "height", "width"],
                        }
                        for cam_key in self._camera_keys
                    },
                },
            }
            with open(info_path, "w") as fh:
                json.dump(info, fh, indent=2)

        # Append-only jsonl files – create if missing.
        for fname in ("episodes.jsonl", "tasks.jsonl"):
            fpath = root / "meta" / fname
            if not fpath.exists():
                fpath.touch()

    def _count_existing_episodes(self) -> int:
        path = self._root / "meta" / "episodes.jsonl"
        if not path.exists():
            return 0
        with open(path) as fh:
            return sum(1 for line in fh if line.strip())

    def _count_existing_frames(self) -> int:
        path = self._root / "meta" / "info.json"
        if not path.exists():
            return 0
        with open(path) as fh:
            return json.load(fh).get("total_frames", 0)

    def _get_or_create_task_index(self, task: str) -> int:
        """Return an existing task index or append a new entry to tasks.jsonl."""
        tasks_path = self._root / "meta" / "tasks.jsonl"
        tasks: list[dict] = []
        if tasks_path.exists():
            with open(tasks_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
        for t in tasks:
            if t["task"] == task:
                return int(t["task_index"])
        idx = len(tasks)
        with open(tasks_path, "a") as fh:
            fh.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        return idx

    # ── Episode serialisation ─────────────────────────────────────────────────

    def save_episode(self, frames: list[dict]) -> int:
        """Write one episode (video + parquet + meta) and return its index."""
        ep_idx = self.episode_index
        task_description = frames[0]["task"]
        task_index = self._get_or_create_task_index(task_description)
        valid_index = self._get_or_create_task_index("valid")
        n_frames = len(frames)

        self._save_video(ep_idx, frames)
        self._save_parquet(ep_idx, frames, task_index, valid_index)

        self.global_frame_index += n_frames
        self._append_episode_meta(ep_idx, task_index, n_frames)
        self._update_info_json()
        self.episode_index += 1
        return ep_idx

    def _save_video(self, ep_idx: int, frame_buffer: list[dict]) -> None:
        """Write one mp4 per camera, recording the real frame size for info.json."""
        for cam_key in self._camera_keys:
            frames = [
                fd["frames"][cam_key]
                for fd in frame_buffer
                if fd["frames"].get(cam_key) is not None
            ]
            if not frames:
                logger.warning(
                    "No frames for camera '%s' in episode; skipping its video.",
                    cam_key,
                )
                continue
            h, w, _ = frames[0].shape
            self._video_shapes[cam_key] = (h, w)
            video_dir = (
                self._root / "videos" / _chunk_dir(ep_idx) / video_column(cam_key)
            )
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self._fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

    def _save_parquet(
        self, ep_idx: int, frame_buffer: list[dict], task_index: int, valid_index: int
    ) -> None:
        if any(np.isnan(fd["tip_target_dist"]) for fd in frame_buffer):
            logger.warning(
                "Episode %d: some frames have no tip→target label (NaN). "
                "Was the red_target TF published for the whole episode?",
                ep_idx,
            )
        n_frames = len(frame_buffer)
        rows = []
        for i, fd in enumerate(frame_buffer):
            rows.append(
                {
                    "observation.state": fd["observation.state"].tolist(),
                    "action": fd["action"].tolist(),
                    # Frame-index timestamp so dt == 1/fps exactly (LeRobot validates
                    # it).
                    "timestamp": i / self._fps,
                    "annotation.human.action.task_description": task_index,
                    "annotation.human.validity": valid_index,
                    "task_index": task_index,
                    "episode_index": ep_idx,
                    "index": self.global_frame_index + i,
                    "next.reward": 0.0,
                    "next.done": i == n_frames - 1,
                    # Task label: red_target pose in the drill-tip frame, xyz(3) +
                    # rot6d(6), and ‖xyz‖. Auxiliary regression targets, not policy
                    # inputs.
                    "next.tip_target_9d": fd["tip_target_9d"].tolist(),
                    "next.tip_target_dist": fd["tip_target_dist"],
                }
            )
        df = pd.DataFrame(rows)
        out_dir = self._root / "data" / _chunk_dir(ep_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"episode_{ep_idx:06d}.parquet", index=False)

    def _append_episode_meta(self, ep_idx: int, task_index: int, length: int) -> None:
        path = self._root / "meta" / "episodes.jsonl"
        entry = {"episode_index": ep_idx, "tasks": [task_index], "length": length}
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _update_info_json(self) -> None:
        path = self._root / "meta" / "info.json"
        with open(path) as fh:
            info = json.load(fh)

        n_tasks = 0
        tasks_path = self._root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as fh:
                n_tasks = sum(1 for line in fh if line.strip())

        next_ep = self.episode_index + 1
        info["total_episodes"] = next_ep
        info["total_frames"] = self.global_frame_index
        info["total_tasks"] = n_tasks
        info["total_videos"] = next_ep * len(self._camera_keys)
        info["total_chunks"] = (next_ep - 1) // _CHUNKS_SIZE + 1
        info["splits"] = {"train": f"0:{next_ep}"}

        # Refresh each camera's declared shape from the real frame size once known.
        for cam_key, (h, w) in self._video_shapes.items():
            feat = info.get("features", {}).get(video_column(cam_key))
            if feat is not None:
                feat["shape"] = [3, h, w]

        with open(path, "w") as fh:
            json.dump(info, fh, indent=2)
