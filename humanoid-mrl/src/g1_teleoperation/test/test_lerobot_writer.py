# MIT License
"""Tests for the LeRobot dataset writer (pure disk I/O, no ROS)."""

import json

import numpy as np
import pytest

from g1_definitions import definitions
from g1_teleoperation.lerobot_writer import LeRobotDatasetWriter


def _frame(task="drill", tip_dist=0.1):
    rng = np.random.default_rng(0)
    return {
        "observation.state": rng.random(definitions.STATE_DIM).astype(np.float32),
        "action": rng.random(definitions.ACTION_DIM).astype(np.float32),
        "tip_target_9d": rng.random(9).astype(np.float32),
        "tip_target_dist": tip_dist,
        "frames": {"ego_view": np.zeros((32, 48, 3), dtype=np.uint8)},
        "task": task,
    }


@pytest.fixture
def writer(tmp_path):
    return LeRobotDatasetWriter(tmp_path, fps=30, camera_keys=["ego_view"])


def test_init_scaffolds_dataset(writer, tmp_path):
    assert (tmp_path / "meta" / "modality.json").exists()
    assert (tmp_path / "meta" / "info.json").exists()
    assert (tmp_path / "data" / "chunk-000").is_dir()
    assert (tmp_path / "videos" / "chunk-000" / "observation.images.ego_view").is_dir()

    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())
    assert modality["state"].keys() == definitions.STATE_KEY_SIZES.keys()
    assert modality["action"].keys() == definitions.ACTION_KEY_SIZES.keys()


def test_save_episode_writes_parquet_video_and_meta(writer, tmp_path):
    pd = pytest.importorskip("pandas")

    ep = writer.save_episode([_frame() for _ in range(5)])
    assert ep == 0
    assert writer.episode_index == 1
    assert writer.global_frame_index == 5

    parquet = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(parquet)
    assert len(df) == 5
    assert df["timestamp"].tolist() == pytest.approx([i / 30 for i in range(5)])
    assert bool(df["next.done"].iloc[-1]) and not df["next.done"].iloc[:-1].any()

    video = (
        tmp_path
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    assert video.exists() and video.stat().st_size > 0

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 5
    # Declared video shape refreshed from the real frame size.
    assert info["features"]["observation.images.ego_view"]["shape"] == [3, 32, 48]


def test_task_indices_are_deduplicated(writer, tmp_path):
    writer.save_episode([_frame(task="drill")])
    writer.save_episode([_frame(task="drill")])
    writer.save_episode([_frame(task="other")])

    tasks = [
        json.loads(line)
        for line in (tmp_path / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [t["task"] for t in tasks] == ["drill", "valid", "other"]


def test_resume_from_existing_dataset(writer, tmp_path):
    writer.save_episode([_frame() for _ in range(3)])
    writer.save_episode([_frame() for _ in range(2)])

    resumed = LeRobotDatasetWriter(tmp_path, fps=30, camera_keys=["ego_view"])
    assert resumed.episode_index == 2
    assert resumed.global_frame_index == 5
