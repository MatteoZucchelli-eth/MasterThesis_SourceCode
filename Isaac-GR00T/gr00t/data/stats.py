#!/usr/bin/env python

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

"""
Calculate dataset statistics for LeRobot datasets.

Usage:
    python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag <embodiment_tag>
    python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag <embodiment_tag> --modality-config-path <config.py>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations.
    modality_config_path: Optional path to a .py config file for custom embodiment tags not in the built-in registry.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.types import ActionRepresentation, ActionType, EmbodimentTag, ModalityConfig
from gr00t.data.utils import to_json_serializable


LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    def __init__(
        self,
        dataset_path: Path | str,
        embodiment_tag: EmbodimentTag,
        action_key: str,
        embodiment_config: dict[str, ModalityConfig] | None = None,
    ):
        """
        Args:
            embodiment_config: The resolved MODALITY_CONFIGS[embodiment_tag.value] entry to use.
                If None, it is looked up from the global MODALITY_CONFIGS registry. Callers that
                construct this loader inside a spawned worker process (e.g. via
                ProcessPoolExecutor with the "spawn" context) MUST pass this explicitly: custom
                embodiments (e.g. NEW_EMBODIMENT) are registered into MODALITY_CONFIGS at runtime
                via register_modality_config(), and that registration lives only in the process
                that ran it -- a freshly spawned worker re-imports gr00t.data.stats from scratch
                and never replays it, so the registry lookup would raise KeyError there.
        """
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        if embodiment_config is None:
            embodiment_config = MODALITY_CONFIGS[embodiment_tag.value]
        # Check action config
        assert action_key in embodiment_config["action"].modality_keys
        idx = embodiment_config["action"].modality_keys.index(action_key)
        action_configs = embodiment_config["action"].action_configs
        assert action_configs is not None, embodiment_config["action"]
        self.action_config = action_configs[idx]
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=embodiment_config["action"].delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        assert state_key in embodiment_config["state"].modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=embodiment_config["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )
        self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        df = self.loader[trajectory_id]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)
        trajectories = []
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]
            if self.action_config.type == ActionType.EEF:
                action_format = self.action_config.format
                reference_frame = EndEffectorPose.from_action_format(last_state, action_format)
                traj = EndEffectorActionChunk.from_array(actions, action_format).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(traj.to(action_format).astype(np.float32))
            elif self.action_config.type == ActionType.NON_EEF:
                reference_frame = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        return len(self.loader)


# Per-worker RelativeActionLoader, built once in each pool process so we don't
# re-parse dataset metadata for every episode.
_WORKER_LOADER: "RelativeActionLoader | None" = None


def _init_stats_worker(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    embodiment_config: dict[str, ModalityConfig],
) -> None:
    global _WORKER_LOADER
    _WORKER_LOADER = RelativeActionLoader(
        dataset_path, embodiment_tag, group_key, embodiment_config=embodiment_config
    )


def _worker_load_relative_actions(episode_id: int) -> list[np.ndarray]:
    assert _WORKER_LOADER is not None
    return _WORKER_LOADER.load_relative_actions(episode_id)


def _resolve_num_workers(num_workers: int | None, num_items: int) -> int:
    """Number of processes to use for trajectory loading.

    Defaults to the CPU allocation visible to the process (respecting cgroup/affinity
    limits), overridable via GR00T_STATS_NUM_WORKERS. Capped at 16 to bound memory and
    IPC overhead, and never more than the number of episodes.
    """
    if num_workers is None:
        env = os.environ.get("GR00T_STATS_NUM_WORKERS")
        if env is not None:
            num_workers = int(env)
        else:
            try:
                available = len(os.sched_getaffinity(0))
            except AttributeError:  # not available on all platforms
                available = os.cpu_count() or 1
            num_workers = min(available, 16)
    return max(1, min(num_workers, num_items))


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
    num_workers: int | None = None,
) -> dict:
    # Resolve once in this process. Custom embodiments (e.g. NEW_EMBODIMENT) are registered
    # into MODALITY_CONFIGS at runtime, and that registration doesn't exist in the spawned
    # worker processes below -- so we pass the resolved dict down to them explicitly instead
    # of letting each worker redo the (process-local) registry lookup.
    embodiment_config = MODALITY_CONFIGS[embodiment_tag.value]
    loader = RelativeActionLoader(
        dataset_path, embodiment_tag, group_key, embodiment_config=embodiment_config
    )
    num_episodes = len(loader)
    if max_episodes != -1:
        num_episodes = min(num_episodes, max_episodes)
    episode_ids = list(range(num_episodes))

    resolved_workers = _resolve_num_workers(num_workers, num_episodes)
    trajectories = []
    desc = f"Loading trajectories for key {group_key}"
    if resolved_workers == 1:
        for episode_id in tqdm(episode_ids, desc=desc):
            trajectories.extend(loader.load_relative_actions(episode_id))
    else:
        # Loading is CPU-bound per-timestep pose math, so fan out across processes.
        # Trajectory order does not affect min/max/quantile/mean/std, and ProcessPoolExecutor.map
        # preserves input order anyway, so the result is identical to the serial path.
        # Use a spawn context: this may run after CUDA is initialized (e.g. inside the
        # dataset factory on rank 0), where forking a CUDA-aware process is unsafe.
        del loader  # each worker builds its own; avoid holding a second copy
        ctx = mp.get_context("spawn")
        chunksize = max(1, num_episodes // (resolved_workers * 8))
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=ctx,
            initializer=_init_stats_worker,
            initargs=(dataset_path, embodiment_tag, group_key, embodiment_config),
        ) as executor:
            for episode_trajectories in tqdm(
                executor.map(_worker_load_relative_actions, episode_ids, chunksize=chunksize),
                total=num_episodes,
                desc=desc,
            ):
                trajectories.extend(episode_trajectories)
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


def generate_rel_stats(dataset_path: Path | str, embodiment_tag: EmbodimentTag) -> None:
    dataset_path = Path(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    stats_path = Path(dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    for action_key in sorted(action_keys):
        if action_key in stats:
            continue
        print(f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}")
        stats[action_key] = calculate_stats_for_key(dataset_path, embodiment_tag, action_key)
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    modality_config_path: str | None = None,
):
    """Generate dataset statistics.

    Args:
        dataset_path: Path to the dataset.
        embodiment_tag: Embodiment tag for modality configurations.
        modality_config_path: Optional path to a .py modality config file. Required for custom
            embodiment tags not in the built-in MODALITY_CONFIGS registry.
    """
    if modality_config_path is not None:
        import importlib
        import sys

        config_path = Path(modality_config_path)
        if config_path.exists() and config_path.suffix == ".py":
            sys.path.append(str(config_path.parent))
            importlib.import_module(config_path.stem)
            print(f"Loaded modality config: {config_path}")
        else:
            raise FileNotFoundError(
                f"Modality config path does not exist or is not a .py file: {modality_config_path}"
            )
    generate_stats(dataset_path)
    generate_rel_stats(dataset_path, embodiment_tag)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
