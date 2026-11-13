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

"""Inference server for the perception-only model (Gr00tN1d7Perception).

Mirrors run_gr00t_server.py, but serves a PerceptionPolicy. The wire protocol is
identical — it registers the standard `get_action` endpoint — so the existing
gr00t.policy.server_client.PolicyClient queries it unchanged; the returned
"action" dict carries the target-perception outputs (see PerceptionPolicy).

Usage:
    python gr00t/eval/run_perception_server.py \
        --model-path <perception-checkpoint-dir> \
        --embodiment-tag new_embodiment \
        --port 5555
"""

from dataclasses import dataclass
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.perception_policy import PerceptionPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class PerceptionServerConfig:
    """Configuration for running the GR00T perception inference server."""

    model_path: str
    """Path to the perception model checkpoint directory."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive)."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input validation."""

    visibility_threshold: float = 0.5
    """Sigmoid threshold above which the target is reported `visible`."""


def main(config: PerceptionServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T PERCEPTION inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path:     {config.model_path}")
    print(f"  Device:         {config.device}")
    print(f"  Host:           {config.host}")
    print(f"  Port:           {config.port}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = PerceptionPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        visibility_threshold=config.visibility_threshold,
    )

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    print(f"\n✓ Perception server ready — listening on {config.host}:{config.port}\n")
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(PerceptionServerConfig))
