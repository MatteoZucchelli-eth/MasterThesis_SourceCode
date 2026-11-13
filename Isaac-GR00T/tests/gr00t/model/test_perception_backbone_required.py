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


"""The perception finetune must refuse to start on bare Cosmos-Reason2-2B weights.

The backbone ARCHITECTURE is always Cosmos-Reason2-2B -- GR00T N1.7's VLM is that same
2B model, robot-finetuned -- so omitting the checkpoint never produced a "from scratch"
run, it produced a normal-looking run on un-adapted weights with nothing in the logs to
say so. That silence is the whole reason this is an error rather than a default.
"""

from types import SimpleNamespace

from gr00t.configs.finetune_config import GR00T_VLM_CHECKPOINT, PerceptionFinetuneConfig
from gr00t.model.gr00t_n1d7.setup_perception import Gr00tN1d7PerceptionPipeline
import pytest


class _Stub(Gr00tN1d7PerceptionPipeline):
    """Exercise _create_model's guard without building a 2B backbone."""

    def __init__(self, checkpoint, skip_weight_loading=False):
        self.config = SimpleNamespace(
            training=SimpleNamespace(
                start_from_checkpoint=checkpoint,
                skip_weight_loading=skip_weight_loading,
            ),
            model=SimpleNamespace(),
        )
        self.transformers_loading_kwargs = {}

    model_class = staticmethod(
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("model construction reached: the guard did not fire first")
        )
    )


class TestBackboneCheckpointRequired:
    @pytest.mark.parametrize("checkpoint", [None, ""])
    def test_missing_checkpoint_raises(self, checkpoint):
        with pytest.raises(ValueError, match="GR00T-lineage"):
            _Stub(checkpoint)._create_model()

    @pytest.mark.parametrize("sentinel", ["scratch", "none", "fresh", "Scratch", " NONE "])
    def test_retired_sentinels_raise_rather_than_becoming_a_path(self, sentinel):
        """These used to select bare Cosmos. Passed through they would reach the loader
        as a literal directory name and die with an unhelpful "no safetensors in
        scratch" long after the run started."""
        with pytest.raises(ValueError, match="GR00T-lineage"):
            _Stub(sentinel)._create_model()

    def test_error_names_the_flag_and_the_escape_hatch(self):
        with pytest.raises(ValueError) as exc:
            _Stub(None)._create_model()
        message = str(exc.value)
        assert "--base-model-path" in message
        assert "--skip-weight-loading" in message
        assert "Cosmos-Reason2-2B" in message

    def test_smoke_tests_can_still_skip_weight_loading(self):
        """--skip-weight-loading is architecture-only CI, not a way back to bare Cosmos."""
        with pytest.raises(AssertionError, match="guard did not fire first"):
            _Stub(None, skip_weight_loading=True)._create_model()


class TestPerceptionConfigDefaults:
    def test_base_model_path_defaults_to_the_groot_release(self):
        assert PerceptionFinetuneConfig().base_model_path == GR00T_VLM_CHECKPOINT
        assert GR00T_VLM_CHECKPOINT == "nvidia/GR00T-N1.7-3B"

    def test_base_model_path_is_no_longer_optional(self):
        """A None default was what let the bare-Cosmos path exist in the first place."""
        annotation = PerceptionFinetuneConfig.__annotations__["base_model_path"]
        assert "None" not in str(annotation)
