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


"""The visibility BCE is class-weighted: the real captures are ~94:6 visible:not.

visibility_neg_weight scales the target_visible=0 rows. The loss is normalized by the
weight sum rather than mean-reduced, so visibility_loss_weight keeps its meaning as the
negative weight moves, and neg_weight=1 is exactly the old unweighted mean.
"""

from gr00t.configs.model.gr00t_n1d7_perception import Gr00tN1d7PerceptionConfig
from gr00t.model.gr00t_n1d7.perception import perception_losses
import torch


MERGE = 2
H = 64
GH = GW = H // MERGE
N_PER_VIEW = GH * GW


class TestVisibilityClassWeight:
    """visibility_neg_weight rebalances a split that is ~94:6 in the real captures."""

    @staticmethod
    def _call(neg_weight: float):
        config = Gr00tN1d7PerceptionConfig(
            image_target_size=(64, 64),
            depth_loss_weight=0.0,
            wall_normal_loss_weight=0.0,
            visibility_neg_weight=neg_weight,
        )
        batch = 4
        head_out = {
            "heat_logits": torch.zeros(batch, N_PER_VIEW),
            "offsets": torch.zeros(batch, N_PER_VIEW, 2),
            # confidently "visible" everywhere: only the negatives are penalised
            "vis_logit": torch.full((batch,), 4.0),
        }
        inputs = {
            "target_uv": torch.full((batch, 2), 32.0),
            "target_visible": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        }
        return perception_losses(
            config, head_out, inputs, GH, GW, use_depth=False, use_normal=False
        )["visibility_loss"]

    def test_default_reproduces_the_unweighted_mean(self):
        # 3 positives at BCE(4, 1), 1 negative at BCE(4, 0), plain mean.
        expected = (
            3 * torch.nn.functional.softplus(torch.tensor(-4.0))
            + torch.nn.functional.softplus(torch.tensor(4.0))
        ) / 4
        assert torch.allclose(self._call(1.0), expected, atol=1e-6)

    def test_upweighting_negatives_raises_the_loss(self):
        assert self._call(10.0) > self._call(1.0)

    def test_stays_normalized_not_summed(self):
        # Weight-normalized: with every row a negative-weight of 1 the value is a
        # mean, so a large weight cannot make the term blow past the worst per-row loss.
        worst = torch.nn.functional.softplus(torch.tensor(4.0))
        assert self._call(100.0) <= worst + 1e-6
