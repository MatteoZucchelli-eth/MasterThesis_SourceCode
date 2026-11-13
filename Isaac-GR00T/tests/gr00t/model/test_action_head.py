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
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_head_does_not_apply_state_dropout(self):
        # State dropout is owned by the processor (raw states); the head must not
        # re-apply it to the encoded features (double dropout). With prob=1.0 the
        # old head-side dropout would zero every state feature entering the DiT.
        config = _small_config(state_dropout_prob=1.0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        out = head.forward(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["loss"])
        assert out["state_features"].abs().sum() > 0


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad


class TestTipTargetHead:
    """Test the optional tip-target auxiliary head (xyz + rot6d regression)."""

    @staticmethod
    def _tip_config(**overrides):
        defaults = dict(use_tip_target_head=True, tip_target_loss_weight=0.5)
        defaults.update(overrides)
        return _small_config(**defaults)

    @staticmethod
    def _tip_target(batch_size=2):
        target = torch.randn(batch_size, 9) * 0.1
        target[:, 3:] = torch.nn.functional.normalize(target[:, 3:], dim=-1)
        return target

    def test_rot6d_orthonormalize(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import rot6d_orthonormalize

        r = rot6d_orthonormalize(torch.randn(4, 6))
        a, b = r[..., :3], r[..., 3:]
        assert torch.allclose(a.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(b.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose((a * b).sum(-1), torch.zeros(4), atol=1e-5)

    def test_forward_adds_tip_loss(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = self._tip_target()
        out = head.forward(_make_backbone_output(config), action_input)
        assert "tip_target_loss" in out
        assert torch.isfinite(out["tip_target_loss"])
        assert torch.isfinite(out["loss"])
        # Total loss = action loss + weight * tip loss.
        expected = out["action_loss_scalar"] + 0.5 * out["tip_target_loss"]
        assert torch.allclose(out["loss"], expected, atol=1e-5)
        # Conditioning appends one token to the single state token.
        assert out["state_features"].shape[1] == 2

    def test_forward_masks_nan_rows(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        target = self._tip_target()
        target[0] = float("nan")  # unresolved label on one row
        action_input["tip_target_9d"] = target
        out = head.forward(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["tip_target_loss"])
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        for p in head.tip_target_head.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_forward_all_nan_skips_tip_loss(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = torch.full((2, 9), float("nan"))
        out = head.forward(_make_backbone_output(config), action_input)
        assert "tip_target_loss" not in out
        assert torch.isfinite(out["loss"])

    def test_forward_without_target_conditions_on_prediction(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "tip_target_loss" not in out
        assert "tip_target_pred" in out
        assert torch.isfinite(out["loss"])

    def test_get_action_returns_orthonormal_prediction(self):
        config = self._tip_config()
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        pred = out["tip_target_pred"]
        assert pred.shape == (2, 9)
        a, b = pred[..., 3:6], pred[..., 6:]
        assert torch.allclose(a.norm(dim=-1), torch.ones(2), atol=1e-4)
        assert torch.allclose((a * b).sum(-1), torch.zeros(2), atol=1e-4)

    def test_no_conditioning_keeps_single_state_token(self):
        config = self._tip_config(use_tip_target_conditioning=False)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        action_input["tip_target_9d"] = self._tip_target()
        out = head.forward(_make_backbone_output(config), action_input)
        assert head.tip_target_head.cond_embed is None
        assert out["state_features"].shape[1] == 1
        assert "tip_target_loss" in out

    def test_xyz_scaling_roundtrip(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead

        head = TipTargetHead(
            latent_dim=8, hidden_dim=8, cond_dim=8, use_conditioning=True, xyz_scale=0.1
        )
        raw = self._tip_target()
        assert torch.allclose(head.unscale(head.scale(raw)), raw, atol=1e-6)
        assert torch.allclose(head.scale(raw)[..., :3], raw[..., :3] / 0.1)


class TestTipTargetCameraFrameLoss:
    """The position loss is split lateral-vs-depth in the CAMERA frame.

    Tip-frame `z` is not depth: the optical axis sits at [0.9037, -0.0012, -0.4281] in
    the tip frame (Kabsch over the heatmap sidecar, rigid, rms 1.18 mm), so depth is
    0.9037*x - 0.4281*z. These tests pin the two properties that make the split safe to
    land: it is a no-op at the default weights, and it actually isolates depth.
    """

    @staticmethod
    def _config(**overrides):
        defaults = dict(use_tip_target_head=True, tip_target_loss_weight=0.5)
        defaults.update(overrides)
        return _small_config(**defaults)

    def _run(self, config, seed=0):
        torch.manual_seed(seed)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(config)
        target = torch.randn(2, 9) * 0.1
        target[:, 3:] = torch.nn.functional.normalize(target[:, 3:], dim=-1)
        action_input["tip_target_9d"] = target
        torch.manual_seed(seed + 1000)  # same noise draw for both arms
        return head.forward(_make_backbone_output(config), action_input)

    def test_default_weights_reproduce_the_plain_xyz_mse(self):
        """2/3 lateral + 1/3 depth == the single xyz MSE it replaced.

        A rotation preserves the L2 norm, so an equally-weighted split of the rotated
        residual is numerically the same number. This is what lets the split land as
        instrumentation rather than as a silent change of objective.
        """
        out = self._run(self._config())
        # Reconstruct the old objective from the per-part terms: mean over 3 xyz dims.
        old_xyz_mse = (2.0 * out["tip_target_lat_loss"] + out["tip_target_depth_loss"]) / 3.0
        assert torch.allclose(out["tip_target_pos_loss"], old_xyz_mse, atol=1e-6)

    def test_reweighting_depth_changes_the_loss(self):
        base = self._run(self._config())
        heavy = self._run(self._config(tip_target_depth_loss_weight=3.0))
        assert not torch.allclose(base["tip_target_pos_loss"], heavy["tip_target_pos_loss"])
        # Only the depth term is rescaled; the lateral one is untouched.
        assert torch.allclose(base["tip_target_lat_loss"], heavy["tip_target_lat_loss"], atol=1e-6)

    def test_millimetre_metrics_are_reported(self):
        out = self._run(self._config())
        for key in ("tip_target_lat_err_mm", "tip_target_depth_err_mm", "tip_target_pos_err_mm"):
            assert key in out and torch.isfinite(out[key])
        # The full position error is at least as large as either component.
        assert out["tip_target_pos_err_mm"] >= out["tip_target_depth_err_mm"] - 1e-4

    def test_depth_axis_is_not_tip_z(self):
        """A pure tip-frame +z error must land mostly in LATERAL, not in depth.

        The optical axis is only -0.4281 along tip z, so a naive x,y-vs-z split would
        call this a depth error when 90% of it is laterally observable.
        """
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead as _AH

        head = _AH(self._config())
        pred = torch.tensor([[0.0, 0.0, 1.0] + [0.0] * 6])
        tgt = torch.zeros(1, 9)
        resid = head._tip_target_pos_residual_cam(pred, tgt)
        lateral = resid[0, :2].norm().item()
        depth = resid[0, 2].abs().item()
        assert depth == pytest.approx(0.4281 * 0.1, abs=1e-4)
        assert lateral > depth  # most of a tip-z error is laterally observable


class TestTipTargetWristMasking:
    """Hiding the wrist view from the auxiliary head only."""

    def test_view_token_mask_picks_alternating_runs(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import _view_token_mask

        # Four image runs separated by a non-image token: ego, wrist, ego, wrist —
        # the 2-timestep case (register_new_modality.py's [-1, 0]). The single-frame
        # case that register_autocollect_modality.py now defaults to is the first two
        # runs of this, and falls out of the same modulo.
        img = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]], dtype=torch.bool)
        wrist = _view_token_mask(img, num_views=2, view_index=1)
        expected = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)
        assert torch.equal(wrist, expected)
        ego = _view_token_mask(img, num_views=2, view_index=0)
        assert torch.equal(ego | wrist, img)
        assert not (ego & wrist).any()

    def test_single_view_masks_nothing(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import _view_token_mask

        img = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
        assert not _view_token_mask(img, num_views=1, view_index=0).any()

    def test_hidden_tokens_do_not_reach_the_head(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead

        torch.manual_seed(0)
        head = TipTargetHead(
            latent_dim=8, hidden_dim=8, cond_dim=8, use_conditioning=False, xyz_scale=0.1
        )
        head.eval()
        embeds = torch.randn(1, 6, 8)
        mask = torch.ones(1, 6, dtype=torch.bool)
        hide = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.bool)

        with torch.no_grad():
            base = head.predict_scaled(embeds, mask, hide_tokens=hide)
            # Scribbling over the hidden positions must not move the prediction.
            scribbled = embeds.clone()
            scribbled[:, 3:5] = torch.randn(1, 2, 8) * 100
            other = head.predict_scaled(scribbled, mask, hide_tokens=hide)
        assert torch.allclose(base, other, atol=1e-5)

    def test_masking_is_training_only(self):
        config = _small_config(
            use_tip_target_head=True,
            tip_target_loss_weight=0.5,
            tip_target_wrist_mask_prob=1.0,
        )
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        # eval() must not consult image_mask at all; this would raise if it did.
        out = head.get_action(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["tip_target_pred"]).all()


class TestTipTargetPoolResidual:
    """`attn_residual` must start exactly at the mean-pool solution."""

    def test_zero_init_makes_every_query_the_mean_vector(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead, _masked_mean_pool

        torch.manual_seed(0)
        head = TipTargetHead(
            latent_dim=8,
            hidden_dim=8,
            cond_dim=8,
            use_conditioning=False,
            xyz_scale=0.1,
            num_queries=3,
            num_attn_heads=2,
        )
        head.eval()
        embeds = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            queries = head.pool_queries.unsqueeze(0).expand(2, -1, -1)
            pooled, _ = head.pool_attn(queries, embeds, embeds, need_weights=False)
            assert torch.allclose(pooled, torch.zeros_like(pooled), atol=1e-6)
            pooled = pooled + _masked_mean_pool(embeds, mask).unsqueeze(1)
            normed = head.pool_norm(pooled)
        # Every query slot carries the same thing: the mean-pooled vector.
        assert torch.allclose(normed[:, 0], normed[:, 1], atol=1e-6)
        assert torch.allclose(normed[:, 0], normed[:, 2], atol=1e-6)

    def test_bare_attn_pool_is_gone(self):
        """The no-residual pool was removed, not kept as an option."""
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead

        with pytest.raises(ValueError, match="tip_target_pool"):
            TipTargetHead(
                latent_dim=8,
                hidden_dim=8,
                cond_dim=8,
                use_conditioning=False,
                xyz_scale=0.1,
                pool="attn",
            )

    def test_rejects_unknown_pool(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import TipTargetHead

        with pytest.raises(ValueError, match="tip_target_pool"):
            TipTargetHead(
                latent_dim=8,
                hidden_dim=8,
                cond_dim=8,
                use_conditioning=False,
                xyz_scale=0.1,
                pool="bogus",
            )


class TestTipTargetTeacherForcingSchedule:
    """Scheduled sampling: hand the conditioning token over once the head is good."""

    @staticmethod
    def _head(**overrides):
        defaults = dict(
            use_tip_target_head=True,
            tip_target_loss_weight=0.5,
            tip_target_tf_gate_mm=10.0,
            tip_target_tf_gate_patience=3,
            tip_target_tf_ramp_steps=10,
            tip_target_tf_gate_ema=0.0,  # EMA follows the input exactly, for determinism
        )
        defaults.update(overrides)
        return Gr00tN1d7ActionHead(_small_config(**defaults))

    def test_gate_stays_shut_above_the_threshold(self):
        head = self._head()
        for _ in range(20):
            assert head._tip_target_pred_prob(torch.tensor(50.0)) == 0.0
        assert head.tip_target_tf_open_step.item() == -1

    def test_gate_needs_the_patience_window(self):
        head = self._head()
        assert head._tip_target_pred_prob(torch.tensor(1.0)) == 0.0  # 1 of 3
        assert head._tip_target_pred_prob(torch.tensor(1.0)) == 0.0  # 2 of 3
        head._tip_target_pred_prob(torch.tensor(1.0))  # 3 of 3 -> opens
        assert head.tip_target_tf_open_step.item() == 3

    def test_a_bad_step_resets_the_window(self):
        head = self._head()
        head._tip_target_pred_prob(torch.tensor(1.0))
        head._tip_target_pred_prob(torch.tensor(1.0))
        head._tip_target_pred_prob(torch.tensor(99.0))  # resets the hold counter
        head._tip_target_pred_prob(torch.tensor(1.0))
        head._tip_target_pred_prob(torch.tensor(1.0))
        assert head.tip_target_tf_open_step.item() == -1

    def test_probability_ramps_and_saturates(self):
        head = self._head()
        for _ in range(3):
            head._tip_target_pred_prob(torch.tensor(1.0))
        probs = [head._tip_target_pred_prob(torch.tensor(1.0)) for _ in range(12)]
        assert probs[0] == pytest.approx(0.1)
        assert probs == sorted(probs)
        assert probs[-1] == pytest.approx(1.0)

    def test_gate_is_sticky(self):
        head = self._head()
        for _ in range(3):
            head._tip_target_pred_prob(torch.tensor(1.0))
        opened = head.tip_target_tf_open_step.item()
        for _ in range(5):
            head._tip_target_pred_prob(torch.tensor(999.0))  # error blows up again
        assert head.tip_target_tf_open_step.item() == opened
        assert head._tip_target_pred_prob(torch.tensor(999.0)) > 0.0

    def test_schedule_survives_a_checkpoint_round_trip(self):
        """A time-limited Euler job must resume the schedule, not restart it."""
        head = self._head()
        for _ in range(6):
            head._tip_target_pred_prob(torch.tensor(1.0))
        state = {k: v.clone() for k, v in head.state_dict().items()}
        assert "tip_target_tf_open_step" in state

        resumed = self._head()
        resumed.load_state_dict(state)
        assert resumed.tip_target_tf_open_step.item() == head.tip_target_tf_open_step.item()
        assert resumed.tip_target_tf_step.item() == head.tip_target_tf_step.item()

    def test_reset_restores_a_fresh_schedule(self):
        """A base checkpoint has no tf_* buffers; the reset must undo the garbage."""
        head = self._head()
        for _ in range(6):
            head._tip_target_pred_prob(torch.tensor(1.0))
        assert head.tip_target_tf_open_step.item() >= 0

        # Stand in for what transformers leaves behind for missing buffers.
        with torch.no_grad():
            head.tip_target_tf_step.fill_(123456)
            head.tip_target_tf_hold.fill_(99)
            head.tip_target_tf_err_ema.fill_(0.0)
        head.reset_tip_target_tf_state()

        assert head.tip_target_tf_step.item() == 0
        assert head.tip_target_tf_hold.item() == 0
        assert head.tip_target_tf_open_step.item() == -1
        assert torch.isnan(head.tip_target_tf_err_ema)
        # And the gate takes the full patience window again.
        assert head._tip_target_pred_prob(torch.tensor(1.0)) == 0.0
        assert head.tip_target_tf_open_step.item() == -1

    def test_reset_is_a_no_op_without_the_head(self):
        head = Gr00tN1d7ActionHead(_small_config(use_tip_target_head=False))
        head.reset_tip_target_tf_state()

    def test_disabled_by_default(self):
        head = Gr00tN1d7ActionHead(
            _small_config(use_tip_target_head=True, tip_target_loss_weight=0.5)
        )
        assert head.tip_target_tf_gate_mm is None
        assert head._tip_target_pred_prob(torch.tensor(0.0)) == 0.0

    def test_eval_reads_the_schedule_without_advancing_it(self):
        head = self._head()
        for _ in range(4):
            head._tip_target_pred_prob(torch.tensor(1.0))
        step = head.tip_target_tf_step.item()
        assert head._tip_target_pred_prob_readonly() == pytest.approx(0.1)
        assert head.tip_target_tf_step.item() == step
