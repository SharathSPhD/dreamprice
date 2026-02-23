"""Tests for RSSM and MambaWorldModel."""

import inspect

import torch
import pytest

from retail_world_model.models.rssm import RSSM
from retail_world_model.models.world_model import MambaWorldModel


class TestRSSM:
    @pytest.fixture
    def rssm(self):
        return RSSM(obs_dim=32, act_dim=4, d_model=64, n_cat=8, n_cls=8)

    def test_no_h_in_posterior(self, rssm):
        """Posterior encode must NEVER receive h_t."""
        sig = inspect.signature(rssm.posterior.forward)
        params = list(sig.parameters.keys())
        assert "h_t" not in params

    def test_encode_obs_shapes(self, rssm):
        x = torch.randn(2, 32)
        z, probs = rssm.encode_obs(x)
        assert z.shape == (2, 64)   # n_cat * n_cls = 8 * 8
        assert probs.shape == (2, 8, 8)

    def test_train_sequence_shapes(self, rssm):
        B, T = 2, 10
        x = torch.randn(B, T, 32)
        a = torch.randn(B, T, 4)
        out = rssm.train_sequence(x, a)

        assert out["z_BT"].shape == (B, T, 64)
        assert out["h_BT"].shape == (B, T, 64)
        assert out["posterior_probs"].shape == (B, T, 8, 8)
        assert out["prior_probs"].shape == (B, T, 8, 8)
        assert out["reward_mean"].shape == (B, T)
        assert out["reward_std"].shape == (B, T)
        assert out["continue_logits"].shape == (B, T)

    def test_imagine_step(self, rssm):
        B = 2
        z = torch.randn(B, 64)
        a = torch.randn(B, 4)
        inference_params = rssm.backbone.init_inference_params(B)

        h, z_next, probs = rssm.imagine_step(z, a, inference_params)
        assert h.shape == (B, 64)
        assert z_next.shape == (B, 64)
        assert probs.shape == (B, 8, 8)


class TestMambaWorldModel:
    @pytest.fixture
    def model(self):
        return MambaWorldModel(obs_dim=32, act_dim=4, d_model=64, n_cat=8, n_cls=8)

    def test_forward_shapes(self, model):
        B, T = 2, 10
        x = torch.randn(B, T, 32)
        a = torch.randn(B, T, 4)
        out = model(x, a)
        assert "z_BT" in out
        assert "h_BT" in out

    def test_imagine_shapes(self, model):
        B, H = 2, 5
        z0 = torch.randn(B, 64)
        actions = torch.randn(B, H, 4)
        out = model.imagine(z0, actions)
        assert out["h_seq"].shape == (B, H, 64)
        assert out["z_seq"].shape == (B, H, 64)
        assert out["r_mean_seq"].shape == (B, H)
        assert out["r_std_seq"].shape == (B, H)

    def test_posterior_never_uses_h(self, model):
        """Verify DRAMA decoupling: posterior forward signature has no h_t."""
        sig = inspect.signature(model.rssm.posterior.forward)
        assert "h_t" not in sig.parameters
