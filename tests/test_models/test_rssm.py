"""Tests for RSSM and MambaWorldModel."""

import inspect

import pytest
import torch

from retail_world_model.models.rssm import RSSM
from retail_world_model.models.world_model import MambaWorldModel

# Use CUDA if available (Mamba2 requires CUDA tensors)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# d_model=256 ensures Mamba2's causal_conv1d strides are properly aligned
# in recurrent mode (smaller d_model causes stride alignment issues)
_D_MODEL = 256
_N_CAT = 8
_N_CLS = 8
_LATENT_DIM = _N_CAT * _N_CLS  # 64
_OBS_DIM = 32
_ACT_DIM = 4


class TestRSSM:
    @pytest.fixture
    def rssm(self):
        return RSSM(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
            d_model=_D_MODEL,
            n_cat=_N_CAT,
            n_cls=_N_CLS,
        ).to(_device)

    def test_no_h_in_posterior(self, rssm):
        """Posterior encode must NEVER receive h_t."""
        sig = inspect.signature(rssm.posterior.forward)
        params = list(sig.parameters.keys())
        assert "h_t" not in params

    def test_encode_obs_shapes(self, rssm):
        x = torch.randn(2, _OBS_DIM, device=_device)
        z, probs = rssm.encode_obs(x)
        assert z.shape == (2, _LATENT_DIM)
        assert probs.shape == (2, _N_CAT, _N_CLS)

    def test_train_sequence_shapes(self, rssm):
        B, T = 2, 10
        x = torch.randn(B, T, _OBS_DIM, device=_device)
        a = torch.randn(B, T, _ACT_DIM, device=_device)
        out = rssm.train_sequence(x, a)

        assert out["z_posterior_BT"].shape == (B, T, _LATENT_DIM)
        assert out["h_BT"].shape == (B, T, _D_MODEL)
        assert out["posterior_probs_BT"].shape == (B, T, _N_CAT, _N_CLS)
        assert out["prior_probs_BT"].shape == (B, T, _N_CAT, _N_CLS)
        assert out["reward_mean"].shape == (B, T)
        assert out["reward_std"].shape == (B, T)
        assert out["continue_logits"].shape == (B, T)
        assert "x_recon_BT" in out
        assert out["x_recon_BT"].shape == (B, T, _OBS_DIM)

    def test_imagine_step(self, rssm):
        B = 2
        z = torch.randn(B, _LATENT_DIM, device=_device)
        a = torch.randn(B, _ACT_DIM, device=_device)
        inference_params = rssm.backbone.init_inference_params(B)

        h, z_next, probs = rssm.imagine_step(z, a, inference_params)
        assert h.shape == (B, _D_MODEL)
        assert z_next.shape == (B, _LATENT_DIM)
        assert probs.shape == (B, _N_CAT, _N_CLS)

    def test_elbo_finite(self):
        """elbo_loss produces finite gradients."""
        B, T = 2, 10
        batch = {
            "x_BT": torch.randn(B, T, _OBS_DIM, device=_device),
            "a_BT": torch.randn(B, T, _ACT_DIM, device=_device),
            "r_BT": torch.randn(B, T, device=_device),
            "done_BT": torch.zeros(B, T, device=_device),
        }
        from retail_world_model.training.losses import elbo_loss

        model = MambaWorldModel(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
            d_model=_D_MODEL,
            n_cat=_N_CAT,
            n_cls=_N_CLS,
        ).to(_device)
        losses = elbo_loss(batch, model)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()


class TestMambaWorldModel:
    @pytest.fixture
    def model(self):
        return MambaWorldModel(
            obs_dim=_OBS_DIM,
            act_dim=_ACT_DIM,
            d_model=_D_MODEL,
            n_cat=_N_CAT,
            n_cls=_N_CLS,
        ).to(_device)

    def test_forward_shapes(self, model):
        B, T = 2, 10
        x = torch.randn(B, T, _OBS_DIM, device=_device)
        a = torch.randn(B, T, _ACT_DIM, device=_device)
        out = model(x, a)
        assert "z_posterior_BT" in out
        assert "h_BT" in out

    def test_imagine_shapes(self, model):
        B, H = 2, 5
        z0 = torch.randn(B, _LATENT_DIM, device=_device)
        actions = torch.randn(B, H, _ACT_DIM, device=_device)
        out = model.imagine(z0, actions)
        assert out["h_seq"].shape == (B, H, _D_MODEL)
        assert out["z_seq"].shape == (B, H, _LATENT_DIM)
        assert out["r_mean_seq"].shape == (B, H)
        assert out["r_std_seq"].shape == (B, H)

    def test_posterior_never_uses_h(self, model):
        """Verify DRAMA decoupling: posterior forward signature has no h_t."""
        sig = inspect.signature(model.rssm.posterior.forward)
        assert "h_t" not in sig.parameters

    def test_reset_state(self, model):
        state = model.reset_state(batch_size=2)
        assert "h" in state
        assert "z" in state
        assert state["h"].shape == (2, _D_MODEL)
        assert state["z"].shape == (2, _LATENT_DIM)

    def test_imagine_step_dict(self, model):
        B = 2
        model.reset_state(batch_size=B)
        z = torch.randn(B, _LATENT_DIM, device=_device)
        a = torch.randn(B, _ACT_DIM, device=_device)
        out = model.imagine_step(z, a)
        assert "h" in out
        assert "z" in out
        assert "r_mean" in out
        assert "r_std" in out
        assert "continue" in out
