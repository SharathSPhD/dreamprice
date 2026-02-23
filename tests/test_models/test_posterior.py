"""Tests for the DecoupledPosterior — DRAMA-style, z_t depends ONLY on x_t."""

import inspect

import pytest
import torch

from retail_world_model.models.posterior import DecoupledPosterior


class TestDecoupledPosterior:
    @pytest.fixture
    def posterior(self):
        return DecoupledPosterior(obs_dim=64, d_model=128, n_cat=8, n_cls=8)

    def test_input_is_only_xt(self, posterior):
        """KEY TEST: posterior.forward() must NOT accept h_t as a parameter."""
        sig = inspect.signature(posterior.forward)
        params = list(sig.parameters.keys())
        assert "h_t" not in params, "Posterior must NEVER receive h_t (DRAMA decoupling)"
        # Only self and x_t
        assert params == ["x_t"], f"Expected only ['x_t'], got {params}"

    def test_output_shapes_2d(self, posterior):
        """Test shapes for (B, obs_dim) input."""
        x = torch.randn(4, 64)
        z_flat, probs = posterior(x)
        assert z_flat.shape == (4, 64)  # n_cat * n_cls = 8 * 8 = 64
        assert probs.shape == (4, 8, 8)

    def test_output_shapes_3d(self, posterior):
        """Test shapes for (B, T, obs_dim) input."""
        x = torch.randn(4, 10, 64)
        z_flat, probs = posterior(x)
        assert z_flat.shape == (4, 10, 64)
        assert probs.shape == (4, 10, 8, 8)

    def test_probs_sum_to_one(self, posterior):
        x = torch.randn(4, 64)
        _, probs = posterior(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows_through_straight_through(self, posterior):
        x = torch.randn(4, 64, requires_grad=True)
        z_flat, probs = posterior(x)
        loss = z_flat.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()
