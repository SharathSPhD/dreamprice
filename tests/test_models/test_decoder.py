"""Tests for CausalDemandDecoder — theta must stay FROZEN."""

import pytest
import torch

from retail_world_model.models.decoder import CausalDemandDecoder


class TestCausalDemandDecoder:
    @pytest.fixture
    def decoder(self):
        return CausalDemandDecoder(
            latent_dim=64,
            n_categories=3,
            n_store_features=8,
        )

    def test_theta_frozen_after_optimizer_step(self, decoder):
        """KEY TEST: theta must remain frozen after an optimizer step."""
        opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        theta_before = decoder.theta.weight.data.clone()

        z = torch.randn(4, 64)
        log_price = torch.randn(4)
        cat_id = torch.randint(0, 3, (4,))
        store_feat = torch.randn(4, 8)

        loss = decoder(z, log_price, cat_id, store_feat).mean()
        loss.backward()
        opt.step()

        assert torch.allclose(decoder.theta.weight.data, theta_before), (
            "theta must ALWAYS be frozen (requires_grad=False)"
        )

    def test_theta_requires_grad_false(self, decoder):
        assert not decoder.theta.weight.requires_grad

    def test_theta_default_values(self, decoder):
        """Default theta should be -2.5 (grocery prior mean)."""
        expected = torch.tensor([[-2.5], [-2.5], [-2.5]])
        assert torch.allclose(decoder.theta.weight.data, expected)

    def test_output_shape(self, decoder):
        z = torch.randn(4, 64)
        log_price = torch.randn(4)
        cat_id = torch.randint(0, 3, (4,))
        store_feat = torch.randn(4, 8)
        out = decoder(z, log_price, cat_id, store_feat)
        assert out.shape == (4,)

    def test_causal_effect_direction(self, decoder):
        """Higher price -> lower demand (negative elasticity)."""
        z = torch.zeros(2, 64)
        store_feat = torch.zeros(2, 8)
        cat_id = torch.zeros(2, dtype=torch.long)

        low_price = torch.tensor([0.5, 0.5])
        high_price = torch.tensor([2.0, 2.0])

        demand_low = decoder(z, low_price, cat_id, store_feat)
        demand_high = decoder(z, high_price, cat_id, store_feat)

        # With theta = -2.5, higher log_price should produce lower demand
        assert (demand_high < demand_low).all(), (
            "Negative elasticity means higher price -> lower demand"
        )
