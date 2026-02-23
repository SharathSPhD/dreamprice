"""Tests for training loss functions."""

from __future__ import annotations

import torch
import pytest

from retail_world_model.training.losses import (
    kl_balancing,
    twohot_ce_loss,
    continue_bce_loss,
    _categorical_kl,
    BETA_DYN,
    BETA_REP,
    FREE_BITS,
)
from retail_world_model.training.offline_utils import mopo_lcb, PercentileReturnNorm
from retail_world_model.inference.imagination import compute_lambda_returns
from retail_world_model.utils.twohot import make_bins


class TestKLBalancing:
    def test_kl_free_bits(self):
        """KL below free_bits threshold is clamped to free_bits."""
        B, T, n_cat, n_cls = 4, 8, 32, 32
        # Nearly identical distributions -> KL near 0 -> clamped to free_bits
        probs = torch.ones(B, T, n_cat, n_cls) / n_cls
        total, kl_dyn, kl_rep = kl_balancing(probs, probs, free_bits=1.0)
        # Each categorical's KL is 0, clamped to 1.0, so per-sample = n_cat * 1.0
        expected_per_cat = FREE_BITS
        assert kl_dyn.item() == pytest.approx(n_cat * expected_per_cat, rel=1e-4)

    def test_kl_balancing_asymmetry(self):
        """beta_dyn=0.5, beta_rep=0.1 — 5:1 asymmetry preserved."""
        B, T, n_cat, n_cls = 4, 8, 4, 4
        posterior = torch.randn(B, T, n_cat, n_cls).softmax(dim=-1)
        prior = torch.randn(B, T, n_cat, n_cls).softmax(dim=-1)

        total, kl_dyn, kl_rep = kl_balancing(
            posterior, prior, beta_dyn=0.5, beta_rep=0.1, free_bits=0.0
        )
        expected = 0.5 * kl_dyn + 0.1 * kl_rep
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_kl_nonzero_for_different_dists(self):
        """KL should be positive when distributions differ."""
        B, T, n_cat, n_cls = 2, 4, 4, 8
        posterior = torch.randn(B, T, n_cat, n_cls).softmax(dim=-1)
        prior = torch.randn(B, T, n_cat, n_cls).softmax(dim=-1)
        total, kl_dyn, kl_rep = kl_balancing(posterior, prior, free_bits=0.0)
        assert kl_dyn.item() > 0
        assert kl_rep.item() > 0


class TestMOPOLCB:
    def test_mopo_lcb(self):
        """r_pessimistic < r_mean when r_std > 0."""
        r_mean = torch.tensor([1.0, 2.0, 3.0])
        r_std = torch.tensor([0.5, 0.5, 0.5])
        r_pess = mopo_lcb(r_mean, r_std, lambda_lcb=1.0)
        assert (r_pess < r_mean).all()
        assert torch.allclose(r_pess, r_mean - r_std)

    def test_mopo_lcb_zero_std(self):
        """With zero std, pessimistic == mean."""
        r_mean = torch.tensor([1.0, 2.0])
        r_std = torch.zeros(2)
        r_pess = mopo_lcb(r_mean, r_std)
        assert torch.allclose(r_pess, r_mean)


class TestLambdaReturns:
    def test_lambda_returns_shape(self):
        """Output same shape as rewards."""
        B, H = 4, 13
        rewards = torch.randn(B, H)
        values = torch.randn(B, H + 1)
        continues = torch.ones(B, H)
        returns = compute_lambda_returns(rewards, values, continues)
        assert returns.shape == (B, H)

    def test_lambda_returns_no_discount(self):
        """With gamma=1, lambda=1, continues=1: returns = sum of future rewards + bootstrap."""
        B = 1
        H = 3
        rewards = torch.ones(B, H)
        values = torch.zeros(B, H + 1)
        continues = torch.ones(B, H)
        returns = compute_lambda_returns(rewards, values, continues, gamma=1.0, lambda_=1.0)
        # R_2 = 1 + 0 = 1, R_1 = 1 + 1 = 2, R_0 = 1 + 2 = 3
        assert returns[0, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert returns[0, 1].item() == pytest.approx(2.0, abs=1e-5)
        assert returns[0, 2].item() == pytest.approx(1.0, abs=1e-5)


class TestTwohotCELoss:
    def test_twohot_ce_finite(self):
        """Loss should be finite for valid inputs."""
        logits = torch.randn(4, 8, 255)
        targets = torch.randn(4, 8)
        bins = make_bins()
        loss = twohot_ce_loss(logits, targets, bins)
        assert torch.isfinite(loss)
        assert loss.item() > 0


class TestContinueBCE:
    def test_continue_bce_finite(self):
        logits = torch.randn(4, 8)
        dones = torch.zeros(4, 8)
        loss = continue_bce_loss(logits, dones)
        assert torch.isfinite(loss)


class TestPercentileReturnNorm:
    def test_normalize_identity_for_small_range(self):
        """When P95-P5 < 1, returns pass through unscaled (max(1, range))."""
        norm = PercentileReturnNorm()
        returns = torch.zeros(100)  # all same -> P95-P5 = 0
        norm.update(returns)
        result = norm.normalize(returns)
        assert torch.allclose(result, returns)

    def test_normalize_compresses(self):
        """When range > 1, returns are compressed."""
        norm = PercentileReturnNorm()
        returns = torch.linspace(-10, 10, 1000)
        norm.update(returns)
        result = norm.normalize(returns)
        assert result.abs().max() < returns.abs().max()
