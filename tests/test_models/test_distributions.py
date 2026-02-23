"""Tests for distribution utilities."""

import torch
import pytest

from retail_world_model.utils.distributions import (
    symlog,
    symexp,
    apply_unimix,
    sample_straight_through,
    twohot_encode,
    twohot_decode,
)


class TestSymlogSymexp:
    def test_inverse_positive(self):
        x = torch.tensor([0.0, 1.0, 10.0, 100.0, 10000.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)

    def test_inverse_negative(self):
        x = torch.tensor([-1.0, -10.0, -100.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)

    def test_inverse_zero(self):
        x = torch.tensor([0.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-7)

    def test_symlog_is_symmetric(self):
        x = torch.tensor([5.0])
        assert torch.allclose(symlog(-x), -symlog(x))

    def test_symlog_defined_at_zero(self):
        assert symlog(torch.tensor(0.0)).item() == 0.0


class TestUnimix:
    def test_sums_to_one(self):
        logits = torch.randn(4, 32)
        probs = apply_unimix(logits, mix=0.01)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_minimum_probability(self):
        # With mix=0.01 and 32 classes, min prob >= 0.01/32
        logits = torch.randn(4, 32)
        logits[:, 0] = -1000.0  # Make one class very unlikely
        probs = apply_unimix(logits, mix=0.01)
        min_prob = 0.01 / 32
        assert (probs >= min_prob - 1e-7).all()

    def test_no_mix_recovers_softmax(self):
        logits = torch.randn(4, 32)
        probs = apply_unimix(logits, mix=0.0)
        expected = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs, expected, atol=1e-6)


class TestStraightThrough:
    def test_gradient_flows(self):
        logits = torch.randn(4, 32, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        z = sample_straight_through(probs)
        loss = z.sum()
        loss.backward()
        assert logits.grad is not None
        assert (logits.grad != 0).any()

    def test_forward_is_onehot(self):
        probs = torch.softmax(torch.randn(4, 32), dim=-1)
        z = sample_straight_through(probs)
        # Each row should have exactly one 1.0 (from one_hot) plus small residual
        # But in forward pass (detaching probs), it's exactly one_hot
        # The straight-through trick: forward = one_hot, backward = probs
        # So z.detach() should be one-hot
        z_det = z.detach()
        row_sums = z_det.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


class TestTwohot:
    def test_encode_decode_roundtrip(self):
        bins = torch.linspace(-20, 20, 255)
        x = torch.tensor([0.0, 1.5, -3.0, 10.0, -15.0])
        x_symlog = symlog(x)
        encoded = twohot_encode(x_symlog, bins)
        decoded = twohot_decode(encoded, bins)
        # twohot_decode applies symexp, so compare to original x
        assert torch.allclose(decoded, x, rtol=0.05, atol=0.1)

    def test_encode_sums_to_one(self):
        bins = torch.linspace(-20, 20, 255)
        x = torch.randn(10)
        encoded = twohot_encode(x, bins)
        sums = encoded.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_encode_shape(self):
        bins = torch.linspace(-20, 20, 255)
        x = torch.randn(4, 8)
        encoded = twohot_encode(x, bins)
        assert encoded.shape == (4, 8, 255)
