"""Distribution utilities: symlog/symexp, unimix, straight-through, twohot encoding."""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(|x| + 1). Defined at zero, symmetric, differentiable everywhere."""
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return x.sign() * (x.abs().exp() - 1)


def apply_unimix(logits: torch.Tensor, mix: float = 0.01) -> torch.Tensor:
    """probs = (1 - mix) * softmax(logits) + mix / n_cls.

    Prevents near-deterministic distributions that destabilize KL computation.
    """
    probs = F.softmax(logits, dim=-1)
    n_cls = logits.shape[-1]
    return (1 - mix) * probs + mix / n_cls


def sample_straight_through(probs: torch.Tensor) -> torch.Tensor:
    """one_hot(argmax) + probs - probs.detach().

    Forward pass: discrete one-hot. Backward pass: gradients flow through probs.
    """
    indices = probs.argmax(dim=-1)
    one_hot = F.one_hot(indices, num_classes=probs.shape[-1]).float()
    return one_hot + probs - probs.detach()


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Soft two-hot labels for distributional prediction.

    For each value in x, puts mass on the two adjacent bins.

    Args:
        x: arbitrary shape, scalar values in symlog space.
        bins: (n_bins,) sorted ascending bin centers.

    Returns:
        Tensor of shape (*x.shape, n_bins) with soft labels summing to 1.
    """
    x = x.unsqueeze(-1)  # (..., 1)
    bins = bins.to(x.device)

    # Clamp to bin range
    x_clamped = x.clamp(bins[0], bins[-1])

    # Find the lower bin index for each value
    # k such that bins[k] <= x < bins[k+1]
    k = (x_clamped >= bins.unsqueeze(0)).sum(dim=-1) - 1  # (...)
    k = k.clamp(0, len(bins) - 2)

    # Compute weights
    b_low = bins[k]       # (...,)
    b_high = bins[k + 1]  # (...,)
    x_clamped = x_clamped.squeeze(-1)

    w_high = (x_clamped - b_low) / (b_high - b_low + 1e-8)
    w_low = 1.0 - w_high

    # Build two-hot vector
    result = torch.zeros(*x_clamped.shape, len(bins), device=x.device, dtype=x.dtype)
    result.scatter_(-1, k.unsqueeze(-1), w_low.unsqueeze(-1))
    result.scatter_(-1, (k + 1).unsqueeze(-1), w_high.unsqueeze(-1))
    return result


def twohot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Recover scalar from distributional prediction: symexp(sum_i probs_i * bins_i)."""
    bins = bins.to(probs.device)
    return symexp((probs * bins).sum(dim=-1))
