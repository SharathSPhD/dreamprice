"""Twohot encoding for distributional prediction (DreamerV3-style)."""

from __future__ import annotations

import torch

from retail_world_model.utils.symlog import symexp

# Default: 255 bins uniformly in symlog[-20, +20]
N_BINS = 255
BIN_LOW = -20.0
BIN_HIGH = 20.0


def make_bins(n_bins: int = N_BINS, low: float = BIN_LOW, high: float = BIN_HIGH) -> torch.Tensor:
    """Return (n_bins,) tensor of bin centres uniformly spaced in symlog space."""
    return torch.linspace(low, high, n_bins)


def twohot_encode(
    target: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Encode scalar targets as soft two-hot labels over *bins*.

    Args:
        target: (...) arbitrary shape of scalar targets (already in symlog space).
        bins: (n_bins,) sorted bin centres.

    Returns:
        (..., n_bins) soft labels that sum to 1 along the last dim.
    """
    target = target.unsqueeze(-1)  # (..., 1)
    bins = bins.to(target.device)

    # Clamp target into bin range
    target = target.clamp(bins[0], bins[-1])

    # Find the lower bin index for each target
    # k such that bins[k] <= target < bins[k+1]
    below = (bins.unsqueeze(0) <= target).sum(dim=-1) - 1  # (...)
    below = below.clamp(0, len(bins) - 2)
    above = below + 1

    # Interpolation weights
    b_low = bins[below]
    b_high = bins[above]
    span = (b_high - b_low).clamp(min=1e-8)
    w_above = (target.squeeze(-1) - b_low) / span
    w_below = 1.0 - w_above

    labels = torch.zeros(*target.shape[:-1], len(bins), device=target.device)
    labels.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
    labels.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return labels


def twohot_decode(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Recover scalar prediction: symexp(sum(probs_i * bins_i)).

    Args:
        probs: (..., n_bins) probability distribution over bins.
        bins: (n_bins,) bin centres in symlog space.

    Returns:
        (...) scalar predictions in original (un-symlogged) space.
    """
    bins = bins.to(probs.device)
    mean_symlog = (probs * bins).sum(dim=-1)
    return symexp(mean_symlog)
