"""Symlog / symexp transforms used throughout DreamPrice."""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(|x| + 1). Defined at zero, symmetric, differentiable everywhere."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
