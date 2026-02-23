"""Offline RL utilities: MOPO LCB, hybrid replay, return normalization."""

from __future__ import annotations

import torch
import torch.utils.data


def mopo_lcb(
    r_mean: torch.Tensor,
    r_std: torch.Tensor,
    lambda_lcb: float = 1.0,
) -> torch.Tensor:
    """MOPO-style lower confidence bound pessimism.

    r_pessimistic = r_mean - lambda_lcb * r_std

    Only modifies the reward signal -- does NOT change critic training.
    Start with lambda_lcb=1.0, search {0.25, 0.5, 1.0, 2.0, 5.0}.
    """
    return r_mean - lambda_lcb * r_std


def hybrid_replay_sample(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    frac_recent: float = 0.30,
    recent_weeks: int = 104,
    total_weeks: int = 400,
    quarter_len: int = 13,
) -> list[int]:
    """Sample indices with hybrid replay: 70% uniform strata + 30% recent.

    70% uniform across quarterly strata (13-week buckets).
    30% from most recent *recent_weeks* weeks.

    Args:
        dataset: Must have a ``week`` attribute or be indexable with week info.
        batch_size: Total number of indices to return.
        frac_recent: Fraction sampled from recent period.
        recent_weeks: How many weeks count as "recent" (default 104 = 2 years).
        total_weeks: Total weeks in dataset.
        quarter_len: Weeks per quarterly stratum.

    Returns:
        List of sampled indices.
    """
    n = len(dataset)  # type: ignore[arg-type]
    n_recent = int(batch_size * frac_recent)
    n_uniform = batch_size - n_recent

    # For simplicity, assume dataset items are ordered by week.
    # Recent = last (recent_weeks / total_weeks) fraction of indices.
    recent_start = max(0, int(n * (1.0 - recent_weeks / total_weeks)))

    # Recent sample
    recent_indices = torch.randint(recent_start, n, (n_recent,)).tolist()

    # Quarterly strata: divide [0, n) into strata of ~(quarter_len/total_weeks * n) each
    n_strata = max(1, total_weeks // quarter_len)
    stratum_size = n // n_strata
    uniform_indices: list[int] = []
    for _ in range(n_uniform):
        s = torch.randint(0, n_strata, (1,)).item()
        lo = s * stratum_size
        hi = min(lo + stratum_size, n)
        idx = torch.randint(lo, hi, (1,)).item()
        uniform_indices.append(idx)

    return uniform_indices + recent_indices


class PercentileReturnNorm:
    """Return normalization: R_norm = R_lambda / max(1, P95 - P5).

    Uses EMA decay=0.99 for running P95, P5 estimates.
    Ensures returns are never amplified, only compressed.
    """

    def __init__(self, ema_decay: float = 0.99) -> None:
        self.ema_decay = ema_decay
        self._p5: float | None = None
        self._p95: float | None = None

    def update(self, returns: torch.Tensor) -> None:
        """Update running percentile estimates from a batch of returns."""
        flat = returns.detach().flatten().float()
        p5 = torch.quantile(flat, 0.05).item()
        p95 = torch.quantile(flat, 0.95).item()

        if self._p5 is None:
            self._p5 = p5
            self._p95 = p95
        else:
            d = self.ema_decay
            self._p5 = d * self._p5 + (1 - d) * p5
            self._p95 = d * self._p95 + (1 - d) * p95  # type: ignore[operator]

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns: R / max(1, P95 - P5)."""
        if self._p5 is None or self._p95 is None:
            return returns
        scale = max(1.0, self._p95 - self._p5)
        return returns / scale
