"""MOPO-style reward ensemble for pessimistic offline RL."""

import torch
import torch.nn as nn

from retail_world_model.utils.distributions import twohot_decode


class RewardEnsemble(nn.Module):
    """5 independent reward heads for MOPO LCB.

    r_pessimistic = r_mean - lambda_lcb * r_std.
    Each head: Linear -> SiLU -> Linear -> n_bins logits (twohot distributional).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 5,
        n_bins: int = 255,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_bins = n_bins

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, n_bins),
            )
            for _ in range(n_heads)
        ])

        # 255 bins uniformly in symlog space [-20, +20]
        self.register_buffer("bins", torch.linspace(-20, 20, n_bins))

    def get_bins(self) -> torch.Tensor:
        """Return bin centers in symlog space."""
        return self.bins

    def forward_logits(self, h_t: torch.Tensor) -> torch.Tensor:
        """Return raw logits from all heads.

        Args:
            h_t: (B, d_model).

        Returns:
            logits: (n_heads, B, n_bins).
        """
        return torch.stack([head(h_t) for head in self.heads], dim=0)

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std reward predictions.

        Args:
            h_t: (B, d_model) — backbone output (NOT internal SSM state).

        Returns:
            r_mean: (B,) mean predicted reward.
            r_std: (B,) std across ensemble heads.
        """
        logits = self.forward_logits(h_t)  # (n_heads, B, n_bins)
        probs = torch.softmax(logits, dim=-1)

        # Decode each head to scalar
        rewards = torch.stack([
            twohot_decode(probs[i], self.bins) for i in range(self.n_heads)
        ], dim=0)  # (n_heads, B)

        r_mean = rewards.mean(dim=0)  # (B,)
        r_std = rewards.std(dim=0)    # (B,)
        return r_mean, r_std
