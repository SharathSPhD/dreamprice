"""DRAMA-style decoupled posterior: z_t = encode(x_t) ONLY, never receives h_t."""

import torch
import torch.nn as nn

from retail_world_model.utils.distributions import apply_unimix, sample_straight_through


class DecoupledPosterior(nn.Module):
    """DRAMA-style decoupled posterior.

    z_t depends ONLY on x_t (observation), never on h_t.
    This enables Mamba-2 parallel SSD scan during training.

    Architecture: x_t -> MLP -> logits (B, n_cat*n_cls) -> reshape (B, n_cat, n_cls)
    Apply unimix -> sample_straight_through -> z_t
    """

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 512,
        n_cat: int = 32,
        n_cls: int = 32,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.n_cat = n_cat
        self.n_cls = n_cls
        self.unimix = unimix

        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.SiLU(),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, n_cat * n_cls),
        )

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into stochastic latent.

        Args:
            x_t: (B, obs_dim) or (B, T, obs_dim). h_t is NEVER an input.

        Returns:
            z_t_onehot: (*batch, n_cat * n_cls) — straight-through one-hot.
            probs: (*batch, n_cat, n_cls) — categorical probabilities with unimix.
        """
        logits = self.net(x_t)  # (..., n_cat * n_cls)
        logits = logits.unflatten(-1, (self.n_cat, self.n_cls))  # (..., n_cat, n_cls)

        probs = apply_unimix(logits, mix=self.unimix)  # (..., n_cat, n_cls)
        z_onehot = sample_straight_through(probs)  # (..., n_cat, n_cls)
        z_flat = z_onehot.flatten(-2)  # (..., n_cat * n_cls)

        return z_flat, probs
