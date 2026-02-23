"""Causal demand decoder with frozen DML-PLIV elasticities."""

import json
from pathlib import Path

import torch
import torch.nn as nn


class CausalDemandDecoder(nn.Module):
    """Demand decoder with frozen causal price channel.

    demand = theta(category_id) * log_price + MLP(z_t, store_features)

    theta: FROZEN elasticity from DML-PLIV estimates.
    Expected elasticity range: -2.0 to -3.0 (grocery).
    """

    def __init__(
        self,
        latent_dim: int,
        n_categories: int,
        n_store_features: int = 12,
        elasticity_path: str | None = None,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.n_categories = n_categories

        # Load or initialize theta (frozen elasticities)
        if elasticity_path and Path(elasticity_path).exists():
            with open(elasticity_path) as f:
                data = json.load(f)
            thetas = data["theta"]
        else:
            # Prior mean for grocery categories
            thetas = [-2.5] * n_categories

        self.theta = nn.Embedding(n_categories, 1)
        self.theta.weight.data = torch.tensor(thetas, dtype=torch.float32).unsqueeze(1)
        self.theta.weight.requires_grad = False  # ALWAYS FROZEN

        # Residual MLP for everything beyond the causal price effect
        self.residual = nn.Sequential(
            nn.Linear(latent_dim + n_store_features, d_hidden),
            nn.SiLU(),
            nn.RMSNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        log_price: torch.Tensor,
        category_id: torch.Tensor,
        store_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted demand.

        Args:
            z_t: (..., latent_dim) latent state.
            log_price: (...,) or (..., 1) log unit price.
            category_id: (...,) long tensor of category indices.
            store_features: (..., n_store_features) store demographics.

        Returns:
            demand: (...,) predicted demand in symlog space.
        """
        # Causal price effect (frozen)
        theta_val = self.theta(category_id).squeeze(-1)  # (...,)
        if log_price.dim() > theta_val.dim():
            log_price = log_price.squeeze(-1)
        causal_effect = theta_val * log_price

        # Residual from latent state + store context
        residual_input = torch.cat([z_t, store_features], dim=-1)
        residual = self.residual(residual_input).squeeze(-1)

        return causal_effect + residual
