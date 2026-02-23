"""Observation encoders: flat MLP and entity-factored variants."""

import torch
import torch.nn as nn

from retail_world_model.utils.distributions import symlog


class ObsEncoder(nn.Module):
    """Flat MLP encoder: Linear -> SiLU -> RMSNorm -> Linear -> SiLU -> RMSNorm -> d_model."""

    def __init__(self, obs_dim: int, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.SiLU(),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.RMSNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, obs_dim) or (B, obs_dim) -> (..., d_model)."""
        return self.net(x)


class EntityEncoder(nn.Module):
    """Per-entity embeddings fused with continuous features.

    UPC + store + brand + month_of_year -> d_slot -> fusion MLP -> d_model.
    """

    def __init__(
        self,
        n_upcs: int,
        n_stores: int,
        n_continuous: int,
        d_slot: int = 64,
        d_model: int = 512,
        n_brands: int = 100,
    ):
        super().__init__()
        self.upc_embed = nn.Embedding(n_upcs, 32)
        self.store_embed = nn.Embedding(n_stores, 16)
        self.brand_embed = nn.Embedding(n_brands, 16)
        self.month_embed = nn.Embedding(12, 6)
        self.continuous_proj = nn.Sequential(
            nn.Linear(n_continuous, 32),
            nn.SiLU(),
            nn.RMSNorm(32),
        )
        cat_dim = 32 + 16 + 16 + 6 + 32  # 102
        self.fusion = nn.Sequential(
            nn.Linear(cat_dim, 128),
            nn.SiLU(),
            nn.RMSNorm(128),
            nn.Linear(128, d_model),
        )

    def forward(
        self,
        upc_ids: torch.Tensor,
        store_ids: torch.Tensor,
        continuous_feats: torch.Tensor,
        month_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns: (..., d_model)."""
        parts = [
            self.upc_embed(upc_ids),
            self.store_embed(store_ids),
        ]
        if month_ids is not None:
            parts.append(self.month_embed(month_ids))
        else:
            # Default zero for month embedding
            parts.append(
                torch.zeros(*upc_ids.shape, 6, device=upc_ids.device, dtype=continuous_feats.dtype)
            )
        # Brand ID not always available; caller should provide or we skip
        cont = self.continuous_proj(symlog(continuous_feats))
        parts.append(cont)
        # Pad brand embed to zero if not provided via forward signature
        parts.insert(
            2, torch.zeros(*upc_ids.shape, 16, device=upc_ids.device, dtype=continuous_feats.dtype)
        )
        return self.fusion(torch.cat(parts, dim=-1))
