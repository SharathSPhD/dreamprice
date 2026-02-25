"""RSSM core: composes encoder, decoupled posterior, Mamba-2 backbone, prior head."""

import torch
import torch.nn as nn

from retail_world_model.models.decoder import CausalDemandDecoder
from retail_world_model.models.encoder import EntityEncoder, ObsEncoder
from retail_world_model.models.mamba_backbone import GRUFallback, MambaBackbone
from retail_world_model.models.posterior import DecoupledPosterior
from retail_world_model.models.reward_head import RewardEnsemble
from retail_world_model.utils.distributions import apply_unimix, sample_straight_through


class RSSM(nn.Module):
    """Recurrent State-Space Model with DRAMA-style decoupled posterior.

    Composes: ObsEncoder/EntityEncoder -> DecoupledPosterior + MambaBackbone/GRU.
    Posterior: z_t = posterior(x_t)  <- DRAMA, no h_t.
    Backbone: h_t = backbone(cat(z_t, a_t))  <- h_t is OUTPUT.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 512,
        n_cat: int = 32,
        n_cls: int = 32,
        elasticity_path: str | None = None,
        n_categories: int = 1,
        n_store_features: int = 12,
        encoder_type: str = "flat",
        backbone_type: str = "mamba",
        n_upcs: int = 0,
        n_stores: int = 0,
        n_brands: int = 100,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.n_cat = n_cat
        self.n_cls = n_cls
        self.encoder_type = encoder_type
        self.backbone_type = backbone_type
        latent_dim = n_cat * n_cls

        # Encoder: obs -> d_model
        if encoder_type == "entity" and n_upcs > 0 and n_stores > 0:
            self.obs_encoder = EntityEncoder(
                n_upcs=n_upcs,
                n_stores=n_stores,
                n_continuous=obs_dim,
                d_model=d_model,
                n_brands=n_brands,
            )
        else:
            self.obs_encoder = ObsEncoder(obs_dim, d_model)

        # Posterior: x_t -> (z_t, probs). NEVER receives h_t.
        self.posterior = DecoupledPosterior(d_model, d_model, n_cat, n_cls)

        # Input projection: cat(z_t, a_t) -> d_model
        self.input_proj = nn.Linear(latent_dim + act_dim, d_model)

        # Backbone: Mamba-2 or forced GRU
        if backbone_type == "gru":
            self.backbone = GRUFallback(d_model=d_model)
        else:
            self.backbone = MambaBackbone(d_model=d_model)

        # Prior head: h_t -> logits for next z
        self.prior_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_cat * n_cls),
        )

        # Decoder
        self.decoder = CausalDemandDecoder(
            latent_dim=latent_dim,
            n_categories=n_categories,
            n_store_features=n_store_features,
            elasticity_path=elasticity_path,
        )

        # Reward ensemble
        self.reward_ensemble = RewardEnsemble(d_model=d_model)

        # Continue head
        self.continue_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        # Observation decoder: reconstruct obs from latent + backbone state
        self.obs_decoder = nn.Sequential(
            nn.Linear(d_model + latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, obs_dim),
        )

    def encode_obs(
        self,
        x_t: torch.Tensor,
        entity_ids: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into stochastic latent.

        Returns (z_t, probs). NEVER passes h_t to posterior.
        """
        encoded = self._encode_raw(x_t, entity_ids)
        return self.posterior(encoded)

    def _encode_raw(
        self,
        x_t: torch.Tensor,
        entity_ids: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route through ObsEncoder or EntityEncoder based on config."""
        if self.encoder_type == "entity" and isinstance(self.obs_encoder, EntityEncoder):
            ids = entity_ids or {}
            default_ids = torch.zeros(
                *x_t.shape[:-1], dtype=torch.long, device=x_t.device
            )
            return self.obs_encoder(
                upc_ids=ids.get("upc_ids", default_ids),
                store_ids=ids.get("store_ids", default_ids),
                continuous_feats=x_t,
                month_ids=ids.get("month_ids"),
            )
        return self.obs_encoder(x_t)

    def prior_from_h(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior z from backbone output h_t.

        Returns (z_prior, prior_probs).
        """
        logits = self.prior_head(h_t)
        logits = logits.unflatten(-1, (self.n_cat, self.n_cls))
        probs = apply_unimix(logits)
        z_flat = sample_straight_through(probs).flatten(-2)
        return z_flat, probs

    def imagine_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        inference_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single recurrent step for imagination rollout.

        Args:
            z_t: (B, latent_dim) current latent.
            a_t: (B, act_dim) action.
            inference_params: Mamba2 InferenceParams for recurrent mode.

        Returns:
            h_next: (B, d_model) backbone output.
            z_next: (B, latent_dim) prior sample.
            prior_probs: (B, n_cat, n_cls).
        """
        inp = self.input_proj(torch.cat([z_t, a_t], dim=-1))
        if self.backbone_type == "gru":
            h_next = self.backbone.step(inp)
        else:
            h_next = self.backbone.step(inp, inference_params)
        z_next, prior_probs = self.prior_from_h(h_next)
        return h_next, z_next, prior_probs

    def train_sequence(
        self,
        x_BT: torch.Tensor,
        a_BT: torch.Tensor,
        entity_ids: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full sequence training pass.

        Args:
            x_BT: (B, T, obs_dim) observations.
            a_BT: (B, T, act_dim) actions.
            entity_ids: Optional dict with upc_ids, store_ids, month_ids tensors.

        Returns:
            Dict with z_BT, h_BT, posterior_probs, prior_probs,
            reward_mean, reward_std, continue_logits.
        """
        B, T, _ = x_BT.shape

        # Encode all observations (DRAMA: posterior depends only on x_t)
        encoded = self._encode_raw(x_BT, entity_ids)  # (B, T, d_model)
        z_BT, posterior_probs = self.posterior(encoded)  # z: (B, T, latent_dim)

        # Backbone: parallel SSD scan over full sequence
        inp = self.input_proj(torch.cat([z_BT, a_BT], dim=-1))  # (B, T, d_model)
        h_BT = self.backbone(inp)  # (B, T, d_model)

        # Prior from backbone output
        prior_logits = self.prior_head(h_BT)  # (B, T, n_cat*n_cls)
        prior_logits = prior_logits.unflatten(-1, (self.n_cat, self.n_cls))
        prior_probs = apply_unimix(prior_logits)

        # Reward predictions
        h_flat = h_BT.reshape(B * T, -1)
        r_mean, r_std = self.reward_ensemble(h_flat)
        r_mean = r_mean.reshape(B, T)
        r_std = r_std.reshape(B, T)

        # Reward logits for twohot loss
        r_logits = self.reward_ensemble.forward_logits(h_flat)  # (n_heads, B*T, n_bins)
        r_logits_mean = r_logits.mean(dim=0).reshape(B, T, -1)  # (B, T, n_bins)

        # Continue predictions
        continue_logits = self.continue_head(h_BT).squeeze(-1)  # (B, T)

        # Observation reconstruction
        x_recon_BT = self.obs_decoder(torch.cat([h_BT, z_BT], dim=-1))

        return {
            "z_posterior_BT": z_BT,
            "h_BT": h_BT,
            "posterior_probs_BT": posterior_probs,
            "prior_probs_BT": prior_probs,
            "x_recon_BT": x_recon_BT,
            "reward_mean": r_mean,
            "reward_std": r_std,
            "reward_logits_BT": r_logits_mean,
            "continue_logits": continue_logits,
        }
