"""Top-level MambaWorldModel composing all components."""

import torch
import torch.nn as nn

from retail_world_model.models.rssm import RSSM


class MambaWorldModel(nn.Module):
    """Top-level world model module.

    Composes RSSM (encoder, posterior, backbone, prior, decoder, reward, continue).
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
    ):
        super().__init__()
        self.rssm = RSSM(
            obs_dim=obs_dim,
            act_dim=act_dim,
            d_model=d_model,
            n_cat=n_cat,
            n_cls=n_cls,
            elasticity_path=elasticity_path,
            n_categories=n_categories,
            n_store_features=n_store_features,
        )

    def forward(
        self,
        x_BT: torch.Tensor,
        a_BT: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns dict with latents, backbone outputs, probs, reward stats, continue logits.
        """
        return self.rssm.train_sequence(x_BT, a_BT)

    def imagine(
        self,
        z0: torch.Tensor,
        a_sequence: torch.Tensor,
        inference_params: object | None = None,
    ) -> dict[str, torch.Tensor]:
        """Latent rollout for H steps using mamba.step() recurrently.

        Args:
            z0: (B, latent_dim) initial latent state.
            a_sequence: (B, H, act_dim) action sequence.
            inference_params: Mamba2 InferenceParams or None.

        Returns:
            Dict with h_seq, z_seq, prior_probs_seq, r_mean_seq, r_std_seq.
        """
        B, H, _ = a_sequence.shape
        if inference_params is None:
            inference_params = self.rssm.backbone.init_inference_params(B, max_seqlen=H)

        h_list = []
        z_list = []
        prior_probs_list = []
        r_mean_list = []
        r_std_list = []

        z_t = z0
        for t in range(H):
            a_t = a_sequence[:, t]
            h_t, z_t, prior_probs = self.rssm.imagine_step(z_t, a_t, inference_params)

            r_mean, r_std = self.rssm.reward_ensemble(h_t)

            h_list.append(h_t)
            z_list.append(z_t)
            prior_probs_list.append(prior_probs)
            r_mean_list.append(r_mean)
            r_std_list.append(r_std)

        return {
            "h_seq": torch.stack(h_list, dim=1),        # (B, H, d_model)
            "z_seq": torch.stack(z_list, dim=1),         # (B, H, latent_dim)
            "prior_probs_seq": torch.stack(prior_probs_list, dim=1),
            "r_mean_seq": torch.stack(r_mean_list, dim=1),  # (B, H)
            "r_std_seq": torch.stack(r_std_list, dim=1),     # (B, H)
        }
