"""Actor-Critic for DreamerV3-style imagination-based policy optimization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from retail_world_model.utils.twohot import make_bins, twohot_decode


class ActorCritic(nn.Module):
    """Actor-Critic for discrete pricing actions with distributional critic.

    Actor: state -> Linear -> SiLU -> Linear -> action logits (Discrete(21) per SKU)
    Critic: state -> Linear -> SiLU -> Linear -> 255 bins (twohot distributional)

    Actor gradient: REINFORCE for discrete actions.
    Entropy regularization: eta=3e-4.
    """

    def __init__(
        self,
        state_dim: int = 1536,  # d_model + z_dim (512 + 1024)
        d_hidden: int = 512,
        action_dim: int = 21,
        n_skus: int = 25,
        n_bins: int = 255,
        eta: float = 3e-4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_skus = n_skus
        self.n_bins = n_bins
        self.eta = eta

        # Actor: produces logits for each SKU's discrete action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_skus * action_dim),
        )

        # Critic: produces distributional value prediction (twohot)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_bins),
        )

        # Register bins as buffer
        self.register_buffer("bins", make_bins(n_bins))

    def act(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.

        Args:
            state: (B, state_dim) concatenation of h_t and z_t.
            deterministic: If True, take argmax actions.

        Returns:
            (actions, log_probs, entropy):
                actions: (B, n_skus) integer actions.
                log_probs: (B,) sum of log probs across SKUs.
                entropy: (B,) sum of entropy across SKUs.
        """
        logits = self.actor(state)  # (B, n_skus * action_dim)
        logits = logits.view(-1, self.n_skus, self.action_dim)  # (B, n_skus, 21)

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()  # (B, n_skus)

        log_probs = dist.log_prob(actions).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        return actions, log_probs, entropy

    def critique(self, state: torch.Tensor) -> torch.Tensor:
        """Predict scalar value from distributional critic.

        Args:
            state: (B, state_dim)

        Returns:
            (B,) scalar value predictions.
        """
        logits = self.critic(state)  # (B, n_bins)
        probs = F.softmax(logits, dim=-1)
        return twohot_decode(probs, self.bins)

    def critique_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw critic logits for loss computation.

        Args:
            state: (B, state_dim)

        Returns:
            (B, n_bins) raw logits.
        """
        return self.critic(state)

    def actor_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """REINFORCE loss with entropy regularization.

        Args:
            log_probs: (B, H) action log probabilities.
            advantages: (B, H) advantage estimates.
            entropy: (B, H) policy entropy.

        Returns:
            Scalar loss.
        """
        # REINFORCE: -log_prob * sg(advantage)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_bonus = -self.eta * entropy.mean()
        return policy_loss + entropy_bonus
