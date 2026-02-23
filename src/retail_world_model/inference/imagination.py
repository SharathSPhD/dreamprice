"""Latent rollout (imagination) for DreamerV3-style actor-critic training."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from retail_world_model.training.offline_utils import mopo_lcb


class ImagineInterface(Protocol):
    """Protocol for world models used in imagination rollouts."""

    def encode_obs(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def imagine_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...
    def reset_state(self, batch_size: int) -> dict[str, torch.Tensor]: ...


def rollout_imagination(
    model: Any,
    z0: torch.Tensor,
    h0: torch.Tensor,
    policy: Any,
    H: int = 13,
    lambda_lcb: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Latent rollout for H steps using policy.

    Uses model.imagine_step() for recurrent inference (NOT parallel forward).

    Args:
        model: World model implementing imagine_step(z, a, h).
        z0: (B, z_dim) initial stochastic state.
        h0: (B, d_model) initial deterministic state (Mamba output).
        policy: Actor with act(h_t) -> (action, log_prob, entropy).
        H: Imagination horizon (default 13 = one retail quarter).
        lambda_lcb: MOPO pessimism coefficient.

    Returns:
        Dict with keys: zs, hs, actions, log_probs, entropies,
        rewards_mean, rewards_std, rewards_pessimistic, continues, values.
    """
    B = z0.shape[0]
    device = z0.device

    zs = [z0]
    hs = [h0]
    actions = []
    log_probs = []
    entropies = []
    rewards_mean = []
    rewards_std = []
    rewards_pessimistic = []
    continues = []
    values = []

    z_t = z0
    h_t = h0

    for _t in range(H):
        # Actor selects action from current state
        state_input = torch.cat([h_t, z_t], dim=-1)
        action, log_prob, entropy = policy.act(state_input)
        actions.append(action)
        log_probs.append(log_prob)
        entropies.append(entropy)

        # Critic evaluates current state
        value = policy.critique(state_input)
        values.append(value)

        # World model predicts next state
        step_out = model.imagine_step(z_t, action.float(), h_t)
        h_t = step_out["h"]
        z_t = step_out["z"]
        r_mean = step_out["r_mean"]
        r_std = step_out.get("r_std", torch.zeros_like(r_mean))
        cont = step_out.get("continue", torch.ones(B, device=device))

        rewards_mean.append(r_mean)
        rewards_std.append(r_std)
        rewards_pessimistic.append(mopo_lcb(r_mean, r_std, lambda_lcb))
        continues.append(cont)

        zs.append(z_t)
        hs.append(h_t)

    # Final value for bootstrap
    state_input = torch.cat([h_t, z_t], dim=-1)
    values.append(policy.critique(state_input))

    return {
        "zs": torch.stack(zs, dim=1),  # (B, H+1, z_dim)
        "hs": torch.stack(hs, dim=1),  # (B, H+1, d_model)
        "actions": torch.stack(actions, dim=1),  # (B, H, ...)
        "log_probs": torch.stack(log_probs, dim=1),  # (B, H)
        "entropies": torch.stack(entropies, dim=1),  # (B, H)
        "rewards_mean": torch.stack(rewards_mean, dim=1),  # (B, H)
        "rewards_std": torch.stack(rewards_std, dim=1),  # (B, H)
        "rewards_pessimistic": torch.stack(rewards_pessimistic, dim=1),  # (B, H)
        "continues": torch.stack(continues, dim=1),  # (B, H)
        "values": torch.stack(values, dim=1),  # (B, H+1)
    }


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.95,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute lambda-returns (GAE-style) for imagined trajectories.

    R_t^lambda = r_t + gamma * c_t * [(1-lambda)*V(s_{t+1}) + lambda*R_{t+1}^lambda]

    Args:
        rewards: (B, H) rewards at each step.
        values: (B, H+1) value estimates (last is bootstrap).
        continues: (B, H) continuation flags (0 at episode end).
        gamma: Discount factor.
        lambda_: GAE lambda.

    Returns:
        (B, H) lambda-returns.
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)

    # Bootstrap from final value
    last_return = values[:, -1]

    for t in reversed(range(H)):
        last_return = (
            rewards[:, t]
            + gamma * continues[:, t] * (
                (1 - lambda_) * values[:, t + 1] + lambda_ * last_return
            )
        )
        returns[:, t] = last_return

    return returns
