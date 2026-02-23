"""Gymnasium wrapper for stable-baselines3 compatibility.

SB3 needs a flat action space. This wraps GroceryPricingEnv's MultiDiscrete
into a single Discrete space for DQN, or Box for SAC/PPO continuous variants.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from retail_world_model.envs.grocery import PRICE_MULTIPLIERS, GroceryPricingEnv


class FlatDiscreteWrapper(gym.Wrapper):
    """Flatten MultiDiscrete(21^n_skus) into a single Discrete for DQN.

    For tractability with DQN, we treat all SKUs as sharing the same action.
    (Single scalar action applied to all SKUs.)
    """

    def __init__(self, env: GroceryPricingEnv) -> None:
        super().__init__(env)
        self.n_actions = len(PRICE_MULTIPLIERS)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply same action to all SKUs
        multi_action = np.full(self.env.n_skus, action, dtype=int)  # type: ignore[union-attr]
        return self.env.step(multi_action)


class ContinuousActionWrapper(gym.Wrapper):
    """Map Box(-1, 1) continuous action to discrete price multiplier index for SAC."""

    def __init__(self, env: GroceryPricingEnv) -> None:
        super().__init__(env)
        n_skus = env.n_skus
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_skus,), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Map continuous [-1, 1] to discrete index [0, 20]
        n_actions = len(PRICE_MULTIPLIERS)
        indices = np.clip(
            ((action + 1.0) / 2.0 * (n_actions - 1)).astype(int),
            0,
            n_actions - 1,
        )
        return self.env.step(indices)


def make_env(
    n_skus: int = 5,
    H: int = 13,
    wrapper: str = "discrete",
    seed: int = 42,
) -> gym.Env:
    """Create a wrapped grocery env for SB3 baselines."""
    rng = np.random.default_rng(seed)
    env = GroceryPricingEnv(
        world_model=None,
        store_features=np.zeros(10, dtype=np.float32),
        initial_obs=np.zeros(20, dtype=np.float32),
        cost_vector=rng.uniform(0.50, 2.00, n_skus).astype(np.float32),
        n_skus=n_skus,
        H=H,
    )
    if wrapper == "discrete":
        return FlatDiscreteWrapper(env)
    elif wrapper == "continuous":
        return ContinuousActionWrapper(env)
    return env
