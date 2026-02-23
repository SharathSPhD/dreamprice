"""Multi-SKU grocery pricing environment backed by a world model."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from retail_world_model.envs.base import BaseGroceryEnv

# 21 discrete price multipliers: {0.90, 0.91, ..., 1.10}
PRICE_MULTIPLIERS = np.linspace(0.90, 1.10, 21)


class GroceryPricingEnv(BaseGroceryEnv):
    """Multi-SKU pricing environment backed by world model.

    Action space: Discrete(21) per SKU — price multipliers -10% to +10% in 1pp steps.
    Observation: world model latent z_t concatenated with store features.
    Episode: H steps (one retail quarter, default 13 weeks).

    Reward = sum over SKUs:
      (price - cost) * units_sold          # gross profit
      - lambda_vol * |price - prev_price|  # price stability penalty
      - relu(margin_floor - margin_pct)    # margin floor enforcement
    """

    def __init__(
        self,
        world_model: Any,
        store_features: np.ndarray,
        initial_obs: np.ndarray,
        cost_vector: np.ndarray,
        n_skus: int = 25,
        H: int = 13,
        lambda_vol: float = 0.05,
        margin_floor: float = 0.10,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.store_features = store_features.astype(np.float32)
        self.initial_obs = initial_obs.astype(np.float32)
        self.cost_vector = cost_vector.astype(np.float32)
        self.n_skus = n_skus
        self.H = H
        self.lambda_vol = lambda_vol
        self.margin_floor = margin_floor

        # Action: one discrete action per SKU
        self.action_space = gym.spaces.MultiDiscrete([len(PRICE_MULTIPLIERS)] * n_skus)

        # Observation: latent dim + store features
        # We use initial_obs dim as proxy; real obs comes from world model
        obs_dim = len(initial_obs) + len(store_features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._prices = np.ones(n_skus, dtype=np.float32)
        self._prev_prices = np.ones(n_skus, dtype=np.float32)
        self._current_obs = initial_obs.copy()
        self._h_t: Any = None  # world model hidden state
        self._z_t: Any = None  # world model stochastic state

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prices = np.ones(self.n_skus, dtype=np.float32)
        self._prev_prices = np.ones(self.n_skus, dtype=np.float32)
        self._current_obs = self.initial_obs.copy()

        # Reset world model state
        if hasattr(self.world_model, "reset_state"):
            state = self.world_model.reset_state()
            self._h_t = state.get("h", None)
            self._z_t = state.get("z", None)

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action).flatten()
        assert len(action_arr) == self.n_skus

        # Apply price multipliers
        self._prev_prices = self._prices.copy()
        multipliers = np.array([PRICE_MULTIPLIERS[a] for a in action_arr], dtype=np.float32)
        self._prices = self._prices * multipliers

        # Use world model to predict demand (units_sold)
        units_sold = self._predict_demand(self._prices, action_arr)

        reward = self._compute_reward(
            self._prices, self.cost_vector, units_sold, self._prev_prices
        )

        self._step_count += 1
        terminated = self._step_count >= self.H
        truncated = False

        info = {
            "prices": self._prices.copy(),
            "units_sold": units_sold.copy(),
            "gross_margin": float(((self._prices - self.cost_vector) * units_sold).sum()),
            "step": self._step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _predict_demand(self, prices: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Use world model to predict demand given current prices."""
        if self.world_model is None:
            # Fallback: simple log-linear demand with elasticity ~ -2.5
            base_demand = 100.0
            log_price = np.log(np.clip(prices, 1e-3, None))
            units = base_demand * np.exp(-2.5 * log_price)
            return np.clip(units, 0, None).astype(np.float32)

        # Use world model step if available
        with torch.no_grad():
            a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            if hasattr(self.world_model, "imagine_step") and self._h_t is not None:
                result = self.world_model.imagine_step(self._z_t, a_t, self._h_t)
                self._h_t = result["h"]
                self._z_t = result["z"]
                demand = result.get("demand", result.get("r_mean", torch.ones(self.n_skus)))
                return demand.squeeze(0).numpy()

        # Fallback
        return np.full(self.n_skus, 50.0, dtype=np.float32)

    def _compute_reward(
        self,
        price: np.ndarray,
        cost: np.ndarray,
        units_sold: np.ndarray,
        prev_price: np.ndarray,
    ) -> float:
        """Reward with gross profit, volatility penalty, and margin floor."""
        gross_margin = ((price - cost) * units_sold).sum()
        volatility_penalty = self.lambda_vol * np.abs(price - prev_price).sum()

        margin_pct = (price - cost) / np.clip(price, 1e-3, None)
        floor_penalty = np.maximum(self.margin_floor - margin_pct, 0.0).sum()

        return float(gross_margin - volatility_penalty - floor_penalty)

    def _get_obs(self) -> np.ndarray:
        """Concatenate current obs with store features."""
        return np.concatenate([self._current_obs, self.store_features]).astype(np.float32)
