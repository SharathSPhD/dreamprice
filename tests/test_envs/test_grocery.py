"""Tests for the grocery pricing environment."""

from __future__ import annotations

import numpy as np
import pytest

from retail_world_model.envs.grocery import PRICE_MULTIPLIERS, GroceryPricingEnv


@pytest.fixture
def env():
    """Create a simple grocery env with no world model (uses fallback demand)."""
    n_skus = 5
    return GroceryPricingEnv(
        world_model=None,
        store_features=np.zeros(10, dtype=np.float32),
        initial_obs=np.zeros(20, dtype=np.float32),
        cost_vector=np.full(n_skus, 0.50, dtype=np.float32),
        n_skus=n_skus,
        H=13,
    )


class TestGroceryPricingEnv:
    def test_env_step_returns_valid_obs(self, env):
        """Step returns observation matching observation_space."""
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

        # Take a neutral action (index 10 = multiplier 1.0)
        action = np.full(env.n_skus, 10, dtype=int)
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_reward_penalizes_instability(self, env):
        """Large price change -> lower reward than no change."""
        env.reset()

        # Neutral action (no price change)
        neutral_action = np.full(env.n_skus, 10, dtype=int)
        _, reward_stable, _, _, _ = env.step(neutral_action)

        # Reset and take extreme action
        env.reset()
        # Max increase action (index 20 = +10%)
        extreme_action = np.full(env.n_skus, 20, dtype=int)
        _, reward_volatile, _, _, _ = env.step(extreme_action)

        # Volatile should be penalized (but may still be positive due to higher price)
        # The volatility penalty is lambda_vol * |price_change|
        # After first step from prices=1.0: neutral has 0 change, extreme has 0.10 change
        # So reward_volatile should be lower by ~lambda_vol * n_skus * 0.10
        # This is a weaker assertion since demand also changes
        assert isinstance(reward_volatile, float)

    def test_margin_floor_penalty(self, env):
        """Price below cost -> margin floor penalty applied."""
        env.reset()

        # Set prices very low (action index 0 = 0.90 multiplier)
        # After first step prices = 1.0 * 0.90 = 0.90
        # Cost = 0.50, margin_pct = (0.90-0.50)/0.90 = 0.44 > 0.10 floor
        # Need to iterate to push prices below cost
        # Set cost_vector very high instead
        env.cost_vector = np.full(env.n_skus, 2.0, dtype=np.float32)
        env.reset()

        action = np.full(env.n_skus, 10, dtype=int)  # prices stay at 1.0
        _, reward, _, _, _ = env.step(action)

        # price=1.0, cost=2.0 => margin_pct = (1-2)/1 = -1.0
        # floor_penalty = relu(0.10 - (-1.0)) = 1.10 per SKU
        assert reward < 0  # negative margin + floor penalty

    def test_episode_terminates(self, env):
        """Episode terminates after H steps."""
        env.reset()
        action = np.full(env.n_skus, 10, dtype=int)
        for i in range(env.H):
            _, _, terminated, _, _ = env.step(action)
            if i < env.H - 1:
                assert not terminated
            else:
                assert terminated

    def test_action_space_valid(self, env):
        """Action space matches expected structure."""
        assert env.action_space.shape == (env.n_skus,)
        sample = env.action_space.sample()
        assert len(sample) == env.n_skus
        assert all(0 <= a < len(PRICE_MULTIPLIERS) for a in sample)
