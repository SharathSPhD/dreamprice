"""Competitive matching baseline.

Set price = competitor_median_price +/- 5% noise.
Since Dominick's lacks time-varying competitor data, we use
category average price as proxy for competitor median.
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run_competitive_matching(
    category_avg_prices: np.ndarray,
    cost_vector: np.ndarray,
    demand_fn,
    noise_pct: float = 0.05,
    n_episodes: int = 10,
    H: int = 13,
    seed: int = 42,
) -> dict[str, float]:
    """Run competitive matching baseline.

    Args:
        category_avg_prices: (n_skus,) average category prices as competitor proxy.
        cost_vector: (n_skus,) per-SKU cost.
        demand_fn: Callable(prices) -> units_sold.
        noise_pct: Price noise as fraction of category average (default 5%).
        n_episodes: Number of evaluation episodes.
        H: Steps per episode.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)

    total_profit = 0.0
    for ep in range(n_episodes):
        ep_profit = 0.0
        for step in range(H):
            noise = rng.uniform(-noise_pct, noise_pct, len(category_avg_prices))
            prices = category_avg_prices * (1 + noise)
            prices = np.clip(prices, cost_vector * 1.01, None)  # ensure above cost
            units_sold = demand_fn(prices)
            profit = ((prices - cost_vector) * units_sold).sum()
            ep_profit += profit
        total_profit += ep_profit

    avg_profit = total_profit / n_episodes
    return {
        "avg_episode_profit": float(avg_profit),
        "profit_per_step": float(avg_profit / H),
        "noise_pct": noise_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Competitive matching baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-skus", type=int, default=25)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cost_vector = rng.uniform(0.50, 3.00, args.n_skus).astype(np.float32)
    category_avg = cost_vector * 1.25  # assume 25% markup is category average

    def demand_fn(prices: np.ndarray) -> np.ndarray:
        base = 100.0
        return np.clip(base * np.exp(-2.5 * np.log(np.clip(prices, 0.01, None))), 0, 10000)

    if args.use_wandb and HAS_WANDB:
        wandb.init(project="dreamprice", group="baselines", name="competitive-matching")

    metrics = run_competitive_matching(
        category_avg, cost_vector, demand_fn,
        n_episodes=args.n_episodes, seed=args.seed,
    )
    print(f"Competitive matching: profit/step={metrics['profit_per_step']:.2f}")

    if args.use_wandb and HAS_WANDB:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
