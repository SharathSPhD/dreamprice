"""Cost-plus fixed markup baseline.

price = cost * (1 + target_margin)
Sweep margins {0.15, 0.20, 0.25, 0.30}.
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run_cost_plus(
    cost_vector: np.ndarray,
    demand_fn,
    target_margin: float,
    n_episodes: int = 10,
    H: int = 13,
    seed: int = 42,
) -> dict[str, float]:
    """Run cost-plus baseline for multiple episodes.

    Args:
        cost_vector: (n_skus,) per-SKU cost.
        demand_fn: Callable(prices) -> units_sold.
        target_margin: Markup fraction (e.g. 0.25 = 25%).
        n_episodes: Number of episodes to evaluate.
        H: Steps per episode.
        seed: Random seed.

    Returns:
        Dict of metrics.
    """
    prices = cost_vector * (1 + target_margin)

    total_profit = 0.0
    total_revenue = 0.0

    for ep in range(n_episodes):
        ep_profit = 0.0
        for step in range(H):
            units_sold = demand_fn(prices)
            profit = ((prices - cost_vector) * units_sold).sum()
            ep_profit += profit
            total_revenue += (prices * units_sold).sum()
        total_profit += ep_profit

    avg_profit = total_profit / n_episodes
    avg_revenue = total_revenue / n_episodes

    return {
        "target_margin": target_margin,
        "avg_episode_profit": float(avg_profit),
        "avg_episode_revenue": float(avg_revenue),
        "profit_per_step": float(avg_profit / H),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-plus baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-skus", type=int, default=25)
    parser.add_argument("--H", type=int, default=13)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Synthetic cost and demand for standalone testing
    cost_vector = np.random.uniform(0.50, 3.00, size=args.n_skus).astype(np.float32)

    def demand_fn(prices: np.ndarray) -> np.ndarray:
        """Simple log-linear demand with elasticity ~ -2.5."""
        base = 100.0
        return np.clip(base * np.exp(-2.5 * np.log(np.clip(prices, 0.01, None))), 0, 10000)

    if args.use_wandb and HAS_WANDB:
        wandb.init(project="dreamprice", group="baselines", name="cost-plus")

    margins = [0.15, 0.20, 0.25, 0.30]
    for margin in margins:
        metrics = run_cost_plus(
            cost_vector, demand_fn, margin,
            n_episodes=args.n_episodes, H=args.H, seed=args.seed,
        )
        print(f"Margin {margin:.0%}: profit/step={metrics['profit_per_step']:.2f}")
        if args.use_wandb and HAS_WANDB:
            wandb.log(metrics)

    if args.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
