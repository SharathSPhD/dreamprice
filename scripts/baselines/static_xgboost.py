"""Static XGBoost demand model + per-SKU price optimization baseline.

1. Train XGBoost: features=[price, week, store_id, upc_id] -> log(units_sold)
2. Optimize price via scipy.optimize.minimize for each SKU independently
3. Log to W&B
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_demand_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 5,
    seed: int = 42,
) -> GradientBoostingRegressor:
    """Train a gradient boosting demand model.

    Args:
        X_train: (N, n_features) features including price.
        y_train: (N,) log(units_sold).
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def optimize_price(
    demand_model: GradientBoostingRegressor,
    base_features: np.ndarray,
    price_idx: int,
    cost: float,
    price_bounds: tuple[float, float] = (0.50, 5.00),
) -> tuple[float, float]:
    """Find profit-maximizing price for a single SKU.

    Args:
        demand_model: Trained demand model.
        base_features: (n_features,) feature vector (price slot will be overwritten).
        price_idx: Index of the price feature in base_features.
        cost: Per-unit cost.
        price_bounds: (min_price, max_price).

    Returns:
        (optimal_price, expected_profit).
    """
    def neg_profit(price: float) -> float:
        features = base_features.copy()
        features[price_idx] = price
        log_demand = demand_model.predict(features.reshape(1, -1))[0]
        demand = np.exp(log_demand)
        return -(price - cost) * demand

    result = minimize_scalar(neg_profit, bounds=price_bounds, method="bounded")
    optimal_price = result.x
    profit = -result.fun
    return float(optimal_price), float(profit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static XGBoost baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-skus", type=int, default=25)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Generate synthetic training data
    n_features = 4  # price, week, store_id, upc_id
    X_train = rng.random((args.n_train, n_features)).astype(np.float32)
    X_train[:, 0] = rng.uniform(0.50, 5.00, args.n_train)  # price
    X_train[:, 1] = rng.integers(1, 401, args.n_train)  # week
    X_train[:, 2] = rng.integers(1, 94, args.n_train)  # store_id
    X_train[:, 3] = rng.integers(0, args.n_skus, args.n_train)  # upc_id

    # Synthetic demand: log(demand) = 5 - 2.5*log(price) + noise
    y_train = 5.0 - 2.5 * np.log(X_train[:, 0]) + rng.normal(0, 0.3, args.n_train)

    model = train_demand_model(X_train, y_train, seed=args.seed)
    print(f"Demand model R2: {model.score(X_train, y_train):.3f}")

    if args.use_wandb and HAS_WANDB:
        wandb.init(project="dreamprice", group="baselines", name="xgboost-static")

    # Optimize price per SKU
    costs = rng.uniform(0.50, 3.00, args.n_skus)
    total_profit = 0.0

    for sku in range(args.n_skus):
        base_features = np.array([1.0, 200, 50, sku], dtype=np.float32)
        opt_price, profit = optimize_price(model, base_features, 0, costs[sku])
        total_profit += profit
        print(f"SKU {sku}: optimal_price={opt_price:.2f}, profit={profit:.2f}")

    metrics = {
        "total_profit": total_profit,
        "avg_profit_per_sku": total_profit / args.n_skus,
    }
    print(f"\nTotal profit: {total_profit:.2f}")

    if args.use_wandb and HAS_WANDB:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
