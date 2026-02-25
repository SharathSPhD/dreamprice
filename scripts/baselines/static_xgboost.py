"""Static XGBoost demand model + per-SKU price optimization baseline.

1. Train GradientBoostingRegressor on Dominick's training data (weeks 1-280)
2. For each test-period SKU-week: predict demand at multiple prices, pick
   the profit-maximising one
3. Report cumulative gross margin over test weeks 341-400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Dominick's CSO data, split into train and test."""
    df = pd.read_csv(
        data_dir / "category" / "wcso.csv",
        usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT", "OK"],
    )
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0)]
    df["unit_price"] = df["PRICE"] / df["QTY"]
    df["cost"] = df["PRICE"] * (1 - df["PROFIT"] / 100) / df["QTY"]
    df["log_move"] = np.log1p(df["MOVE"])
    df["log_price"] = np.log(df["unit_price"].clip(lower=0.01))

    train = df[df["WEEK"] <= 280].copy()
    test = df[df["WEEK"] >= 341].copy()
    return train, test


def train_demand_model(
    train_df: pd.DataFrame,
    seed: int = 42,
) -> GradientBoostingRegressor:
    """Train demand model: log_price, WEEK, STORE -> log(1+MOVE)."""
    features = ["log_price", "WEEK", "STORE"]
    X = train_df[features].values.astype(np.float32)
    y = train_df["log_move"].values.astype(np.float32)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed,
        subsample=0.8,
    )
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"  Demand model train R²: {r2:.4f}")
    return model


def optimise_prices(
    model: GradientBoostingRegressor,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each test row, find the profit-maximizing price."""
    df = test_df.copy()
    optimal_prices = []
    predicted_demands = []

    for _, row in df.iterrows():
        cost = row["cost"]
        week = row["WEEK"]
        store = row["STORE"]

        def neg_profit(log_p: float) -> float:
            x = np.array([[log_p, week, store]], dtype=np.float32)
            log_demand = model.predict(x)[0]
            demand = np.expm1(max(log_demand, 0))
            price = np.exp(log_p)
            return -(price - cost) * demand

        price_min = max(cost * 1.01, 0.10)
        result = minimize_scalar(
            neg_profit,
            bounds=(np.log(price_min), np.log(cost * 3.0)),
            method="bounded",
        )
        opt_price = np.exp(result.x)
        x_opt = np.array([[result.x, week, store]], dtype=np.float32)
        pred_demand = np.expm1(max(model.predict(x_opt)[0], 0))

        optimal_prices.append(opt_price)
        predicted_demands.append(pred_demand)

    df["optimal_price"] = optimal_prices
    df["predicted_demand"] = predicted_demands
    df["gross_margin"] = (df["optimal_price"] - df["cost"]) * df["predicted_demand"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Static XGBoost baseline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/docs/data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test-rows", type=int, default=5000,
                        help="Subsample test rows for tractability")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/docs/results/baselines/static_xgboost.json"),
    )
    args = parser.parse_args()

    print("Loading Dominick's CSO data...")
    train_df, test_df = load_data(args.data_dir)
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    print("Training demand model...")
    model = train_demand_model(train_df, args.seed)

    if args.max_test_rows and len(test_df) > args.max_test_rows:
        test_sample = test_df.sample(args.max_test_rows, random_state=args.seed)
        scale_factor = len(test_df) / args.max_test_rows
        print(f"  Subsampled test to {args.max_test_rows} rows (scale factor: {scale_factor:.1f}x)")
    else:
        test_sample = test_df
        scale_factor = 1.0

    print("Optimising prices for each test row...")
    result_df = optimise_prices(model, test_sample)

    total_margin = result_df["gross_margin"].sum() * scale_factor
    n_weeks = test_df["WEEK"].nunique()

    results = {
        "method": "static_xgboost",
        "total_gross_margin": float(total_margin),
        "mean_return": float(total_margin / n_weeks),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "sampled_rows": len(test_sample),
        "n_weeks": int(n_weeks),
        "eval_type": "data_replay",
    }
    print(f"Static XGBoost: mean return = {results['mean_return']:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
