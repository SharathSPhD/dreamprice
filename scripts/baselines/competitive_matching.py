"""Competitive matching baseline evaluated on Dominick's test data.

Sets each SKU's price to the category-average price (± 2% noise) and
computes gross margin using actual units sold in the test period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_test_data(data_dir: Path) -> pd.DataFrame:
    """Load and filter Dominick's CSO data to test weeks."""
    df = pd.read_csv(
        data_dir / "category" / "wcso.csv",
        usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT", "OK"],
    )
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0)]
    df["unit_price"] = df["PRICE"] / df["QTY"]
    df["cost"] = df["PRICE"] * (1 - df["PROFIT"] / 100) / df["QTY"]
    test = df[df["WEEK"] >= 341].copy()
    return test


def run_competitive_matching(
    test_df: pd.DataFrame,
    noise_pct: float = 0.02,
    seed: int = 42,
) -> dict:
    """Match category-average price with small noise."""
    rng = np.random.default_rng(seed)
    df = test_df.copy()

    cat_avg = df.groupby("UPC")["unit_price"].transform("mean")
    noise = rng.uniform(-noise_pct, noise_pct, len(df))
    df["proposed_price"] = cat_avg * (1 + noise)
    df["proposed_price"] = np.maximum(df["proposed_price"], df["cost"] * 1.01)
    df["gross_margin"] = (df["proposed_price"] - df["cost"]) * df["MOVE"]

    total_margin = df["gross_margin"].sum()
    n_weeks = df["WEEK"].nunique()
    return {
        "method": "competitive_matching",
        "noise_pct": noise_pct,
        "total_gross_margin": float(total_margin),
        "mean_return": float(total_margin / n_weeks),
        "n_rows": len(df),
        "n_weeks": int(n_weeks),
        "eval_type": "data_replay",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Competitive matching baseline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/docs/data"),
    )
    parser.add_argument("--noise-pct", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/docs/results/baselines/competitive_matching.json"),
    )
    args = parser.parse_args()

    print("Loading Dominick's CSO test data (weeks 341-400)...")
    test_df = load_test_data(args.data_dir)
    print(f"  {len(test_df)} rows, {test_df['WEEK'].nunique()} weeks, {test_df['UPC'].nunique()} UPCs")

    results = run_competitive_matching(test_df, args.noise_pct, args.seed)
    print(f"Competitive matching: mean return = {results['mean_return']:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
