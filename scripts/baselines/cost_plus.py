"""Cost-plus fixed markup baseline evaluated on Dominick's test data.

Applies a fixed markup over wholesale cost to each SKU-week in the test period
(weeks 341-400). Returns the cumulative gross margin.
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


def run_cost_plus(test_df: pd.DataFrame, markup: float) -> dict:
    """Apply fixed markup and compute gross margin using actual units sold."""
    df = test_df.copy()
    df["proposed_price"] = df["cost"] * (1 + markup)
    df["gross_margin"] = (df["proposed_price"] - df["cost"]) * df["MOVE"]
    total_margin = df["gross_margin"].sum()
    n_weeks = df["WEEK"].nunique()
    return {
        "method": "cost_plus",
        "markup": markup,
        "total_gross_margin": float(total_margin),
        "mean_return": float(total_margin / n_weeks),
        "n_rows": len(df),
        "n_weeks": int(n_weeks),
        "eval_type": "data_replay",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-plus baseline on Dominick's data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/docs/data"),
    )
    parser.add_argument("--markup", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/docs/results/baselines/cost_plus.json"),
    )
    args = parser.parse_args()

    print("Loading Dominick's CSO test data (weeks 341-400)...")
    test_df = load_test_data(args.data_dir)
    print(f"  {len(test_df)} rows, {test_df['WEEK'].nunique()} weeks, {test_df['UPC'].nunique()} UPCs")

    results = run_cost_plus(test_df, args.markup)
    print(f"Cost-plus ({args.markup:.0%}): mean return = {results['mean_return']:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
