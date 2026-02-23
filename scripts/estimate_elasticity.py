"""Estimate price elasticities via Hausman IV + DML-PLIV for CausalDemandDecoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from retail_world_model.data.copula_correction import compute_2scope_copula_residual
from retail_world_model.data.dominicks_loader import load_category
from retail_world_model.data.transforms import (
    compute_hausman_iv,
    compute_unit_price,
    temporal_split,
)


def estimate_elasticity(
    category: str = "cso",
    data_dir: str = "docs/data",
    output_dir: str = "configs/elasticities",
    n_folds: int = 5,
    n_estimators: int = 500,
) -> dict:
    """Run DML-PLIV on Dominick's data and write elasticity JSON.

    Steps:
        1. Load and merge category data
        2. Compute Hausman IV instrument
        3. Run DML-PLIV via doubleml
        4. Validate first-stage F-stat > 10
        5. Run 2sCOPE robustness check
        6. Write configs/elasticities/{category}.json
    """
    from doubleml import DoubleMLData, DoubleMLPLIV

    # Load data
    movement_path = Path(data_dir) / category / f"w{category}.csv"
    upc_path = Path(data_dir) / category / f"upc{category}.csv"
    demo_path = Path(data_dir) / "demo.csv"

    df = load_category(str(movement_path), str(upc_path), str(demo_path))
    train_df, _, _ = temporal_split(df)

    # Derived features
    train_df = train_df.copy()
    train_df["unit_price"] = compute_unit_price(train_df)
    train_df["log_price"] = np.log(train_df["unit_price"].clip(lower=1e-6))
    train_df["log_move"] = np.log(train_df["MOVE"].clip(lower=1) + 1)
    train_df["hausman_iv"] = compute_hausman_iv(train_df)

    # Drop rows with NaN/inf
    cols_needed = ["log_move", "log_price", "hausman_iv"]
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)

    # Control variables (exogenous)
    exog_cols = []
    for col in [
        "INCOME",
        "EDUC",
        "ETHNIC",
        "HSIZEAVG",
        "SSTRDIST",
        "SSTRVOL",
    ]:
        if col in train_df.columns:
            exog_cols.append(col)
            train_df[col] = train_df[col].fillna(train_df[col].median())

    if not exog_cols:
        exog_cols = ["WEEK"]

    # DoubleML data object
    dml_data = DoubleMLData(
        train_df[["log_move", "log_price", "hausman_iv"] + exog_cols],
        y_col="log_move",
        d_cols="log_price",
        z_cols="hausman_iv",
        x_cols=exog_cols,
    )

    # ML learners
    ml_l = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    ml_m = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    ml_r = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)

    # Fit DML-PLIV
    dml = DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=n_folds)
    dml.fit()

    theta = float(dml.coef[0])
    se = float(dml.se[0])
    ci = dml.confint(level=0.95)
    t_stat = float(dml.t_stat[0])
    p_value = float(dml.pval[0])

    # First-stage F-stat approximation (t_stat^2 for single instrument)
    f_stat = t_stat**2

    print(f"DML-PLIV elasticity: {theta:.4f} (SE={se:.4f})")
    print(f"95% CI: [{float(ci.iloc[0, 0]):.4f}, {float(ci.iloc[0, 1]):.4f}]")
    print(f"F-stat (approx): {f_stat:.1f}")

    if f_stat < 10:
        raise ValueError(
            f"Weak instrument: F-stat={f_stat:.1f} < 10. "
            "Hausman IV may be invalid for this category."
        )

    # 2sCOPE robustness check
    cope_beta = float("nan")
    try:
        copula_resid = compute_2scope_copula_residual(
            train_df, "log_price", "log_move", "hausman_iv"
        )
        from sklearn.linear_model import LinearRegression

        X_cope = np.column_stack(
            [
                train_df["log_price"].values,
                copula_resid.values,
            ]
        )
        y_cope = train_df["log_move"].values
        reg = LinearRegression().fit(X_cope, y_cope)
        cope_beta = float(reg.coef_[0])
        print(f"2sCOPE beta: {cope_beta:.4f}")

        if abs(theta - cope_beta) / abs(theta) > 0.5:
            print(
                f"WARNING: DML-PLIV ({theta:.3f}) and "
                f"2sCOPE ({cope_beta:.3f}) differ by >50%. "
                "Investigate endogeneity assumptions."
            )
    except Exception as e:
        print(f"2sCOPE check failed: {e}")

    # Write output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "theta_causal": theta,
        "se": se,
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
        "f_stat_first_stage": f_stat,
        "p_value": p_value,
        "cope_beta_c": cope_beta,
        "n_obs": len(train_df),
        "n_folds": n_folds,
        "category": category,
    }
    out_file = output_path / f"{category}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Written: {out_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Estimate price elasticities")
    parser.add_argument(
        "--category",
        default="cso",
        help="Dominick's category code",
    )
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--output-dir", default="configs/elasticities")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    estimate_elasticity(
        category=args.category,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
