"""Feature transforms for DreamPrice: symlog, Hausman IV, promotions, etc."""

from __future__ import annotations

import numpy as np
import pandas as pd


def symlog(x: np.ndarray | float) -> np.ndarray | float:
    """sign(x) * log(|x| + 1) -- NEVER use raw log."""
    return np.sign(x) * np.log(np.abs(x) + 1)


def symexp(x: np.ndarray | float) -> np.ndarray | float:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def compute_unit_price(df: pd.DataFrame) -> pd.Series:
    """unit_price = PRICE / QTY."""
    return df["PRICE"] / df["QTY"]


def compute_cost(df: pd.DataFrame) -> pd.Series:
    """cost = PRICE * (1 - PROFIT/100) / QTY."""
    return df["PRICE"] * (1.0 - df["PROFIT"] / 100.0) / df["QTY"]


def compute_hausman_iv(df: pd.DataFrame) -> pd.Series:
    """Hausman IV: mean of log(unit_price) across OTHER stores for each (UPC, WEEK).

    hausman_iv = (sum_log_price_all_stores - own_log_price) / (n_stores - 1)
    """
    df = df.copy()
    df["_log_up"] = np.log(df["PRICE"] / df["QTY"])

    # For each (UPC, WEEK): sum of log prices and count of stores
    group_stats = df.groupby(["UPC", "WEEK"])["_log_up"].agg(["sum", "count"])
    group_stats.columns = ["_sum_log", "_n_stores"]

    df = df.join(group_stats, on=["UPC", "WEEK"])

    # Exclude own store: (sum - own) / (n - 1)
    iv = (df["_sum_log"] - df["_log_up"]) / (df["_n_stores"] - 1)

    # Handle single-store case (n_stores == 1 => division by zero)
    iv = iv.fillna(0.0)

    return iv


def flag_promotions(df: pd.DataFrame) -> pd.Series:
    """Return bool Series. on_promotion if unit_price < 0.95 * modal_price.

    Modal price = mode of unit_price per UPC (SALE field has false negatives).
    """
    unit_price = df["PRICE"] / df["QTY"]

    # Compute modal price per UPC
    modal_price = unit_price.groupby(df["UPC"]).transform(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.median()
    )

    return unit_price < (0.95 * modal_price)


def build_observation_vector(
    df: pd.DataFrame, store_demo_cols: list[str] | None = None
) -> np.ndarray:
    """Stack symlog-transformed features into observation vector.

    Features: [symlog(unit_price), symlog(MOVE), symlog(hausman_iv),
               on_promotion, log(QTY), store_demo_cols...]
    """
    unit_price = compute_unit_price(df)
    hausman_iv = compute_hausman_iv(df)
    on_promo = flag_promotions(df).astype(float)

    features = [
        symlog(unit_price.values),
        symlog(df["MOVE"].values.astype(float)),
        symlog(hausman_iv.values),
        on_promo.values,
        np.log(df["QTY"].values.astype(float) + 1),  # safe log for QTY
    ]

    for col in (store_demo_cols or []):
        features.append(df[col].values.astype(float))

    return np.column_stack(features)


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """STRICTLY chronological split. Train: WEEK<=280, Val: 281-340, Test: 341-400.

    NEVER shuffle across time boundaries.
    """
    train = df[df["WEEK"] <= 280].copy()
    val = df[(df["WEEK"] >= 281) & (df["WEEK"] <= 340)].copy()
    test = df[(df["WEEK"] >= 341) & (df["WEEK"] <= 400)].copy()
    return train, val, test
