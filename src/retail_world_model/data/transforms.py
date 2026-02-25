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
    """on_promotion if unit_price < 0.95 * modal_price per (UPC, STORE)."""
    unit_price = df["PRICE"] / df["QTY"]
    modal_price = unit_price.groupby([df["UPC"], df["STORE"]]).transform(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.median()
    )
    return unit_price < (0.95 * modal_price)


def compute_discount_depth(df: pd.DataFrame) -> pd.Series:
    """discount_depth = (modal_price - unit_price) / modal_price, per (UPC, STORE)."""
    unit_price = df["PRICE"] / df["QTY"]
    modal_price = unit_price.groupby([df["UPC"], df["STORE"]]).transform(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.median()
    )
    return ((modal_price - unit_price) / modal_price.clip(lower=1e-6)).clip(lower=0.0)


def compute_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lag features within (STORE, UPC) groups sorted by WEEK.

    Returns DataFrame with columns like '{value_col}_lag_{lag}'.
    """
    if lags is None:
        lags = [1, 2]
    df = df.sort_values(group_cols + ["WEEK"])
    result = pd.DataFrame(index=df.index)
    grouped = df.groupby(group_cols)[value_col]
    for lag in lags:
        result[f"{value_col}_lag_{lag}"] = grouped.shift(lag)
    return result.fillna(0.0)


def compute_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int = 4,
) -> pd.DataFrame:
    """Rolling mean and std within (STORE, UPC) groups sorted by WEEK."""
    df = df.sort_values(group_cols + ["WEEK"])
    grouped = df.groupby(group_cols)[value_col]
    result = pd.DataFrame(index=df.index)
    result[f"{value_col}_rolling_mean_{window}"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    result[f"{value_col}_rolling_std_{window}"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
    )
    return result


def compute_price_index(df: pd.DataFrame) -> pd.Series:
    """price_index = unit_price / category_mean_price per WEEK."""
    unit_price = df["PRICE"] / df["QTY"]
    cat_mean = unit_price.groupby(df["WEEK"]).transform("mean")
    return unit_price / cat_mean.clip(lower=1e-6)


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos week encoding, quarter one-hot, holiday indicator.

    Dominick's WEEK is an integer starting from 1. We use modulo 52 for
    annual cyclical encoding.
    """
    result = pd.DataFrame(index=df.index)
    week_in_year = (df["WEEK"] - 1) % 52
    result["week_sin"] = np.sin(2 * np.pi * week_in_year / 52)
    result["week_cos"] = np.cos(2 * np.pi * week_in_year / 52)

    quarter = (week_in_year // 13).astype(int)
    for q in range(4):
        result[f"quarter_{q}"] = (quarter == q).astype(float)

    # Holiday proxy: weeks around Thanksgiving (wk 47), Christmas (wk 51),
    # Easter (~wk 14), Memorial Day (~wk 21), Labor Day (~wk 35)
    holiday_weeks = {14, 21, 35, 47, 51}
    result["holiday"] = week_in_year.isin(holiday_weeks).astype(float)
    return result


def build_observation_vector(
    df: pd.DataFrame,
    store_demo_cols: list[str] | None = None,
) -> np.ndarray:
    """Build full observation vector matching blueprint Section 5.

    Per-SKU features (12 dims):
        symlog(unit_price), symlog(MOVE), discount_depth, on_promotion,
        lag_price_1, lag_price_2, lag_move_1, lag_move_2,
        rolling_mean_move_4, price_index, cost_per_unit, profit_margin

    Temporal features (7 dims):
        week_sin, week_cos, quarter_0..3, holiday

    Store context (N_demo dims):
        store demographic columns passed in
    """
    group_cols = ["STORE", "UPC"]
    unit_price = compute_unit_price(df)
    cost = compute_cost(df)

    unit_price_arr = np.asarray(unit_price, dtype=float)
    cost_arr = np.asarray(cost, dtype=float)
    move_arr = np.asarray(df["MOVE"], dtype=float)
    discount_arr = np.asarray(compute_discount_depth(df), dtype=float)
    promo_arr = np.asarray(flag_promotions(df).astype(float), dtype=float)
    lag_price = (
        compute_lag_features(df, group_cols, "unit_price_raw", [1, 2]).to_numpy(dtype=float)
        if "unit_price_raw" in df.columns
        else np.zeros((len(df), 2))
    )
    lag_move = compute_lag_features(df, group_cols, "MOVE", [1, 2]).to_numpy(dtype=float)
    rolling_mean = (
        compute_rolling_features(df, group_cols, "MOVE", 4).iloc[:, 0].to_numpy(dtype=float)
    )
    price_index_arr = np.asarray(compute_price_index(df), dtype=float)
    margin_arr = (unit_price_arr - cost_arr) / np.clip(unit_price_arr, a_min=1e-6, a_max=None)

    per_sku = np.column_stack(
        [
            symlog(unit_price_arr),
            symlog(move_arr),
            discount_arr,
            promo_arr,
            lag_price,
            lag_move,
            rolling_mean.reshape(-1, 1),
            price_index_arr.reshape(-1, 1),
            cost_arr.reshape(-1, 1),
            margin_arr.reshape(-1, 1),
        ]
    )

    temporal = compute_temporal_features(df).to_numpy(dtype=float)

    demo = (
        np.column_stack([np.asarray(df[col], dtype=float) for col in store_demo_cols])
        if store_demo_cols
        else np.empty((len(df), 0))
    )
    np.nan_to_num(demo, copy=False, nan=0.0)

    obs = np.hstack([per_sku, temporal, demo])
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return obs


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
