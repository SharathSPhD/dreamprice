"""Load and clean Dominick's Finer Foods scanner dataset."""

import os
from pathlib import Path

import pandas as pd


def load_movement(path: Path | str) -> pd.DataFrame:
    """Load movement CSV. Drop PRICE_HEX, PROFIT_HEX. Drop OK==0 or PRICE<=0."""
    df = pd.read_csv(path)
    # Drop hex columns if present
    hex_cols = [c for c in ("PRICE_HEX", "PROFIT_HEX") if c in df.columns]
    if hex_cols:
        df = df.drop(columns=hex_cols)
    # Filter bad rows
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0)]
    return df.reset_index(drop=True)


def load_upc(path: Path | str) -> pd.DataFrame:
    """Load UPC metadata CSV."""
    return pd.read_csv(path)


def load_store_demo(path: Path | str) -> pd.DataFrame:
    """Load store demographics CSV."""
    return pd.read_csv(path)


def load_category(
    movement_path: Path | str,
    upc_path: Path | str,
    demo_path: Path | str,
) -> pd.DataFrame:
    """Merge movement, UPC, and demo data. Apply cleaning. Insert zero-sales rows
    for (store, UPC) pairs missing >10% of active weeks."""
    movement = load_movement(movement_path)
    upc = load_upc(upc_path)
    demo = load_store_demo(demo_path)

    # Merge movement with UPC metadata
    df = movement.merge(upc, on="UPC", how="left")
    # Merge with store demographics
    df = df.merge(demo, on="STORE", how="left")

    if not os.environ.get("DREAMPRICE_SKIP_ZERO_SALES"):
        df = _insert_zero_sales(df)

    return df.reset_index(drop=True)


def _insert_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Insert zero-sales rows for (store, UPC) pairs missing >10% of active weeks.

    For missing weeks: MOVE=0, PRICE=median price for that UPC across all stores/weeks.
    Other columns are forward/backward filled or set to defaults.
    """
    min_week = df["WEEK"].min()
    max_week = df["WEEK"].max()
    all_weeks = set(range(min_week, max_week + 1))
    n_active = len(all_weeks)

    if n_active == 0:
        return df

    # Precompute median price per UPC
    median_price_by_upc = df.groupby("UPC")["PRICE"].median()

    fill_rows = []
    for (store, upc), group in df.groupby(["STORE", "UPC"]):
        present_weeks = set(group["WEEK"])
        missing_weeks = all_weeks - present_weeks
        coverage = len(present_weeks) / n_active

        if coverage >= 0.90:
            continue

        # Get a template row for non-movement columns
        template = group.iloc[0].to_dict()
        median_price = median_price_by_upc.get(upc, template["PRICE"])

        for week in missing_weeks:
            row = template.copy()
            row["STORE"] = store
            row["UPC"] = upc
            row["WEEK"] = week
            row["MOVE"] = 0
            row["QTY"] = 1
            row["PRICE"] = median_price
            row["SALE"] = ""
            row["PROFIT"] = template["PROFIT"]
            row["OK"] = 1
            fill_rows.append(row)

    if fill_rows:
        fill_df = pd.DataFrame(fill_rows)
        df = pd.concat([df, fill_df], ignore_index=True)

    return df
