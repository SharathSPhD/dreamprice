"""Load and clean Dominick's Finer Foods scanner dataset."""

import os
from pathlib import Path

import pandas as pd

STORE_DEMO_KEEP_COLS = [
    "STORE",
    "INCOME",
    "EDUC",
    "ETHNIC",
    "HSIZEAVG",
    "SSTRDIST",
    "SSTRVOL",
    "CPDIST5",
    "CPWVOL5",
]


def load_movement(
    path: Path | str,
    store_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Load movement CSV. Drop PRICE_HEX, PROFIT_HEX. Drop OK==0 or PRICE<=0.

    Args:
        path: Path to movement CSV.
        store_ids: If provided, only load rows for these store IDs.
    """
    df = pd.read_csv(path)
    hex_cols = [c for c in ("PRICE_HEX", "PROFIT_HEX") if c in df.columns]
    if hex_cols:
        df = df.drop(columns=hex_cols)
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0)]
    if store_ids is not None:
        df = df[df["STORE"].isin(store_ids)]
    return df.reset_index(drop=True)


def load_upc(path: Path | str) -> pd.DataFrame:
    """Load UPC metadata CSV."""
    return pd.read_csv(path)


def load_store_demo(
    path: Path | str,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load store demographics CSV, optionally keeping only needed columns."""
    if keep_cols is not None:
        present = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in keep_cols if c in present]
        return pd.read_csv(path, usecols=usecols)
    return pd.read_csv(path)


def load_category(
    movement_path: Path | str,
    upc_path: Path | str,
    demo_path: Path | str,
    store_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Merge movement, UPC, and demo data. Apply cleaning.

    Args:
        movement_path: Path to movement CSV (e.g. wcso.csv).
        upc_path: Path to UPC metadata CSV.
        demo_path: Path to store demographics CSV.
        store_ids: If provided, filter to these stores early (before merge)
            to avoid loading the full 7M-row x 510-column merged DataFrame.
    """
    movement = load_movement(movement_path, store_ids=store_ids)
    upc = load_upc(upc_path)
    demo = load_store_demo(demo_path, keep_cols=STORE_DEMO_KEEP_COLS)

    df = movement.merge(upc, on="UPC", how="left")
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
