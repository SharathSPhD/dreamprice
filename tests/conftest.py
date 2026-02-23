"""Shared test fixtures for DreamPrice test suite."""

import numpy as np
import pandas as pd
import pytest

STORE_IDS = [2, 3, 4]
UPCS = [1000000001, 1000000002, 1000000003, 1000000004, 1000000005]
WEEKS = list(range(1, 21))  # 20 weeks


@pytest.fixture
def mini_movement_df() -> pd.DataFrame:
    """Movement DataFrame: 3 stores x 5 UPCs x 20 weeks = 300 rows.

    Realistic PRICE/MOVE/QTY/PROFIT/SALE/OK values for canned soup.
    Includes PRICE_HEX and PROFIT_HEX columns that should be dropped.
    Includes some OK==0 and PRICE<=0 rows for filtering tests.
    """
    rng = np.random.RandomState(42)
    rows = []
    for store in STORE_IDS:
        for upc in UPCS:
            base_price = 1.0 + rng.rand() * 2.0  # $1-$3
            for week in WEEKS:
                price = base_price * (1.0 + 0.1 * rng.randn())
                price = max(price, 0.5)
                move = max(0, int(rng.poisson(20)))
                qty = 1
                profit = 20.0 + 10.0 * rng.randn()  # margin %
                sale_options = ["", "B", "C", "S"]
                sale = rng.choice(sale_options, p=[0.7, 0.1, 0.1, 0.1])
                ok = 1
                rows.append(
                    {
                        "STORE": store,
                        "UPC": upc,
                        "WEEK": week,
                        "MOVE": move,
                        "QTY": qty,
                        "PRICE": round(price, 2),
                        "SALE": sale,
                        "PROFIT": round(profit, 2),
                        "OK": ok,
                        "PRICE_HEX": "3FF0000000000000",
                        "PROFIT_HEX": "4034000000000000",
                    }
                )

    df = pd.DataFrame(rows)
    # Add some bad rows for filtering tests
    bad_rows = pd.DataFrame(
        [
            {
                "STORE": 2,
                "UPC": UPCS[0],
                "WEEK": 21,
                "MOVE": 5,
                "QTY": 1,
                "PRICE": 1.50,
                "SALE": "",
                "PROFIT": 20.0,
                "OK": 0,
                "PRICE_HEX": "0",
                "PROFIT_HEX": "0",
            },
            {
                "STORE": 3,
                "UPC": UPCS[1],
                "WEEK": 21,
                "MOVE": 3,
                "QTY": 1,
                "PRICE": 0.0,
                "SALE": "",
                "PROFIT": 15.0,
                "OK": 1,
                "PRICE_HEX": "0",
                "PROFIT_HEX": "0",
            },
            {
                "STORE": 4,
                "UPC": UPCS[2],
                "WEEK": 21,
                "MOVE": 2,
                "QTY": 1,
                "PRICE": -1.0,
                "SALE": "",
                "PROFIT": 10.0,
                "OK": 1,
                "PRICE_HEX": "0",
                "PROFIT_HEX": "0",
            },
        ]
    )
    return pd.concat([df, bad_rows], ignore_index=True)


@pytest.fixture
def mini_upc_df() -> pd.DataFrame:
    """UPC metadata DataFrame for the 5 test UPCs."""
    return pd.DataFrame(
        {
            "COM_CODE": [324, 324, 324, 325, 325],
            "UPC": UPCS,
            "DESCRIP": [
                "CAMPB CHICKEN NOODL",
                "CAMPB TOMATO SOUP",
                "PROGRESSO MINESTRON",
                "CAMPB CREAM MUSHRM",
                "PROGRESSO LENTIL",
            ],
            "SIZE": ["10.75 OZ", "10.75 OZ", "19 OZ", "10.75 OZ", "19 OZ"],
            "CASE": [12, 12, 12, 12, 12],
            "NITEM": [100001, 100002, 100003, 100004, 100005],
        }
    )


@pytest.fixture
def mini_demo_df() -> pd.DataFrame:
    """Store demographics DataFrame for 3 stores."""
    return pd.DataFrame(
        {
            "STORE": STORE_IDS,
            "INCOME": [10.2, 10.5, 9.8],
            "EDUC": [0.25, 0.30, 0.15],
            "ETHNIC": [0.10, 0.20, 0.35],
            "HSIZEAVG": [2.5, 2.8, 3.1],
            "AGE9": [0.12, 0.10, 0.14],
            "AGE60": [0.20, 0.25, 0.18],
            "WORKWOM": [0.30, 0.28, 0.32],
            "SSTRDIST": [2.1, 3.5, 1.8],
            "SSTRVOL": [1.1, 0.9, 1.3],
            "CPDIST5": [1.9, 2.2, 1.5],
            "CPWVOL5": [0.3, 0.4, 0.5],
            "ZONE": [1, 2, 3],
        }
    )


@pytest.fixture
def clean_movement_df(mini_movement_df: pd.DataFrame) -> pd.DataFrame:
    """Movement DataFrame after cleaning: no hex cols, no bad rows."""
    df = mini_movement_df.copy()
    df = df.drop(columns=["PRICE_HEX", "PROFIT_HEX"])
    df = df[(df["OK"] == 1) & (df["PRICE"] > 0)]
    return df.reset_index(drop=True)
