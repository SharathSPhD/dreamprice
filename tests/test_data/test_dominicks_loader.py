"""Tests for Dominick's data loader."""

import pandas as pd

from retail_world_model.data.dominicks_loader import (
    load_category,
    load_movement,
    load_store_demo,
    load_upc,
)


class TestLoadMovement:
    def test_drops_hex_columns(self, tmp_path, mini_movement_df):
        path = tmp_path / "wcso.csv"
        mini_movement_df.to_csv(path, index=False)
        df = load_movement(path)
        assert "PRICE_HEX" not in df.columns
        assert "PROFIT_HEX" not in df.columns

    def test_drops_ok_zero(self, tmp_path, mini_movement_df):
        path = tmp_path / "wcso.csv"
        mini_movement_df.to_csv(path, index=False)
        df = load_movement(path)
        assert (df["OK"] == 1).all()

    def test_drops_price_lte_zero(self, tmp_path, mini_movement_df):
        path = tmp_path / "wcso.csv"
        mini_movement_df.to_csv(path, index=False)
        df = load_movement(path)
        assert (df["PRICE"] > 0).all()

    def test_output_columns(self, tmp_path, mini_movement_df):
        path = tmp_path / "wcso.csv"
        mini_movement_df.to_csv(path, index=False)
        df = load_movement(path)
        expected = {"STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"}
        assert expected.issubset(set(df.columns))

    def test_row_count(self, tmp_path, mini_movement_df):
        path = tmp_path / "wcso.csv"
        mini_movement_df.to_csv(path, index=False)
        df = load_movement(path)
        # 300 good rows, 3 bad rows dropped
        assert len(df) == 300


class TestLoadUPC:
    def test_columns(self, tmp_path, mini_upc_df):
        path = tmp_path / "upccso.csv"
        mini_upc_df.to_csv(path, index=False)
        df = load_upc(path)
        assert set(df.columns) == {"COM_CODE", "UPC", "DESCRIP", "SIZE", "CASE", "NITEM"}

    def test_row_count(self, tmp_path, mini_upc_df):
        path = tmp_path / "upccso.csv"
        mini_upc_df.to_csv(path, index=False)
        df = load_upc(path)
        assert len(df) == 5


class TestLoadStoreDemo:
    def test_columns(self, tmp_path, mini_demo_df):
        path = tmp_path / "demo.csv"
        mini_demo_df.to_csv(path, index=False)
        df = load_store_demo(path)
        assert "STORE" in df.columns

    def test_row_count(self, tmp_path, mini_demo_df):
        path = tmp_path / "demo.csv"
        mini_demo_df.to_csv(path, index=False)
        df = load_store_demo(path)
        assert len(df) == 3


class TestLoadCategory:
    def test_merges_all_sources(self, tmp_path, mini_movement_df, mini_upc_df, mini_demo_df):
        mini_movement_df.to_csv(tmp_path / "wcso.csv", index=False)
        mini_upc_df.to_csv(tmp_path / "upccso.csv", index=False)
        mini_demo_df.to_csv(tmp_path / "demo.csv", index=False)

        df = load_category(
            movement_path=tmp_path / "wcso.csv",
            upc_path=tmp_path / "upccso.csv",
            demo_path=tmp_path / "demo.csv",
        )
        # Should have movement + upc + demo columns
        assert "DESCRIP" in df.columns
        assert "INCOME" in df.columns
        assert "PRICE_HEX" not in df.columns
        assert "PROFIT_HEX" not in df.columns

    def test_no_bad_rows(self, tmp_path, mini_movement_df, mini_upc_df, mini_demo_df):
        mini_movement_df.to_csv(tmp_path / "wcso.csv", index=False)
        mini_upc_df.to_csv(tmp_path / "upccso.csv", index=False)
        mini_demo_df.to_csv(tmp_path / "demo.csv", index=False)

        df = load_category(
            movement_path=tmp_path / "wcso.csv",
            upc_path=tmp_path / "upccso.csv",
            demo_path=tmp_path / "demo.csv",
        )
        assert (df["OK"] == 1).all()
        assert (df["PRICE"] > 0).all()

    def test_zero_sales_insertion(self, tmp_path, mini_upc_df, mini_demo_df):
        """If a (store, UPC) pair is missing >10% of active weeks, zero-sales rows
        should be inserted."""
        # Create movement data with a gap: store 2, UPC 0 present in 18/20 weeks
        import numpy as np

        rng = np.random.RandomState(99)
        rows = []
        for store in [2, 3]:
            for upc in [1000000001]:
                weeks = list(range(1, 21))
                if store == 2:
                    # Remove 4 weeks (20%) to trigger zero-fill
                    weeks = [w for w in weeks if w not in [5, 10, 15, 20]]
                for week in weeks:
                    rows.append(
                        {
                            "STORE": store,
                            "UPC": upc,
                            "WEEK": week,
                            "MOVE": max(0, int(rng.poisson(20))),
                            "QTY": 1,
                            "PRICE": round(1.5 + 0.1 * rng.randn(), 2),
                            "SALE": "",
                            "PROFIT": 20.0,
                            "OK": 1,
                            "PRICE_HEX": "0",
                            "PROFIT_HEX": "0",
                        }
                    )
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / "wcso.csv", index=False)

        upc_df = mini_upc_df[mini_upc_df["UPC"] == 1000000001]
        upc_df.to_csv(tmp_path / "upccso.csv", index=False)

        demo_df = mini_demo_df[mini_demo_df["STORE"].isin([2, 3])]
        demo_df.to_csv(tmp_path / "demo.csv", index=False)

        result = load_category(
            movement_path=tmp_path / "wcso.csv",
            upc_path=tmp_path / "upccso.csv",
            demo_path=tmp_path / "demo.csv",
        )
        # Store 2 should now have 20 weeks (16 original + 4 zero-filled)
        store2 = result[result["STORE"] == 2]
        assert len(store2) == 20
        # The zero-filled rows should have MOVE==0
        filled = store2[store2["WEEK"].isin([5, 10, 15, 20])]
        assert (filled["MOVE"] == 0).all()
