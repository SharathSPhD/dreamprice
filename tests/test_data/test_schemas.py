"""Tests for Pydantic data schemas."""

from retail_world_model.data.schemas import (
    MovementRow,
    ProcessedTuple,
    StoreDemoRow,
    UPCRow,
)


class TestMovementRow:
    def test_valid_row(self):
        row = MovementRow(
            STORE=2, UPC=1000000001, WEEK=1, MOVE=20, QTY=1,
            PRICE=1.59, SALE="", PROFIT=22.5, OK=1,
        )
        assert row.STORE == 2
        assert row.PRICE == 1.59

    def test_sale_codes(self):
        for code in ["", "B", "C", "S"]:
            row = MovementRow(
                STORE=2, UPC=1, WEEK=1, MOVE=10, QTY=1,
                PRICE=1.0, SALE=code, PROFIT=10.0, OK=1,
            )
            assert row.SALE == code

    def test_zero_move_allowed(self):
        row = MovementRow(
            STORE=2, UPC=1, WEEK=1, MOVE=0, QTY=1,
            PRICE=1.0, SALE="", PROFIT=10.0, OK=1,
        )
        assert row.MOVE == 0


class TestUPCRow:
    def test_valid_row(self):
        row = UPCRow(
            COM_CODE=324, UPC=1000000001,
            DESCRIP="CAMPB CHICKEN NOODL", SIZE="10.75 OZ",
            CASE=12, NITEM=100001,
        )
        assert row.DESCRIP == "CAMPB CHICKEN NOODL"


class TestStoreDemoRow:
    def test_valid_row(self):
        row = StoreDemoRow(
            STORE=2, INCOME=10.2, EDUC=0.25, ETHNIC=0.10,
            HSIZEAVG=2.5, SSTRDIST=2.1, SSTRVOL=1.1,
            CPDIST5=1.9, CPWVOL5=0.3,
        )
        assert row.STORE == 2


class TestProcessedTuple:
    def test_valid_tuple(self):
        pt = ProcessedTuple(
            store=2, upc=1000000001, week=1,
            unit_price=1.59, cost=1.24, move=20,
            on_promotion=False, discount_depth=0.0,
            hausman_iv=0.45,
        )
        assert pt.unit_price == 1.59
        assert pt.on_promotion is False
