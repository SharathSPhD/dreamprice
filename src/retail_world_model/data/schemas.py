"""Pydantic v2 schemas for Dominick's dataset rows."""

from pydantic import BaseModel, ConfigDict


class MovementRow(BaseModel):
    """Raw movement file row."""

    STORE: int
    UPC: int
    WEEK: int
    MOVE: int
    QTY: int
    PRICE: float
    SALE: str
    PROFIT: float
    OK: int


class UPCRow(BaseModel):
    """UPC metadata row."""

    COM_CODE: int
    UPC: int
    DESCRIP: str
    SIZE: str
    CASE: int
    NITEM: int


class StoreDemoRow(BaseModel):
    """Store demographics row (300+ columns)."""

    STORE: int
    model_config = ConfigDict(extra="allow")


class ProcessedTuple(BaseModel):
    """Processed observation tuple after cleaning and feature engineering."""

    store: int
    upc: int
    week: int
    unit_price: float
    cost: float
    move: int
    on_promotion: bool
    discount_depth: float
    hausman_iv: float
    model_config = ConfigDict(extra="allow")
