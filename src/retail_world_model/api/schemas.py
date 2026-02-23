"""Pydantic request / response schemas for the pricing API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    store_id: int
    upc_ids: list[int]
    current_prices: list[float]
    week: int
    horizon: int = Field(default=13, description="Rollout horizon in weeks (default 1 quarter)")


class PricingResponse(BaseModel):
    recommended_prices: list[float]
    expected_units: list[float]
    expected_profit: float
    uncertainty_bounds: list[tuple[float, float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    device: str
    categories: list[str]


class ImagineStepEvent(BaseModel):
    step: int
    prices: list[float]
    expected_units: list[float]
    expected_profit: float
