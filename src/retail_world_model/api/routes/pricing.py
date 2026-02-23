"""Pricing recommendation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from retail_world_model.api.schemas import PricingRequest, PricingResponse

router = APIRouter(tags=["pricing"])


@router.post("/recommend", response_model=PricingResponse)
async def recommend_prices(request: PricingRequest, raw: Request) -> PricingResponse:
    """Return price recommendations via the dynamic batcher."""
    batcher = raw.app.state.batcher
    return await batcher.submit(request)


@router.post("/imagine", response_model=PricingResponse)
async def imagine_trajectory(request: PricingRequest, raw: Request) -> PricingResponse:
    """Run a full imagination rollout and return the final recommendation."""
    model_fn = raw.app.state.model_fn
    return await model_fn(request)
