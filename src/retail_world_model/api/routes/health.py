"""Health and model-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from retail_world_model.api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    state = request.app.state
    return HealthResponse(
        status="ok",
        model_loaded=getattr(state, "model_loaded", False),
        device=getattr(state, "device", "cpu"),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    state = request.app.state
    return ModelInfoResponse(
        model_name=getattr(state, "model_name", "dreamprice"),
        version=getattr(state, "model_version", "0.1.0"),
        device=getattr(state, "device", "cpu"),
        categories=getattr(state, "categories", []),
    )
