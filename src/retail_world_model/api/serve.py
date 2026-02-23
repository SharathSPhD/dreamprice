"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import random
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from retail_world_model.api.batching import DynamicBatcher
from retail_world_model.api.routes import health, pricing, stream
from retail_world_model.api.schemas import PricingRequest, PricingResponse

# ---------------------------------------------------------------------------
# Stub model functions (used when no real checkpoint is provided)
# ---------------------------------------------------------------------------


async def _stub_batch_fn(requests: list[Any]) -> list[PricingResponse]:
    """Process a batch of pricing requests with stub predictions."""
    results: list[PricingResponse] = []
    for req in requests:
        n = len(req.upc_ids)
        results.append(
            PricingResponse(
                recommended_prices=[
                    p * (0.95 + 0.1 * random.random()) for p in req.current_prices
                ],
                expected_units=[random.uniform(10, 100) for _ in range(n)],
                expected_profit=random.uniform(50, 500),
                uncertainty_bounds=[(p * 0.9, p * 1.1) for p in req.current_prices],
            )
        )
    return results


async def _stub_model_fn(request: PricingRequest) -> PricingResponse:
    """Single-request stub for /imagine."""
    return (await _stub_batch_fn([request]))[0]


async def _stub_stream_fn(request: PricingRequest) -> AsyncGenerator[dict[str, Any], None]:
    """Yield step-by-step stub imagination results."""
    n = len(request.upc_ids)
    for step in range(request.horizon):
        yield {
            "step": step,
            "prices": [p * (0.95 + 0.1 * random.random()) for p in request.current_prices],
            "expected_units": [random.uniform(10, 100) for _ in range(n)],
            "expected_profit": random.uniform(50, 500),
        }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(model_path: str | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    model_path:
        Path to a saved model checkpoint.  When *None* the app starts with
        stub functions so that the API surface can be tested without a
        trained model.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # --- startup ---
        if model_path is not None:
            # Future: load real model here
            app.state.model_loaded = True
            app.state.device = "cuda"
        else:
            app.state.model_loaded = False
            app.state.device = "cpu"

        app.state.model_name = "dreamprice"
        app.state.model_version = "0.1.0"
        app.state.categories = ["cso", "ber", "sdr"]

        # Wire up model functions (stubs when no checkpoint).
        batch_fn = _stub_batch_fn
        app.state.model_fn = _stub_model_fn
        app.state.stream_fn = _stub_stream_fn

        batcher = DynamicBatcher(
            process_fn=batch_fn,
            max_batch_size=8,
            max_wait_ms=50.0,
        )
        app.state.batcher = batcher
        await batcher.start()

        yield

        # --- shutdown ---
        await batcher.stop()

    app = FastAPI(
        title="DreamPrice API",
        description="Learned world model for retail pricing recommendations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(pricing.router)
    app.include_router(stream.router)

    return app
