"""SSE streaming endpoint for step-by-step imagination rollouts."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from retail_world_model.api.schemas import PricingRequest

router = APIRouter(tags=["streaming"])


@router.post("/stream")
async def stream_recommendation(request: PricingRequest, raw: Request) -> EventSourceResponse:
    """Stream intermediate rollout states as server-sent events."""
    stream_fn = raw.app.state.stream_fn

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        async for step_data in stream_fn(request):
            yield {"data": json.dumps(step_data)}

    return EventSourceResponse(event_generator())
