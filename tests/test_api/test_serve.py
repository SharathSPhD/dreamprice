"""Tests for the FastAPI application."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from retail_world_model.api.serve import create_app


@pytest_asyncio.fixture
async def client():
    app = create_app(model_path=None)
    # Manually run the lifespan so app.state is populated.
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["device"] == "cpu"


@pytest.mark.asyncio
async def test_model_info(client: AsyncClient):
    resp = await client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "dreamprice"
    assert "cso" in data["categories"]


@pytest.mark.asyncio
async def test_recommend_returns_valid_response(client: AsyncClient):
    payload = {
        "store_id": 1,
        "upc_ids": [100, 200],
        "current_prices": [2.99, 3.49],
        "week": 100,
        "horizon": 13,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommended_prices"]) == 2
    assert len(data["expected_units"]) == 2
    assert isinstance(data["expected_profit"], float)
    assert len(data["uncertainty_bounds"]) == 2


@pytest.mark.asyncio
async def test_imagine_returns_valid_response(client: AsyncClient):
    payload = {
        "store_id": 1,
        "upc_ids": [100],
        "current_prices": [2.99],
        "week": 50,
    }
    resp = await client.post("/imagine", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommended_prices"]) == 1
    assert isinstance(data["expected_profit"], float)


@pytest.mark.asyncio
async def test_stream_returns_sse(client: AsyncClient):
    payload = {
        "store_id": 1,
        "upc_ids": [100],
        "current_prices": [2.99],
        "week": 50,
        "horizon": 3,
    }
    resp = await client.post("/stream", json=payload)
    assert resp.status_code == 200
    # SSE responses have text/event-stream content type.
    assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_recommend_validates_input(client: AsyncClient):
    # Missing required fields.
    resp = await client.post("/recommend", json={"store_id": 1})
    assert resp.status_code == 422
