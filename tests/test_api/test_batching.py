"""Tests for the DynamicBatcher."""

from __future__ import annotations

import asyncio
import time

import pytest

from retail_world_model.api.batching import DynamicBatcher


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _echo_batch(items: list[int]) -> list[int]:
    """Trivial batch processor that returns inputs as-is."""
    return items


async def _slow_echo_batch(items: list[int]) -> list[int]:
    await asyncio.sleep(0.01)
    return [x * 2 for x in items]


@pytest.mark.asyncio
async def test_single_request():
    batcher = DynamicBatcher(_echo_batch, max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    result = await batcher.submit(42)
    assert result == 42
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_collects_multiple_requests():
    collected_sizes: list[int] = []

    async def tracking_batch(items: list[int]) -> list[int]:
        collected_sizes.append(len(items))
        return items

    batcher = DynamicBatcher(tracking_batch, max_batch_size=4, max_wait_ms=500)
    await batcher.start()

    # Submit 4 requests concurrently -- should be batched together.
    results = await asyncio.gather(*(batcher.submit(i) for i in range(4)))
    assert list(results) == [0, 1, 2, 3]
    assert any(s > 1 for s in collected_sizes), "Expected at least one batch with >1 items"
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_fires_on_timeout():
    """A single request should be processed after max_wait_ms even without
    reaching max_batch_size."""
    batcher = DynamicBatcher(_echo_batch, max_batch_size=8, max_wait_ms=30)
    await batcher.start()

    start = time.monotonic()
    result = await batcher.submit(99)
    elapsed_ms = (time.monotonic() - start) * 1000

    assert result == 99
    # Should fire within roughly max_wait_ms (with some tolerance).
    assert elapsed_ms < 200, f"Took too long: {elapsed_ms:.0f}ms"
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_fires_at_max_size():
    collected_sizes: list[int] = []

    async def tracking_batch(items: list[int]) -> list[int]:
        collected_sizes.append(len(items))
        return items

    batcher = DynamicBatcher(tracking_batch, max_batch_size=3, max_wait_ms=5000)
    await batcher.start()

    results = await asyncio.gather(*(batcher.submit(i) for i in range(3)))
    assert list(results) == [0, 1, 2]
    # The batch should have fired because we hit max_batch_size, not timeout.
    assert 3 in collected_sizes
    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_propagates_exceptions():
    async def failing_batch(items: list[int]) -> list[int]:
        raise ValueError("boom")

    batcher = DynamicBatcher(failing_batch, max_batch_size=8, max_wait_ms=30)
    await batcher.start()

    with pytest.raises(ValueError, match="boom"):
        await batcher.submit(1)

    await batcher.stop()


@pytest.mark.asyncio
async def test_batch_processes_results_correctly():
    batcher = DynamicBatcher(_slow_echo_batch, max_batch_size=8, max_wait_ms=50)
    await batcher.start()

    results = await asyncio.gather(*(batcher.submit(i) for i in range(5)))
    assert list(results) == [0, 2, 4, 6, 8]
    await batcher.stop()
