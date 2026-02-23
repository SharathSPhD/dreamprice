"""Asyncio queue-based dynamic batcher for inference requests."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine


class DynamicBatcher:
    """Collects incoming requests and processes them in batches.

    Fires when either ``max_batch_size`` items are queued **or**
    ``max_wait_ms`` milliseconds have elapsed since the first item
    arrived in the current batch window -- whichever comes first.
    """

    def __init__(
        self,
        process_fn: Callable[[list[Any]], Coroutine[Any, Any, list[Any]]],
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ) -> None:
        self._process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[tuple[Any, asyncio.Future[Any]]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background batch-processing loop."""
        self._task = asyncio.get_event_loop().create_task(self._batch_loop())

    async def stop(self) -> None:
        """Cancel the background loop and wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: Any) -> Any:
        """Add a request to the queue and return its result."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put((request, future))
        return await future

    async def _batch_loop(self) -> None:
        """Collect items up to max_batch_size or max_wait_ms, then process."""
        while True:
            batch: list[tuple[Any, asyncio.Future[Any]]] = []

            # Block until at least one item arrives.
            first = await self._queue.get()
            batch.append(first)

            # Collect more items up to the batch limit or timeout.
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                now = asyncio.get_event_loop().time()
                remaining = deadline - now
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Process the batch.
            requests = [req for req, _ in batch]
            futures = [fut for _, fut in batch]
            try:
                results = await self._process_fn(requests)
                for fut, result in zip(futures, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
