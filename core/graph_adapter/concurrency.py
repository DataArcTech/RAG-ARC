"""Concurrency helpers for GraphDeepSearchAdapter usage."""
import asyncio
import weakref
from contextlib import asynccontextmanager

from core.graph_adapter.base import GraphDeepSearchAdapter

_ADAPTER_LOCKS: "weakref.WeakKeyDictionary[GraphDeepSearchAdapter, asyncio.Lock]" = weakref.WeakKeyDictionary()


def adapter_lock(adapter: GraphDeepSearchAdapter) -> asyncio.Lock:
    """Return a per-adapter asyncio.Lock used to serialize adapter calls."""

    lock = _ADAPTER_LOCKS.get(adapter)
    if lock is None:
        lock = asyncio.Lock()
        _ADAPTER_LOCKS[adapter] = lock
    return lock


@asynccontextmanager
async def adapter_locked(adapter: GraphDeepSearchAdapter):
    """Async context manager that guards adapter calls with the per-adapter lock."""

    lock = adapter_lock(adapter)
    async with lock:
        yield

