"""Concurrency helpers for GraphDeepSearchAdapter usage."""
import asyncio
import os
import weakref
from contextlib import asynccontextmanager

from core.graph_adapter.base import GraphDeepSearchAdapter

_ADAPTER_LOCKS: "weakref.WeakKeyDictionary[GraphDeepSearchAdapter, asyncio.Lock]" = weakref.WeakKeyDictionary()

_FORCE_LOCK_ENV = "DEEPSEARCH_FORCE_ADAPTER_LOCK"


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    value = str(raw).strip().lower()
    return value in {"1", "true", "yes", "on"}


def adapter_supports_concurrency(adapter: GraphDeepSearchAdapter) -> bool:
    """Best-effort check for whether an adapter is safe to call concurrently.

    Default is conservative (serialized) unless the adapter explicitly opts in:
    - attribute `supports_concurrent_calls = True`, or
    - metadata capability metrics/extra containing `concurrency_safe: True`.

    Set `DEEPSEARCH_FORCE_ADAPTER_LOCK=1` to force serialization for safety/debugging.
    """

    if _env_truthy(_FORCE_LOCK_ENV):
        return False

    try:
        flag = getattr(adapter, "supports_concurrent_calls", None)
        if isinstance(flag, bool):
            return flag
    except Exception:
        return False

    try:
        metadata = adapter.metadata()
    except Exception:
        return False

    capabilities = getattr(metadata, "capabilities", None) or ()
    for cap in capabilities:
        metrics = getattr(cap, "metrics", None) or {}
        extra = getattr(cap, "extra", None) or {}
        if metrics.get("concurrency_safe") is True or extra.get("concurrency_safe") is True:
            return True
    return False


def adapter_requires_lock(adapter: GraphDeepSearchAdapter) -> bool:
    """Return True when adapter calls should be serialized."""

    return not adapter_supports_concurrency(adapter)


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

    if adapter_requires_lock(adapter):
        lock = adapter_lock(adapter)
        async with lock:
            yield
        return
    yield
