"""A small in-process TTL + LRU cache.

Why: deepsearch/rag often needs *process-local* reuse (avoid repeated heavy init / repeated
deterministic lookups) without introducing Redis IO pressure.

Non-goals:
- Cross-process persistence
- Perfect clock guarantees (we use time.monotonic for TTL)
"""
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float


class TTLRUCache(Generic[K, V]):
    """Thread-safe TTL cache with LRU eviction."""

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self._max_entries = int(max_entries)
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        self._data: "OrderedDict[K, _Entry[V]]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._data)}

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def _purge_expired_locked(self, now: float) -> None:
        # OrderedDict iterates in LRU order; we can stop early after finding a non-expired entry.
        keys_to_delete = []
        for key, entry in self._data.items():
            if entry.expires_at > now:
                break
            keys_to_delete.append(key)
        for key in keys_to_delete:
            self._data.pop(key, None)

    def get(self, key: K) -> Optional[V]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired_locked(now)
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= now:
                self._data.pop(key, None)
                self._misses += 1
                return None
            # mark as most-recently-used
            self._data.move_to_end(key, last=True)
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V) -> None:
        now = time.monotonic()
        with self._lock:
            self._purge_expired_locked(now)
            expires_at = now + self._ttl_seconds
            self._data[key] = _Entry(value=value, expires_at=expires_at)
            self._data.move_to_end(key, last=True)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

