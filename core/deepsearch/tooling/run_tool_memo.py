"""Run-scoped tool memoization for DeepSearch graph reasoning.

Goal: reduce repeated identical tool calls within a single DeepSearch run
without introducing cross-run/global state or changing correctness.

Policy:
- Cache is run-scoped and bounded (LRU).
- Only cache explicitly-allowed tool names/prefixes (config-driven).
- Key includes owner scope + file_scope hints + tool_args (stable JSON).
"""
import json
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from config.core.deepsearch import runtime_cache_defaults
from encapsulation.data_model.deepsearch import ToolResultPayload


def _stable_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return json.dumps(str(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunToolMemoStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class MemoEntry:
    result: ToolResultPayload
    produced_evidence_ids: tuple[str, ...]


class RunToolMemoizer:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_entries: int | None = None,
        include_prefixes: tuple[str, ...] | None = None,
        exclude_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        if enabled is None:
            enabled = bool(getattr(runtime_cache_defaults, "DEFAULT_RUN_TOOL_MEMO_ENABLED", True))
        if max_entries is None:
            max_entries = int(getattr(runtime_cache_defaults, "DEFAULT_RUN_TOOL_MEMO_MAX_ENTRIES", 512))
        if include_prefixes is None:
            include_prefixes = tuple(getattr(runtime_cache_defaults, "DEFAULT_RUN_TOOL_MEMO_INCLUDE_PREFIXES", ()))
        if exclude_prefixes is None:
            exclude_prefixes = tuple(getattr(runtime_cache_defaults, "DEFAULT_RUN_TOOL_MEMO_EXCLUDE_PREFIXES", ()))

        self._enabled = bool(enabled)
        self._max_entries = max(0, int(max_entries))
        self._include = tuple(str(x) for x in include_prefixes if str(x).strip())
        self._exclude = tuple(str(x) for x in exclude_prefixes if str(x).strip())

        self._lru: OrderedDict[str, MemoEntry] = OrderedDict()
        self._stats = RunToolMemoStats()

    def enabled(self) -> bool:
        return bool(self._enabled and self._max_entries > 0)

    def stats(self) -> dict[str, int]:
        s = self._stats
        return {
            "enabled": int(self.enabled()),
            "max_entries": int(self._max_entries),
            "entries": len(self._lru),
            "hits": int(s.hits),
            "misses": int(s.misses),
            "stores": int(s.stores),
            "evictions": int(s.evictions),
            "skipped": int(s.skipped),
        }

    def _bump(self, **kwargs: int) -> None:
        s = self._stats
        self._stats = RunToolMemoStats(
            hits=s.hits + int(kwargs.get("hits", 0)),
            misses=s.misses + int(kwargs.get("misses", 0)),
            stores=s.stores + int(kwargs.get("stores", 0)),
            evictions=s.evictions + int(kwargs.get("evictions", 0)),
            skipped=s.skipped + int(kwargs.get("skipped", 0)),
        )

    def is_cacheable(self, tool_name: str) -> bool:
        if not self.enabled():
            return False
        name = str(tool_name or "").strip()
        if not name:
            return False
        for prefix in self._exclude:
            if name == prefix or name.startswith(prefix):
                return False
        if not self._include:
            return False
        for prefix in self._include:
            if name == prefix or name.startswith(prefix):
                return True
        return False

    def make_key(
        self,
        *,
        tool_name: str,
        owner_scope_id: str,
        tool_args: dict[str, Any] | None,
        file_scope_hint: dict[str, Any] | None,
    ) -> str:
        payload = {
            "tool": str(tool_name or "").strip(),
            "owner": str(owner_scope_id or "").strip(),
            "tool_args": tool_args or {},
            "file_scope_hint": file_scope_hint or {},
        }
        # Hash to keep keys short and stable even when args are large.
        return _sha(_stable_json(payload))

    def get(self, key: str) -> Optional[MemoEntry]:
        if not self.enabled():
            self._bump(skipped=1)
            return None
        token = str(key or "").strip()
        if not token:
            self._bump(skipped=1)
            return None
        hit = self._lru.get(token)
        if hit is None:
            self._bump(misses=1)
            return None
        self._lru.move_to_end(token, last=True)
        self._bump(hits=1)
        return hit

    def put(self, key: str, entry: MemoEntry) -> None:
        if not self.enabled():
            self._bump(skipped=1)
            return
        token = str(key or "").strip()
        if not token:
            self._bump(skipped=1)
            return
        self._lru[token] = entry
        self._lru.move_to_end(token, last=True)
        self._bump(stores=1)

        while len(self._lru) > self._max_entries:
            self._lru.popitem(last=False)
            self._bump(evictions=1)

