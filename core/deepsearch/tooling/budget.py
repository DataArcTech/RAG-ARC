"""Tool-call budget tracking for DeepSearch.

This module provides an explicit, configurable upper bound on total tool invocations
performed during one DeepSearch run, and exposes the remaining budget to LLM prompts.

Design goals:
- No hidden global mutable state: use a ContextVar holding a *mutable* budget object
  that is set/reset per run and shared across asyncio tasks.
- Observable: attach snapshots (used/remaining/max) to tool diagnostics and graph_context.metadata.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from contextvars import ContextVar, Token


class ToolBudgetExceededError(RuntimeError):
    def __init__(self, *, tool_name: str, max_calls_total: int, used_calls: int) -> None:
        remaining = max(0, int(max_calls_total) - int(used_calls))
        super().__init__(
            f"Tool budget exceeded before calling '{tool_name}': "
            f"used_calls={used_calls}, max_calls_total={max_calls_total}, remaining_calls={remaining}"
        )
        self.tool_name = tool_name
        self.max_calls_total = int(max_calls_total)
        self.used_calls = int(used_calls)
        self.remaining_calls = remaining


@dataclass
class ToolBudget:
    """Mutable per-run budget shared across async tasks."""

    max_calls_total: int
    used_calls: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def snapshot(self) -> Dict[str, int]:
        max_calls = max(0, int(self.max_calls_total))
        used = max(0, int(self.used_calls))
        remaining = max(0, max_calls - used) if max_calls > 0 else 0
        return {
            "max_calls_total": max_calls,
            "used_calls": used,
            "remaining_calls": remaining,
        }

    async def claim(self, *, tool_name: str, n: int = 1) -> Dict[str, int]:
        """Consume budget for an impending tool invocation.

        Returns updated snapshot after the claim.
        """

        delta = max(1, int(n))
        async with self._lock:
            max_calls = max(0, int(self.max_calls_total))
            if max_calls <= 0:
                raise ToolBudgetExceededError(tool_name=tool_name, max_calls_total=max_calls, used_calls=self.used_calls)
            if self.used_calls + delta > max_calls:
                raise ToolBudgetExceededError(tool_name=tool_name, max_calls_total=max_calls, used_calls=self.used_calls)
            self.used_calls += delta
            return self.snapshot()


_TOOL_BUDGET: ContextVar[Optional[ToolBudget]] = ContextVar("deepsearch_tool_budget", default=None)


def set_tool_budget(budget: ToolBudget) -> Token[Optional[ToolBudget]]:
    return _TOOL_BUDGET.set(budget)


def reset_tool_budget(token: Token[Optional[ToolBudget]]) -> None:
    _TOOL_BUDGET.reset(token)


def get_tool_budget() -> Optional[ToolBudget]:
    return _TOOL_BUDGET.get()


def attach_tool_budget_metadata(*, graph_context: Any, snapshot: Dict[str, int]) -> None:
    """Best-effort attach tool budget snapshot to a GraphQueryContext-like object."""

    if graph_context is None:
        return
    try:
        meta = getattr(graph_context, "metadata", None)
    except Exception:
        meta = None
    if isinstance(meta, dict):
        meta.setdefault("tool_budget", {})
        if isinstance(meta.get("tool_budget"), dict):
            meta["tool_budget"].update(snapshot)
        else:
            meta["tool_budget"] = dict(snapshot)
        return
    # Fallback for dict-like payloads
    if isinstance(graph_context, dict):
        ctx_meta = graph_context.setdefault("metadata", {})
        if isinstance(ctx_meta, dict):
            bucket = ctx_meta.setdefault("tool_budget", {})
            if isinstance(bucket, dict):
                bucket.update(snapshot)
            else:
                ctx_meta["tool_budget"] = dict(snapshot)

