"""Context-scoped trace emitter for DeepSearch.

Core modules can emit trace events without depending on any transport. The
application layer is responsible for injecting a run-scoped TraceEmitter
implementation (e.g. publishing into Redis streams for SSE replay).
"""
import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol


TraceTag = Literal[
    "think",
    "write_outline",
    "tool_call",
    "tool_response",
    "write",
    "terminate",
]


@dataclass(frozen=True)
class TraceEvent:
    """A single user-visible trace event."""

    tag: TraceTag
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


class TraceEmitter(Protocol):
    """Transport-agnostic event publisher injected by the application layer."""

    async def emit(self, event: TraceEvent) -> None:
        """Publish one trace event."""


_TRACE_EMITTER: contextvars.ContextVar[Optional[TraceEmitter]] = contextvars.ContextVar(
    "deepsearch_trace_emitter",
    default=None,
)


def get_trace_emitter() -> Optional[TraceEmitter]:
    return _TRACE_EMITTER.get()


def set_trace_emitter(emitter: Optional[TraceEmitter]) -> contextvars.Token:
    return _TRACE_EMITTER.set(emitter)


def reset_trace_emitter(token: contextvars.Token) -> None:
    _TRACE_EMITTER.reset(token)


async def emit_trace(tag: TraceTag, content: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    """Emit a trace event if an emitter is configured.

    This function must never raise, as trace publishing is best-effort and must
    not break the research pipeline.
    """

    emitter = get_trace_emitter()
    if emitter is None:
        return
    try:
        await emitter.emit(
            TraceEvent(
                tag=tag,
                content=str(content),
                meta=dict(meta or {}),
            )
        )
    except Exception:
        return

