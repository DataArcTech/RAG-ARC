"""Trace utilities for DeepSearch.

This package provides an infra-neutral trace event interface that allows the
application layer (API/Celery/CLI) to publish user-visible progress such as:
thinking → planning → tool calls → tool results → reflections → final report.
"""

from .context import (
    TraceEmitter,
    TraceEvent,
    TraceTag,
    emit_trace,
    get_trace_emitter,
    reset_trace_emitter,
    set_trace_emitter,
)

__all__ = [
    "TraceEmitter",
    "TraceEvent",
    "TraceTag",
    "emit_trace",
    "get_trace_emitter",
    "set_trace_emitter",
    "reset_trace_emitter",
]

