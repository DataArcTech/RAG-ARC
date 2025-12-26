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
from .protocol import TRACE_FORMAT_VERSION, TRACE_SCHEMA_VERSION, with_trace_protocol

__all__ = [
    "TraceEmitter",
    "TraceEvent",
    "TraceTag",
    "TRACE_SCHEMA_VERSION",
    "TRACE_FORMAT_VERSION",
    "with_trace_protocol",
    "emit_trace",
    "get_trace_emitter",
    "set_trace_emitter",
    "reset_trace_emitter",
]
