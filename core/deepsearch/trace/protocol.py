"""Protocol metadata helpers for DeepSearch trace/progress payloads."""
from typing import Any, Dict, Optional


TRACE_SCHEMA_VERSION = 1
TRACE_FORMAT_VERSION = 1


def with_trace_protocol(
    payload: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    schema_version: int = TRACE_SCHEMA_VERSION,
    trace_format_version: int = TRACE_FORMAT_VERSION,
) -> Dict[str, Any]:
    """Attach non-breaking protocol metadata to a payload.

    The returned object is a shallow copy of `payload`.
    """

    enriched = dict(payload or {})
    enriched.setdefault("schema_version", int(schema_version))
    enriched.setdefault("trace_format_version", int(trace_format_version))
    if run_id:
        enriched.setdefault("run_id", str(run_id))
    return enriched

