import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.deepsearch.trace import TraceEmitter, TraceEvent, get_trace_emitter, set_trace_emitter
from core.utils.json_safe import json_safe

logger = logging.getLogger(__name__)


class CapturingTraceEmitter(TraceEmitter):
    """Trace emitter that forwards to a delegate and captures JSON-safe events for persistence."""

    def __init__(self, *, delegate: Optional[TraceEmitter], sink: List[Dict[str, Any]]) -> None:
        self._delegate = delegate
        self._sink = sink

    async def emit(self, event: TraceEvent) -> None:
        try:
            meta = event.meta
            if is_dataclass(meta):
                meta = asdict(meta)
            payload = {
                "tag": event.tag,
                "content": str(event.content),
                "meta": json_safe(meta) if isinstance(meta, dict) else {"meta": json_safe(meta)},
            }
            self._sink.append(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to capture trace event: %s", exc, exc_info=True)

        delegate = self._delegate
        if delegate is None:
            return
        try:
            await delegate.emit(event)
        except Exception:
            return


def attach_trace_capture(*, sink: List[Dict[str, Any]] | None = None) -> Tuple[Any, List[Dict[str, Any]]]:
    """Install a capturing trace emitter and return (token, sink)."""

    events: List[Dict[str, Any]] = sink if isinstance(sink, list) else []
    delegate = get_trace_emitter()
    token = set_trace_emitter(CapturingTraceEmitter(delegate=delegate, sink=events))
    return token, events

