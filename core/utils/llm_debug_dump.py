"""Debug dump utilities for capturing LLM prompts/responses during DeepSearch runs.

This is opt-in and controlled by environment variables (no runtime behavior changes by default).

Env:
- DEEPSEARCH_LLM_DUMP_PATH: when set to an `io://...` virtual directory, persist one JSON object per event
  via IOManager (no local filesystem writes in core/app layers).
"""
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from framework.virtual_paths import io_key, is_io_path

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_dir_virtual() -> Optional[str]:
    token = str(os.getenv("DEEPSEARCH_LLM_DUMP_PATH", "") or "").strip()
    if not token:
        return None
    if not is_io_path(token):
        logger.warning("DEEPSEARCH_LLM_DUMP_PATH must be an io:// virtual directory, got: %r", token)
        return None
    return token


def _jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _jsonable(v)
        return out
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    # Best-effort fallback.
    return str(obj)


def append_llm_event(event: Dict[str, Any]) -> None:
    """Persist a debug event when DEEPSEARCH_LLM_DUMP_PATH is set to an io:// virtual dir."""

    dump_dir = _dump_dir_virtual()
    if dump_dir is None:
        return
    try:
        from framework.register import Register

        io_manager = Register().get_object("io_manager")
    except Exception as exc:
        logger.warning("LLM debug dump enabled but io_manager is unavailable: %s", exc)
        return

    payload = dict(event or {})
    payload.setdefault("ts_utc", _utc_now_iso())
    run_id = str(payload.get("run_id") or "run").strip() or "run"
    llm_call_id = str(payload.get("llm_call_id") or "call").strip() or "call"
    event_name = str(payload.get("event") or "event").strip() or "event"

    try:
        root_key = io_key(dump_dir)
        namespace, prefix = (root_key.split("/", 1) + [""])[:2]
        namespace = namespace or "deepsearch_llm_dump"
        key = "/".join([p for p in [prefix, run_id, llm_call_id, f"{event_name}.json"] if p])
        io_manager.put_text(
            namespace=namespace,
            key=key,
            text=json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
            content_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to persist LLM debug dump event: %s", exc)
        return
