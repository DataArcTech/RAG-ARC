"""Debug dump utilities for capturing LLM prompts/responses during DeepSearch runs.

This is opt-in and controlled by environment variables (no runtime behavior changes by default).

Env:
- DEEPSEARCH_LLM_DUMP_PATH: when set, append JSONL events (one per LLM call).
"""
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_path() -> Optional[Path]:
    token = str(os.getenv("DEEPSEARCH_LLM_DUMP_PATH", "") or "").strip()
    if not token:
        return None
    try:
        return Path(token)
    except Exception:
        return None


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
    """Append an event as a single JSONL line when DEEPSEARCH_LLM_DUMP_PATH is set."""

    path = _coerce_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event or {})
        payload.setdefault("ts_utc", _utc_now_iso())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(payload), ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Debug-only; never break the main pipeline.
        return

