"""JSON envelope helpers for LLM-visible tool summaries.

We use a lightweight `{ "thinking": "...", "answer": ... }` shape so that:
- the next LLM step can reliably read prior decisions/actions without relying on dev-only diagnostics
- developers can inspect the same content in traces/logs

Note: "thinking" should stay brief and high-level (not step-by-step chain-of-thought).
"""
import json
from typing import Any, Dict, Optional

from core.utils.json_extract import safe_json_loads


def build_llm_envelope(*, thinking: str, answer: Any, extra: Optional[Dict[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {
        "thinking": str(thinking or "").strip(),
        "answer": answer,
    }
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    return json.dumps(payload, ensure_ascii=False)


def try_parse_llm_envelope(raw: str) -> Optional[Dict[str, Any]]:
    """Return parsed envelope if the input looks like our `{thinking, answer}` JSON."""

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    obj = safe_json_loads(text, expected="dict")
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None
    if "thinking" not in obj or "answer" not in obj:
        return None
    return obj


def compact_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[: max(0, max_chars - 1)].rstrip() + "…"
    return text
