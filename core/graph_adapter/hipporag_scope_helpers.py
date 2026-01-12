import os
import re
from typing import Any, List, Mapping, Optional

from core.deepsearch.utils.file_scope import FileScope
from core.prompts import build_file_scope_xlang_rewrite_prompt
from core.utils.json_extract import safe_json_loads
from config.core.deepsearch.file_scope_xlang_defaults import load_file_scope_xlang_thresholds
from config.benchmark_mode import benchmark_mode_enabled


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


def coerce_file_scope(query_options: Optional[Mapping[str, Any]]) -> FileScope:
    if not isinstance(query_options, Mapping):
        return FileScope()
    raw = query_options.get("file_scope")
    if isinstance(raw, Mapping):
        return FileScope(
            file_ids=frozenset(str(x).strip() for x in (raw.get("file_ids") or []) if str(x).strip()),
            filename_contains=tuple(str(x).strip() for x in (raw.get("filename_contains") or []) if str(x).strip()),
            source=str(raw.get("source") or "adapter_query_options").strip() or "adapter_query_options",
        )
    return FileScope()


def file_scope_xlang_retry_enabled() -> bool:
    if benchmark_mode_enabled():
        return False
    raw = os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_RETRY", "1")
    return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}


def maybe_rewrite_query_for_scope(*, llm_client: Any, query: str) -> str | None:
    chat = getattr(llm_client, "chat", None)
    if not callable(chat):
        return None

    text = str(query or "").strip()
    if not text:
        return None

    cjk = len(_CJK_RE.findall(text))
    alpha = len(_ASCII_ALPHA_RE.findall(text))
    total = max(1, len(text))
    cjk_ratio = cjk / total
    alpha_ratio = alpha / total

    thresholds = load_file_scope_xlang_thresholds()
    direction: str | None = None
    if alpha_ratio >= thresholds.alpha_ratio_to_zh_min and cjk_ratio < thresholds.cjk_ratio_to_zh_max:
        direction = "to_zh"
    elif cjk_ratio >= thresholds.cjk_ratio_to_en_min and alpha_ratio < thresholds.alpha_ratio_to_en_max:
        direction = "to_en"
    else:
        return None

    prompt = build_file_scope_xlang_rewrite_prompt(query=text)
    try:
        raw = chat([{"role": "user", "content": prompt}])
    except Exception:
        return None

    payload = safe_json_loads(str(raw or ""), expected="dict")
    if not isinstance(payload, dict):
        return None

    zh_hans = str(payload.get("zh_hans") or "").strip()
    zh_hant = str(payload.get("zh_hant") or "").strip()
    en = str(payload.get("en") or "").strip()

    additions: List[str] = []
    if direction == "to_zh":
        if zh_hans:
            additions.append(zh_hans)
        if zh_hant and zh_hant != zh_hans:
            additions.append(zh_hant)
    elif direction == "to_en" and en:
        additions.append(en)

    if not additions:
        return None
    return f"{text}\n\n" + "\n".join(additions)
