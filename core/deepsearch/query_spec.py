"""QuerySpec: lightweight pre-agent query analysis for DeepSearch.

Classifies whether the question needs deep document search and extracts
retrieval seed terms (BM25 + regex). Input is ONLY (question, target_langs).
"""

import os
from typing import Any, Dict, List

from core.deepsearch.utils.llm_json import call_llm_json_with_retry
from core.prompts.deepsearch.query_spec import (
    QUERY_SPEC_MULTILANG_SUFFIX_EN,
    QUERY_SPEC_SYSTEM_PROMPT_EN,
    QUERY_SPEC_USER_PROMPT_TEMPLATE_EN,
)


def _get_target_langs() -> List[str]:
    """Read target languages from env QUERY_VARIANTS_LANGS."""
    raw = os.getenv("QUERY_VARIANTS_LANGS", "")
    if not raw.strip():
        return []
    return [lang.strip() for lang in raw.split(",") if lang.strip()]


def _coerce_query_spec(raw: Dict[str, Any], *, question: str) -> Dict[str, Any]:
    """Coerce raw LLM output into a stable QuerySpec shape."""
    spec: Dict[str, Any] = {}

    report_needed = raw.get("report_needed")
    spec["report_needed"] = bool(report_needed) if report_needed is not None else True

    style = str(raw.get("report_style") or "").strip().lower()
    if style not in {"deepsearch", "research"}:
        style = "deepsearch"
    spec["report_style"] = style

    bm25_raw = raw.get("bm25_terms")
    if isinstance(bm25_raw, list):
        spec["bm25_terms"] = [str(t).strip() for t in bm25_raw if str(t).strip()][:10]
    else:
        spec["bm25_terms"] = []

    regex_raw = raw.get("regex_patterns")
    if isinstance(regex_raw, list):
        spec["regex_patterns"] = [str(p).strip() for p in regex_raw if str(p).strip()][:5]
    else:
        spec["regex_patterns"] = []

    by_lang = raw.get("bm25_terms_by_lang")
    if isinstance(by_lang, dict):
        spec["bm25_terms_by_lang"] = {
            str(k).strip(): [str(t).strip() for t in v if str(t).strip()][:10]
            for k, v in by_lang.items()
            if isinstance(v, list) and str(k).strip()
        }
    else:
        spec["bm25_terms_by_lang"] = {}

    rp_by_lang = raw.get("regex_patterns_by_lang")
    if isinstance(rp_by_lang, dict):
        spec["regex_patterns_by_lang"] = {
            str(k).strip(): [str(p).strip() for p in v if str(p).strip()][:5]
            for k, v in rp_by_lang.items()
            if isinstance(v, list) and str(k).strip()
        }
    else:
        spec["regex_patterns_by_lang"] = {}

    spec["reasoning"] = str(raw.get("reasoning") or "").strip()

    return spec


async def generate_query_spec(
    *,
    llm_connector: Any,
    question: str,
    target_langs: List[str] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
    attempts: int = 2,
) -> Dict[str, Any]:
    """Generate a QuerySpec from a user question using a lightweight LLM call."""
    if llm_connector is None:
        raise RuntimeError("QuerySpec generation requires an LLM connector")

    q = str(question or "").strip()
    if not q:
        raise ValueError("QuerySpec generation requires a non-empty question")

    if target_langs is None:
        target_langs = _get_target_langs()

    langs_str = ", ".join(target_langs) if target_langs else "auto-detect"
    user_content = QUERY_SPEC_USER_PROMPT_TEMPLATE_EN.format(question=q, target_langs=langs_str)
    if len(target_langs) > 1:
        user_content += QUERY_SPEC_MULTILANG_SUFFIX_EN

    messages = [
        {"role": "system", "content": QUERY_SPEC_SYSTEM_PROMPT_EN},
        {"role": "user", "content": user_content},
    ]

    llm_kwargs: Dict[str, Any] = {}
    if isinstance(model_name, str) and model_name.strip():
        llm_kwargs["model"] = model_name.strip()

    raw = await call_llm_json_with_retry(
        llm_connector=llm_connector,
        messages=messages,
        expected="object",
        temperature=temperature,
        max_tokens=max_tokens,
        attempts=attempts,
        llm_kwargs=llm_kwargs or None,
    )
    if not isinstance(raw, dict):
        raw = {}
    return _coerce_query_spec(raw, question=q)


__all__ = ["generate_query_spec", "_coerce_query_spec", "_get_target_langs"]
