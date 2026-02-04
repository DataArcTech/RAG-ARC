"""LLM-driven query variants for retrieval (domain-agnostic).

Design note:
- This module must be safe in offline/unit-test environments. It will only call
  an LLM when an explicit `llm_connector` is provided.
"""
import logging
import json
from typing import Optional

from config.query_variants import (
    QUERY_VARIANTS_ENABLED,
    QUERY_VARIANTS_LANGS,
    QUERY_VARIANTS_MAX,
    QUERY_VARIANTS_ZH_HANS_HANT_ENABLED,
)
from core.prompts.query_variants import (
    QUERY_VARIANTS_SYSTEM_PROMPT,
    QUERY_VARIANTS_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _low_cost_model_name(llm_connector) -> Optional[str]:
    cfg = getattr(llm_connector, "config", None)
    token = getattr(cfg, "low_cost_model_name", None) if cfg is not None else None
    token = str(token or "").strip()
    return token or None


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        payload = json.loads(snippet)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _llm_rewrite_variants(llm_connector, query: str, langs: list[str]) -> dict[str, str]:
    if llm_connector is None:
        return {}

    payload = {"query": query, "target_langs": list(langs)}
    messages = [
        {"role": "system", "content": QUERY_VARIANTS_SYSTEM_PROMPT},
        {"role": "user", "content": QUERY_VARIANTS_USER_PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False))},
    ]
    kwargs = {"temperature": 0.0, "max_tokens": 512}
    low_cost = _low_cost_model_name(llm_connector)
    if low_cost:
        kwargs["model"] = low_cost
    try:
        chat = getattr(llm_connector, "chat", None)
        if callable(chat):
            response = chat(messages, **kwargs)
        else:
            # Async-only connectors are ignored here (keep it deterministic + offline-safe).
            return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query variants LLM call failed; using base query only. error=%s", exc)
        return {}
    parsed = _extract_json_object(str(response or "").strip())
    if not parsed:
        return {}
    out: dict[str, str] = {}
    for key in langs:
        token = str(parsed.get(key) or "").strip()
        if token:
            out[key] = token
    return out


def generate_query_variants(query: str, *, llm_connector=None) -> list[str]:
    """
    Generate a small set of deterministic query variants.

    Always includes the original query (first).
    """
    base = str(query or "").strip()
    if not base:
        return []

    if not QUERY_VARIANTS_ENABLED:
        return [base]

    variants: list[str] = [base]

    langs: list[str] = []
    for lang in QUERY_VARIANTS_LANGS:
        token = str(lang or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in {"zh-hans", "zh-hant"} and not QUERY_VARIANTS_ZH_HANS_HANT_ENABLED:
            continue
        langs.append(token)

    if langs and llm_connector is not None:
        rewritten = _llm_rewrite_variants(llm_connector, base, langs)
        for lang in langs:
            candidate = str(rewritten.get(lang) or "").strip()
            if candidate:
                variants.append(candidate)

    variants = _dedupe_preserve_order(variants)
    desired_max = int(QUERY_VARIANTS_MAX)
    minimum = 1 + len(langs)
    if desired_max < minimum:
        desired_max = minimum
    return variants[: desired_max]


__all__ = ["generate_query_variants"]
