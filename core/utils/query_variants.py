"""Query variants for retrieval (domain-agnostic).

Design note:
- This module must be safe in offline/unit-test environments.
- Deterministic variants (e.g. zh-Hans/zh-Hant conversion) must work without an
  LLM. LLM-based enrichment is optional and only happens when an explicit
  `llm_connector` is provided.
"""
import logging
import json
import functools
import hashlib
from typing import Optional

from config.query_variants import (
    QUERY_VARIANTS_ENABLED,
    QUERY_VARIANTS_LANGS,
    QUERY_VARIANTS_LLM_CACHE_ENABLED,
    QUERY_VARIANTS_LLM_CACHE_MAX_ENTRIES,
    QUERY_VARIANTS_LLM_CACHE_TTL_SECONDS,
    QUERY_VARIANTS_LLM_MAX_TOKENS,
    QUERY_VARIANTS_LLM_TEMPERATURE,
    QUERY_VARIANTS_MAX,
    QUERY_VARIANTS_ZH_HANS_HANT_ENABLED,
)
from core.prompts.query_variants import (
    QUERY_VARIANTS_SYSTEM_PROMPT,
    QUERY_VARIANTS_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_OPENCC_AVAILABLE: bool | None = None
_LLM_REWRITE_CACHE = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_fingerprint() -> str:
    return _sha(QUERY_VARIANTS_SYSTEM_PROMPT + "\n" + QUERY_VARIANTS_USER_PROMPT_TEMPLATE)


def _get_llm_rewrite_cache():
    global _LLM_REWRITE_CACHE
    if _LLM_REWRITE_CACHE is not None:
        return _LLM_REWRITE_CACHE
    if not QUERY_VARIANTS_LLM_CACHE_ENABLED or int(QUERY_VARIANTS_LLM_CACHE_MAX_ENTRIES) <= 0:
        _LLM_REWRITE_CACHE = False
        return _LLM_REWRITE_CACHE
    try:
        from framework.cache import TTLRUCache

        _LLM_REWRITE_CACHE = TTLRUCache(
            max_entries=int(QUERY_VARIANTS_LLM_CACHE_MAX_ENTRIES),
            ttl_seconds=float(QUERY_VARIANTS_LLM_CACHE_TTL_SECONDS),
        )
    except Exception:  # noqa: BLE001
        _LLM_REWRITE_CACHE = False
    return _LLM_REWRITE_CACHE


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


def _llm_rewrite_variants(llm_connector, query: str, langs: list[str], *, cache_scope: str | None = None) -> dict[str, str]:
    if llm_connector is None:
        return {}

    normalized = str(query or "").strip()
    langs_key = ",".join([str(x).strip() for x in langs if str(x).strip()])
    cfg = getattr(llm_connector, "config", None)
    default_model = str(getattr(cfg, "model_name", "") or "").strip()
    low_cost = _low_cost_model_name(llm_connector)
    used_model = low_cost or default_model
    scope_key = str(cache_scope or "").strip()
    cache = _get_llm_rewrite_cache()
    cache_key = None
    # Cache is only safe when the caller provides an explicit scope (owner/tenant).
    # Without a scope, caching would risk cross-tenant query leakage.
    if cache not in (None, False) and scope_key:
        cache_key = _sha("|".join(["qv_llm", scope_key, used_model, _prompt_fingerprint(), langs_key, normalized]))
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return {str(k): str(v) for k, v in cached.items()}

    payload = {"query": query, "target_langs": list(langs)}
    messages = [
        {"role": "system", "content": QUERY_VARIANTS_SYSTEM_PROMPT},
        {"role": "user", "content": QUERY_VARIANTS_USER_PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False))},
    ]
    kwargs = {"temperature": float(QUERY_VARIANTS_LLM_TEMPERATURE), "max_tokens": int(QUERY_VARIANTS_LLM_MAX_TOKENS)}
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
    if cache_key and cache not in (None, False) and out:
        try:
            cache.set(cache_key, dict(out))
        except Exception:  # noqa: BLE001
            pass
    return out


def _detect_opencc_available() -> bool:
    global _OPENCC_AVAILABLE  # small module-level cache; deterministic + test-safe
    if _OPENCC_AVAILABLE is not None:
        return bool(_OPENCC_AVAILABLE)
    try:
        from opencc import OpenCC  # noqa: F401

        _OPENCC_AVAILABLE = True
        return True
    except Exception:  # noqa: BLE001
        _OPENCC_AVAILABLE = False
        return False


@functools.lru_cache(maxsize=4)
def _opencc_converter(config: str):
    # Lazy import to keep unit tests / offline environments safe.
    from opencc import OpenCC

    return OpenCC(config)


def _opencc_convert(text: str, *, config: str) -> str | None:
    """
    Convert text via OpenCC when available.

    Returns:
        Converted text if OpenCC is available and conversion changes the string,
        otherwise None.
    """
    if not text.strip():
        return None
    if not _detect_opencc_available():
        return None
    try:
        converted = str(_opencc_converter(config).convert(text))
    except Exception:  # noqa: BLE001
        return None
    converted = converted.strip()
    if not converted or converted == text:
        return None
    return converted


def _extract_ascii_keyword_query(text: str) -> str | None:
    """
    Extract lightweight ASCII tokens to help BM25 match mixed-language corpora.

    Avoid regex to keep this robust and easy to reason about.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    tokens: list[str] = []
    buf: list[str] = []

    def _flush():
        if not buf:
            return
        token = "".join(buf).strip()
        buf.clear()
        if token:
            tokens.append(token)

    allowed_punct = {"-", "_", "/", "."}
    for ch in raw:
        if ch.isascii() and (ch.isalnum() or ch in allowed_punct):
            buf.append(ch)
        else:
            _flush()
    _flush()

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    query = " ".join(out).strip()
    return query or None


def generate_query_variants(query: str, *, llm_connector=None, cache_scope: str | None = None) -> list[str]:
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

    # Deterministic variants (no LLM required).
    for lang in langs:
        key = str(lang).strip().lower()
        if key == "zh-hans":
            candidate = _opencc_convert(base, config="t2s")
            if candidate:
                variants.append(candidate)
        elif key == "zh-hant":
            candidate = _opencc_convert(base, config="s2t")
            if candidate:
                variants.append(candidate)
        elif key == "en":
            candidate = _extract_ascii_keyword_query(base)
            if candidate:
                variants.append(candidate)

    # Optional LLM enrichment (if provided).
    if langs and llm_connector is not None:
        rewritten = _llm_rewrite_variants(llm_connector, base, langs, cache_scope=cache_scope)
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
