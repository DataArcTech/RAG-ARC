"""Helpers for strict JSON tool outputs.

Many DeepSearch tools require model outputs to be valid JSON. Even good models sometimes emit
markdown fences or get truncated. This module provides a centralized, configurable retry loop.

Note: This lives under `core/deepsearch/utils/` so it can be reused by tools and other
DeepSearch components without creating circular imports.
"""
from typing import Any, Dict, List, Mapping, Optional

from config.core.deepsearch import tool_defaults
from core.prompts.deepsearch import JSON_RETRY_INSTRUCTION_EN

from core.utils.json_extract import extract_json_from_text, safe_json_loads


async def _call_llm_async(llm: Any, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    """Invoke sync/async LLM connectors transparently.

    NOTE: This is duplicated from `core.deepsearch.tools.base` on purpose to avoid importing
    the `core.deepsearch.tools` package (which has import-time registrations and can create
    circular imports when called from utilities).
    """

    if llm is None:
        raise RuntimeError("LLM connector is required for this tool")

    async_chat = getattr(llm, "achat", None)
    if callable(async_chat):
        return await async_chat(messages, **kwargs)

    chat = getattr(llm, "chat", None)
    if not callable(chat):
        raise RuntimeError("LLM connector does not expose chat/achat methods")
    return chat(messages, **kwargs)


async def call_llm_json_with_retry(
    *,
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    expected: str,
    temperature: float,
    max_tokens: int,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any | None:
    """Call an LLM and parse JSON with centralized retries.

    - Uses the shared JSON extractor/parsers from core.utils.json_extract.
    - Retries are done by asking the model to re-emit strict JSON (no heuristic brace-closing).
    """

    tries = int(attempts) if attempts is not None else int(
        getattr(tool_defaults, "SEARCH_ENTITY_EXTRACT_JSON_REPAIR_ATTEMPTS", 2)
    )
    tries = max(1, min(tries, 8))

    expected_norm = expected
    if expected_norm == "object":
        expected_norm = "dict"

    # Keep the original message list stable; append retry instruction on later attempts.
    base_messages = list(messages or [])
    last_raw: str | None = None
    extra_kwargs: Dict[str, Any] = dict(llm_kwargs or {})
    for attempt in range(tries):
        call_messages: List[Dict[str, Any]] = list(base_messages)
        if attempt > 0:
            # Feed the prior output back as an assistant message so the model can repair it, then
            # request strict JSON only. Keep excerpts short to avoid ballooning context.
            if last_raw:
                call_messages.append({"role": "assistant", "content": last_raw[:2000]})
            call_messages.append({"role": "user", "content": JSON_RETRY_INSTRUCTION_EN})
        raw = await _call_llm_async(
            llm_connector,
            call_messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            **extra_kwargs,
        )
        last_raw = str(raw or "")
        extracted = extract_json_from_text(last_raw)
        payload = safe_json_loads(extracted or last_raw, expected=expected_norm)
        if payload is not None:
            return payload
    return None


__all__ = ["call_llm_json_with_retry"]
