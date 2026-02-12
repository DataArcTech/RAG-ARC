"""Global helpers for strict JSON LLM outputs with retries."""
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from config.core import llm_json_retry_defaults as json_retry_defaults
from core.prompts.llm_json import JSON_RETRY_INSTRUCTION_EN
from core.prompts.runtime_context import prepend_today_line
from core.utils.json_extract import extract_json_from_text, safe_json_loads


def _normalize_expected(expected: str | None) -> str | None:
    if expected is None:
        return None
    token = str(expected or "").strip().lower() or "dict"
    if token in {"any", "json"}:
        return None
    if token == "object":
        return "dict"
    return token


def _truncate_raw(raw: str, *, limit: int | None = None) -> str:
    if limit is None:
        limit = int(getattr(json_retry_defaults, "LLM_JSON_RETRY_DEFAULT_MAX_RAW_CHARS", 2000))
    if limit <= 0:
        return ""
    if len(raw) <= limit:
        return raw
    return raw[:limit]


def _normalize_messages(messages: List[Dict[str, Any]], *, include_today_line: bool) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in list(messages or []):
        row = dict(item or {})
        if include_today_line and str(row.get("role") or "").strip() == "system":
            row["content"] = prepend_today_line(str(row.get("content") or ""))
        normalized.append(row)
    return normalized


async def _call_llm_async(llm: Any, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    if llm is None:
        raise RuntimeError("LLM connector is required for this tool")

    async_chat = getattr(llm, "achat", None)
    if callable(async_chat):
        return await async_chat(messages, **kwargs)

    chat = getattr(llm, "chat", None)
    if not callable(chat):
        raise RuntimeError("LLM connector does not expose chat/achat methods")
    return chat(messages, **kwargs)


def _call_llm_sync(llm: Any, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    if llm is None:
        raise RuntimeError("LLM connector is required for this tool")
    chat = getattr(llm, "chat", None)
    if not callable(chat):
        raise RuntimeError("LLM connector does not expose chat() method")
    return chat(messages, **kwargs)


def _coerce_attempts(attempts: int | None) -> int:
    if attempts is None:
        attempts = int(json_retry_defaults.LLM_JSON_RETRY_DEFAULT_ATTEMPTS)
    max_attempts = int(json_retry_defaults.LLM_JSON_RETRY_MAX_ATTEMPTS)
    attempts = max(1, int(attempts))
    if max_attempts > 0:
        attempts = min(attempts, max_attempts)
    return attempts


def _merge_kwargs(
    *,
    temperature: float | None,
    max_tokens: int | None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    apply_default_temperature: bool = False,
) -> Dict[str, Any]:
    extra_kwargs: Dict[str, Any] = dict(llm_kwargs or {})
    if temperature is not None:
        extra_kwargs["temperature"] = float(temperature)
    elif apply_default_temperature and "temperature" not in extra_kwargs:
        extra_kwargs["temperature"] = float(json_retry_defaults.LLM_JSON_RETRY_DEFAULT_TEMPERATURE)
    if max_tokens is not None:
        extra_kwargs["max_tokens"] = int(max_tokens)
    return extra_kwargs


def _extract_payload(raw: str, expected: str | None) -> Any | None:
    extracted = extract_json_from_text(raw)
    return safe_json_loads(extracted or raw, expected=expected)


def _resolve_retry_instruction(retry_instruction: str | None) -> str:
    text = str(retry_instruction or "").strip()
    return text or JSON_RETRY_INSTRUCTION_EN


async def call_llm_json_with_retry(
    *,
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    expected: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    include_today_line: bool = False,
    max_raw_chars: int | None = None,
    retry_instruction: str | None = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str]]:
    """Call an LLM and parse JSON with centralized retries (async)."""

    tries = _coerce_attempts(attempts)
    expected_norm = _normalize_expected(expected)
    base_messages = _normalize_messages(messages or [], include_today_line=include_today_line)
    retry_message = _resolve_retry_instruction(retry_instruction)

    last_raw: str = ""
    for attempt in range(tries):
        call_messages = list(base_messages)
        if attempt > 0:
            if last_raw:
                call_messages.append({"role": "assistant", "content": _truncate_raw(last_raw, limit=max_raw_chars)})
            call_messages.append({"role": "user", "content": retry_message})
        extra_kwargs = _merge_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
            apply_default_temperature=attempt > 0,
        )
        raw = await _call_llm_async(llm_connector, call_messages, **extra_kwargs)
        last_raw = str(raw or "")
        payload = _extract_payload(last_raw, expected_norm)
        if payload is not None:
            return (payload, last_raw) if return_raw else payload
    return (None, last_raw) if return_raw else None


async def repair_json_from_raw_with_retry(
    *,
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    raw: str,
    expected: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    include_today_line: bool = False,
    max_raw_chars: int | None = None,
    retry_instruction: str | None = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str]]:
    """Given a failed raw response, ask the model to repair and return valid JSON."""

    tries = _coerce_attempts(attempts)
    expected_norm = _normalize_expected(expected)
    base_messages = _normalize_messages(messages or [], include_today_line=include_today_line)
    retry_message = _resolve_retry_instruction(retry_instruction)
    last_raw = str(raw or "")

    for _ in range(tries):
        call_messages = list(base_messages)
        if last_raw:
            call_messages.append({"role": "assistant", "content": _truncate_raw(last_raw, limit=max_raw_chars)})
        call_messages.append({"role": "user", "content": retry_message})
        extra_kwargs = _merge_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
            apply_default_temperature=True,
        )
        next_raw = await _call_llm_async(llm_connector, call_messages, **extra_kwargs)
        last_raw = str(next_raw or "")
        payload = _extract_payload(last_raw, expected_norm)
        if payload is not None:
            return (payload, last_raw) if return_raw else payload
    return (None, last_raw) if return_raw else None


def call_llm_json_with_retry_sync(
    *,
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    expected: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    include_today_line: bool = False,
    max_raw_chars: int | None = None,
    retry_instruction: str | None = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str]]:
    """Call an LLM and parse JSON with centralized retries (sync)."""

    tries = _coerce_attempts(attempts)
    expected_norm = _normalize_expected(expected)
    base_messages = _normalize_messages(messages or [], include_today_line=include_today_line)
    retry_message = _resolve_retry_instruction(retry_instruction)

    last_raw: str = ""
    for attempt in range(tries):
        call_messages = list(base_messages)
        if attempt > 0:
            if last_raw:
                call_messages.append({"role": "assistant", "content": _truncate_raw(last_raw, limit=max_raw_chars)})
            call_messages.append({"role": "user", "content": retry_message})
        extra_kwargs = _merge_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
            apply_default_temperature=attempt > 0,
        )
        raw = _call_llm_sync(llm_connector, call_messages, **extra_kwargs)
        last_raw = str(raw or "")
        payload = _extract_payload(last_raw, expected_norm)
        if payload is not None:
            return (payload, last_raw) if return_raw else payload
    return (None, last_raw) if return_raw else None


def call_prompt_json_with_retry_sync(
    *,
    infer_once: Callable[[str], str],
    prompt: str,
    expected: str | None,
    attempts: int | None = None,
    max_raw_chars: int | None = None,
    retry_instruction: str | None = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str]]:
    """Run prompt-based generation and parse JSON with centralized retries (sync)."""

    tries = _coerce_attempts(attempts)
    expected_norm = _normalize_expected(expected)
    base_prompt = str(prompt or "").strip()
    retry_message = _resolve_retry_instruction(retry_instruction)

    last_raw = ""
    for attempt in range(tries):
        if attempt <= 0:
            current_prompt = base_prompt
        else:
            current_prompt = (
                f"{base_prompt}\n\n{retry_message}\n"
                f"Previous output:\n{_truncate_raw(last_raw, limit=max_raw_chars)}"
            ).strip()
        raw = infer_once(current_prompt)
        last_raw = str(raw or "")
        payload = _extract_payload(last_raw, expected_norm)
        if payload is not None:
            return (payload, last_raw) if return_raw else payload
    return (None, last_raw) if return_raw else None


def repair_json_from_raw_with_retry_sync(
    *,
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    raw: str,
    expected: str | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    include_today_line: bool = False,
    max_raw_chars: int | None = None,
    retry_instruction: str | None = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str]]:
    """Sync variant of JSON repair retries using a failed raw response."""

    tries = _coerce_attempts(attempts)
    expected_norm = _normalize_expected(expected)
    base_messages = _normalize_messages(messages or [], include_today_line=include_today_line)
    retry_message = _resolve_retry_instruction(retry_instruction)
    last_raw = str(raw or "")

    for _ in range(tries):
        call_messages = list(base_messages)
        if last_raw:
            call_messages.append({"role": "assistant", "content": _truncate_raw(last_raw, limit=max_raw_chars)})
        call_messages.append({"role": "user", "content": retry_message})
        extra_kwargs = _merge_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
            apply_default_temperature=True,
        )
        next_raw = _call_llm_sync(llm_connector, call_messages, **extra_kwargs)
        last_raw = str(next_raw or "")
        payload = _extract_payload(last_raw, expected_norm)
        if payload is not None:
            return (payload, last_raw) if return_raw else payload
    return (None, last_raw) if return_raw else None


__all__ = [
    "call_llm_json_with_retry",
    "call_llm_json_with_retry_sync",
    "repair_json_from_raw_with_retry",
    "repair_json_from_raw_with_retry_sync",
    "call_prompt_json_with_retry_sync",
]
