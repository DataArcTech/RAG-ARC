"""DeepSearch wrapper for global JSON retry helpers."""
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from config.core.deepsearch import tool_defaults
from core.utils.llm_json import call_llm_json_with_retry as _call_llm_json_with_retry


async def call_llm_json_with_retry(
    llm_connector: Any,
    messages: List[Dict[str, Any]],
    expected: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    attempts: int | None = None,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    return_raw: bool = False,
) -> Union[Any, Tuple[Any, str], str]:
    """DeepSearch compatibility shim that preserves prior defaults.

    Backward compatibility:
    - positional invocation is supported;
    - when ``expected`` is omitted and ``return_raw=False``, we return raw text.
    """

    if attempts is None:
        attempts = int(getattr(tool_defaults, "SEARCH_ENTITY_EXTRACT_JSON_REPAIR_ATTEMPTS", 2))
    legacy_raw_mode = expected is None and not bool(return_raw)
    payload = await _call_llm_json_with_retry(
        llm_connector=llm_connector,
        messages=messages,
        expected=expected,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        attempts=attempts,
        llm_kwargs=llm_kwargs,
        include_today_line=True,
        return_raw=return_raw or legacy_raw_mode,
    )
    if legacy_raw_mode and isinstance(payload, tuple):
        _parsed, raw = payload
        return raw
    return payload


__all__ = ["call_llm_json_with_retry"]
