"""SSE helpers aligned with Qwen(OpenAI-compatible) streaming format."""
import json
import time
import uuid
from typing import Any, Dict, Iterator, Optional


def sse_json(payload: Any) -> str:
    """Return a single SSE `data:` event containing compact JSON."""

    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_done() -> str:
    """Return the final SSE event used by Qwen/OpenAI-compatible streams."""

    return "data: [DONE]\n\n"


def new_chatcmpl_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def now_epoch_seconds() -> int:
    return int(time.time())


def openai_chat_completion_chunk(
    *,
    chunk_id: str,
    model: str,
    created: int,
    delta: Dict[str, Any],
    index: int = 0,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Qwen-like OpenAI stream chunk object.

    DashScope/Qwen examples include several nullable fields. We keep them to be
    maximally compatible with clients expecting that shape.
    """

    return {
        "id": chunk_id,
        "choices": [
            {
                "delta": delta,
                "finish_reason": finish_reason,
                "index": index,
                "logprobs": None,
            }
        ],
        "created": created,
        "model": model,
        "object": "chat.completion.chunk",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": None,
    }


def _has_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
            or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
        ):
            return True
    return False


def iter_text_deltas(text: str, *, chunk_chars: int | None = None) -> Iterator[str]:
    """Split text into small incremental deltas similar to Qwen token streaming."""

    if not text:
        return

    if chunk_chars is None:
        chunk_chars = 3 if _has_cjk(text) else 12
    chunk_chars = max(1, int(chunk_chars))

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        yield text[start:end]
        start = end


def delta_envelope(
    *,
    content: str | None = None,
    role: str | None = None,
    tool_calls: Any | None = None,
) -> Dict[str, Any]:
    """Build a delta object aligned to DashScope/Qwen examples."""

    return {
        "content": content if content is not None else "",
        "function_call": None,
        "refusal": None,
        "role": role,
        "tool_calls": tool_calls,
    }
