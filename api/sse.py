"""SSE helpers aligned with Qwen(OpenAI-compatible) streaming format."""
import json
import time
import uuid
from typing import Any, Dict, Iterator, Optional


def sse_json(payload: Any) -> str:
    """Return a single SSE `data:` event containing compact JSON."""

    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def sse_json_wrapped(payload: Any, request_id: str | None = None, code: int = 200, message: str = "success") -> str:
    """Return a single SSE `data:` event wrapped in standard response format."""
    if request_id is None:
        request_id = uuid.uuid4().hex
    wrapped = {
        "code": code,
        "message": message,
        "data": payload,
        "request_id": request_id
    }
    return f"data: {json.dumps(wrapped, ensure_ascii=False, separators=(',', ':'))}\n\n"


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
        chunk_chars = 1 if _has_cjk(text) else 6
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
    """Build an OpenAI/Qwen-compatible delta object.

    Mirrors common OpenAI-compatible streaming behavior:
    - first chunk: {"role":"assistant","content":"","refusal":null}
    - content chunks: {"content":"..."}
    - final chunk: {}
    """

    delta: Dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
        delta["refusal"] = None
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return delta
