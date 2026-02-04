import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from encapsulation.llm.utils.image_data_url import file_to_data_url

logger = logging.getLogger(__name__)


def build_multimodal_user_message(*, text: str, image_paths: List[Path]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": str(text or "")}]
    for path in image_paths:
        parts.append({"type": "image_url", "image_url": {"url": file_to_data_url(path)}})
    return {"role": "user", "content": parts}


async def call_llm_with_optional_images_async(
    llm: Any,
    *,
    messages: List[Dict[str, Any]],
    user_message_index: int,
    image_paths: List[Path],
    warn_context: str,
    **kwargs: Any,
) -> str:
    """
    Best-effort multimodal call:
    - If image_paths empty: normal call.
    - Else: try vision message (OpenAI-style parts). On failure: warn and retry as text-only.
    """
    from core.deepsearch.tools.base import call_llm_async

    if not image_paths:
        return await call_llm_async(llm, messages, warn_context=warn_context, **kwargs)

    if user_message_index < 0 or user_message_index >= len(messages):
        return await call_llm_async(llm, messages, warn_context=warn_context, **kwargs)

    original = messages[user_message_index]
    original_content = original.get("content")
    if not isinstance(original_content, str):
        return await call_llm_async(llm, messages, warn_context=warn_context, **kwargs)

    multimodal_messages = [dict(m) for m in messages]
    multimodal_messages[user_message_index] = build_multimodal_user_message(
        text=original_content,
        image_paths=image_paths,
    )

    try:
        return await call_llm_async(llm, multimodal_messages, warn_context=f"{warn_context}.multimodal", **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Multimodal input unsupported or failed (%s): %s; falling back to text-only using captions/alt text.",
            warn_context,
            exc,
        )
        return await call_llm_async(llm, messages, warn_context=f"{warn_context}.text_fallback", **kwargs)
