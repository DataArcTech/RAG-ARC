"""History message management for stream chat."""
import os
import logging
import uuid
from datetime import datetime
from typing import Optional
from encapsulation.data_model.orm_models import ChatMessage
from api.routers.rag_inference_handlers import get_message_handler
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)


async def create_user_message(
    session_id: uuid.UUID,
    query: str
) -> ChatMessage:
    """Create and save user message."""
    message_handler = get_message_handler()
    user_message = ChatMessage(
        session_id=session_id,
        content={"role": "user", "content": query},
        created_at=datetime.now(),
    )
    return await get_thread_pool().run_blocking(
        message_handler.create_message,
        user_message
    )


async def load_and_process_history(
    session_id: uuid.UUID,
    exclude_message_id: uuid.UUID
) -> tuple[list[ChatMessage], Optional[str], bool]:
    """Load history messages and process them.
    
    Returns:
        Tuple of (history_messages, history_text, first_turn)
    """
    message_handler = get_message_handler()
    history_messages = await get_thread_pool().run_blocking(
        message_handler.list_messages_by_session,
        session_id,
    )
    
    # Exclude the message we just created
    history_messages = [
        msg for msg in history_messages
        if msg.id != exclude_message_id
    ]
    
    # Check if this is the first turn
    first_turn = not any(
        msg.content.get("role") == "assistant"
        for msg in history_messages
        if isinstance(msg.content, dict)
    )
    
    # Limit history length
    context_turns = int(os.getenv("SSE_CONTEXT_TURNS", "5"))
    max_history_messages = context_turns * 2
    if len(history_messages) > max_history_messages:
        history_messages = history_messages[-max_history_messages:]
    
    # Build history text
    history_text = "\n".join(
        f"{msg.content['role']}: {msg.content['content']}"
        for msg in history_messages
    ) if history_messages else None
    
    # Enforce context budget
    if history_text:
        history_tokens = (
            len(history_text)
            if any(ord(ch) >= 128 for ch in history_text)
            else len(history_text) // 4
        )
        max_context_tokens = int(os.getenv("SSE_MAX_CONTEXT_TOKENS", "8192"))
        threshold_fraction = float(os.getenv("SSE_MAX_CONTEXT_FRACTION", "0.9"))
        allowed_tokens = int(max_context_tokens * threshold_fraction)
        
        if history_tokens > allowed_tokens:
            logger.warning(
                "History too long: estimated_tokens=%d, allowed=%d, truncating history",
                history_tokens,
                allowed_tokens,
            )
            reduced_turns = max(1, context_turns // 2)
            max_history_messages = reduced_turns * 2
            if len(history_messages) > max_history_messages:
                history_messages = history_messages[-max_history_messages:]
                history_text = "\n".join(
                    f"{msg.content['role']}: {msg.content['content']}"
                    for msg in history_messages
                )
    
    return history_messages, history_text, first_turn
