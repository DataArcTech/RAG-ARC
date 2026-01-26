"""事件生成器初始化函数"""
import asyncio
from typing import Any, AsyncGenerator
from api.sse import (
    delta_envelope,
    openai_chat_completion_chunk,
    sse_json_wrapped,
)
from api.routers.rag_inference_handlers import get_rag_inference_handler
from ..utils.history_manager import create_user_message, load_and_process_history
from ..task.task_helpers import create_and_register_task
from .event_generator import _create_progress_emitter


async def yield_initial_assistant_event(
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str
) -> AsyncGenerator[str, None]:
    """生成初始 assistant role 事件"""
    yield sse_json_wrapped(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(role="assistant", content=""),
        ),
        request_id=request_id
    )


async def _initialize_chat_context(
    session_id: Any,
    query: str,
    request_id: str
) -> tuple[Any, Any, Any, bool, Any]:
    """
    初始化聊天上下文
    
    返回: (user_message, history_messages, history_text, first_turn, task_info)
    """
    # 创建用户消息和加载历史
    user_message = await create_user_message(session_id, query)
    history_messages, history_text, first_turn = await load_and_process_history(
        session_id,
        user_message.id
    )
    
    # 创建任务并更新状态
    task_info = await create_and_register_task(
        session_id=session_id,
        user_message_id=user_message.id,
        query=query,
        request_id=request_id
    )
    
    return user_message, history_messages, history_text, first_turn, task_info


def initialize_streaming_state(
    request_id: str,
    loop: Any
) -> tuple[list[str], Any, list, dict, Any, Any]:
    """
    初始化流式处理状态
    
    返回: (response_parts, queue, stream_error, prepared, rag_inference_handler, emit_progress)
    """
    response_parts: list[str] = []
    queue: asyncio.Queue[object | None] = asyncio.Queue()
    stream_error: list[Exception | None] = [None]
    prepared: dict[str, Any] = {}
    rag_inference_handler = get_rag_inference_handler()
    emit_progress = _create_progress_emitter(request_id, queue, loop)
    
    return response_parts, queue, stream_error, prepared, rag_inference_handler, emit_progress
