"""任务恢复相关函数"""
import asyncio
import logging
from typing import Any, AsyncGenerator
from fastapi.responses import StreamingResponse
from .task_registry import ChatTaskInfo, get_chat_task_registry
from api.sse import sse_json_wrapped, sse_done
from api.routers.rag_inference_handlers import get_message_handler
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)


async def _yield_task_resumed_event(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """发送任务恢复通知事件"""
    yield sse_json_wrapped({
        "type": "task_resumed",
        "task_id": task_info.task_id,
        "response_length": task_info.response_length,
        "events_count": task_info.events_count
    }, request_id)


async def _yield_completed_task_response(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """发送已完成任务的完整响应"""
    if task_info.error:
        yield sse_json_wrapped(
            {"error": {"message": task_info.error}},
            request_id=request_id,
            code=500,
            message="error"
        )
    else:
        if task_info.assistant_message_id:
            assistant_msg = await get_thread_pool().run_blocking(
                get_message_handler().get_message,
                task_info.assistant_message_id
            )
            if assistant_msg:
                content = assistant_msg.content.get("content", "") if isinstance(assistant_msg.content, dict) else ""
                yield sse_json_wrapped({
                    "type": "complete_response",
                    "content": content
                }, request_id)
    yield sse_done()


async def _replay_cached_events(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """回放已缓存的事件"""
    registry = get_chat_task_registry()
    
    # 优先从 Redis 回放
    redis_events = await registry.get_redis_events(task_info.task_id)
    if redis_events:
        logger.info("Replaying %d events from Redis for task %s", len(redis_events), task_info.task_id)
        for event in redis_events:
            yield sse_json_wrapped(event["data"], request_id)
    elif task_info.response_text:
        # 发送响应摘要
        yield sse_json_wrapped({
            "type": "response_summary",
            "text": task_info.response_text,
            "length": task_info.response_length
        }, request_id)


async def _poll_task_updates(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """轮询任务更新并发送新事件"""
    registry = get_chat_task_registry()
    last_length = task_info.response_length
    
    while not task_info.done and not task_info.cancelled:
        await asyncio.sleep(0.5)
        
        updated_info = await registry.get(task_info.task_id)
        if not updated_info:
            break
        
        # 发送新的文本增量
        if updated_info.response_length > last_length:
            new_text = updated_info.response_text[last_length:updated_info.response_length]
            yield sse_json_wrapped({
                "type": "text_delta",
                "content": new_text
            }, request_id)
            last_length = updated_info.response_length
        
        # 发送新的 Redis 事件
        if registry._use_redis_events:
            new_events = await registry.get_redis_events(task_info.task_id)
            if len(new_events) > task_info.events_count:
                for event in new_events[task_info.events_count:]:
                    yield sse_json_wrapped(event["data"], request_id)
                task_info.events_count = len(new_events)


async def _yield_task_completion_event(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """发送任务完成事件"""
    if task_info.error:
        yield sse_json_wrapped(
            {"error": {"message": task_info.error}},
            request_id=request_id,
            code=500,
            message="error"
        )
    else:
        yield sse_done()
