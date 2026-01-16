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
    logger.info("Yielding task_resumed event for task %s (done=%s, cancelled=%s, response_length=%d)", 
               task_info.task_id, task_info.done, task_info.cancelled, task_info.response_length)
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
    logger.info("Task %s is done, sending completed response (assistant_message_id=%s, error=%s)", 
                task_info.task_id, task_info.assistant_message_id, task_info.error)
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
                logger.info("Sending complete_response for task %s, content_length=%d", 
                            task_info.task_id, len(content))
                yield sse_json_wrapped({
                    "type": "complete_response",
                    "content": content
                }, request_id)
            else:
                logger.warning("Assistant message %s not found for completed task %s", 
                              task_info.assistant_message_id, task_info.task_id)
                # 如果没有找到消息，发送响应摘要
                if task_info.response_text:
                    yield sse_json_wrapped({
                        "type": "response_summary",
                        "text": task_info.response_text,
                        "length": task_info.response_length
                    }, request_id)
        else:
            logger.warning("Task %s completed but no assistant_message_id, sending response_summary", 
                          task_info.task_id)
            # 如果没有 assistant_message_id，发送响应摘要
            if task_info.response_text:
                yield sse_json_wrapped({
                    "type": "response_summary",
                    "text": task_info.response_text,
                    "length": task_info.response_length
                }, request_id)
    yield sse_done()


async def _replay_cached_events(
    task_info: ChatTaskInfo,
    request_id: str
) -> Any:
    """回放已缓存的事件"""
    import time
    registry = get_chat_task_registry()
    
    # 优先从 Redis 回放
    redis_events = await registry.get_redis_events(task_info.task_id)
    if redis_events:
        logger.info("Replaying %d events from Redis for task %s", len(redis_events), task_info.task_id)
        for event in redis_events:
            yield sse_json_wrapped(event["data"], request_id)
    elif task_info.response_text:
        # 发送响应摘要
        # 如果任务已完成，尝试从数据库获取完整的响应文本
        response_text = task_info.response_text
        response_length = task_info.response_length
        
        # 如果任务已完成但没有assistant_message_id，尝试从数据库查找
        if (task_info.done or task_info.assistant_message_id) and task_info.user_message_id:
            try:
                from api.routers.rag_inference_handlers import get_message_handler
                from framework.thread_pool import get_thread_pool

                message_handler = get_message_handler()
                messages = await get_thread_pool().run_blocking(
                    message_handler.list_messages_by_session,
                    task_info.session_id,
                    limit=50
                )
                
                # 查找user_message之后的assistant message
                # 消息按 created_at 升序排列（oldest first），所以 assistant message 在 user_message 之后
                user_message_index = None
                for i, msg in enumerate(messages):
                    if str(msg.id) == str(task_info.user_message_id):
                        user_message_index = i
                        break
                
                if user_message_index is not None:
                    # 在user_message之后查找assistant message（索引更大）
                    for i in range(user_message_index + 1, len(messages)):
                        msg = messages[i]
                        if msg.content.get("role") == "assistant":
                            # 使用数据库中的完整响应文本
                            db_response_text = msg.content.get("content", "")
                            if db_response_text and len(db_response_text) > len(response_text):
                                logger.info("Task %s: Using complete response from database (length=%d) instead of partial response_text (length=%d)", 
                                           task_info.task_id, len(db_response_text), len(response_text))
                                response_text = db_response_text
                                response_length = len(db_response_text)
                            break
            except Exception as e:
                logger.warning("Failed to get complete response from database for task %s: %s", 
                             task_info.task_id, e)
        
        # 检查任务是否已完成
        is_task_completed = task_info.done or task_info.assistant_message_id is not None
        
        # 如果任务有响应文本但没有assistant_message_id，且任务标记为done，说明任务已完成但未保存assistant message
        # 主动完成最终化：创建assistant message并标记任务完成
        # 注意：只有在任务明确标记为done时才完成最终化，如果任务还在processing，不要完成最终化
        if response_text and not task_info.assistant_message_id and task_info.user_message_id and task_info.done:
            try:
                logger.info("Task %s is done but has no assistant_message_id, completing finalization...", 
                           task_info.task_id)
                from api.routers.rag_inference_modules.stream_chat.response.response_builder import create_assistant_message
                from api.routers.rag_inference_modules.stream_chat.task.task_helpers import mark_task_completed
                
                # 创建基本的assistant message（只有content，没有sources等）
                assistant_message = await create_assistant_message(
                    session_id=task_info.session_id,
                    assistant_response=response_text,
                    sources_for_frontend=[],  # 空列表，因为sources信息可能已丢失
                    subgraph_data=None,
                    return_subgraph=False,
                    raw_llm_response=None,
                    raw_mindmap_response=None,
                    deepsearch_trace_file_path=None
                )
                
                # 标记任务完成
                await mark_task_completed(
                    task_info,
                    task_info.session_id,
                    task_info.user_message_id,
                    assistant_message_id=assistant_message.id
                )
                
                logger.info("Task %s finalization completed: assistant_message_id=%s", 
                           task_info.task_id, assistant_message.id)
                
                # 更新task_info
                registry = get_chat_task_registry()
                task_info = await registry.get(task_info.task_id) or task_info
                is_task_completed = True  # 更新完成状态
            except Exception as e:
                logger.warning("Failed to complete finalization for task %s: %s", 
                             task_info.task_id, e, exc_info=True)
        
        if is_task_completed:
            # 任务已完成，发送response_summary
            logger.info("Task %s is completed, sending response_summary (length=%d)", 
                       task_info.task_id, response_length)
            yield sse_json_wrapped({
                "type": "response_summary",
                "text": response_text,
                "length": response_length
            }, request_id)
        else:
            # 任务还在processing，发送已有的response_text作为text_delta，然后继续轮询
            logger.info("Task %s still processing, sending existing response_text as text_delta (length=%d), will continue polling", 
                       task_info.task_id, response_length)
            if response_text:
                # 发送已有的响应文本作为text_delta事件
                logger.info("Task %s: Yielding text_delta event (length=%d)", task_info.task_id, len(response_text))
                yield sse_json_wrapped({
                    "type": "text_delta",
                    "content": response_text
                }, request_id)
                logger.info("Task %s: text_delta event yielded, _replay_cached_events will return", task_info.task_id)
            # 不发送response_summary，让函数正常返回，继续执行轮询
            logger.info("Task %s: _replay_cached_events returning, will continue to polling in _resume_task_sse", task_info.task_id)
    else:
        # 如果没有缓存事件且没有响应文本，任务可能还在检索阶段
        # 等待一段时间，让任务有机会生成内容
        logger.info("No cached events and no response_text for task %s, waiting for task to generate content...", 
                   task_info.task_id)
        wait_start = time.time()
        max_wait_time = 30.0  # 最多等待30秒
        
        while time.time() - wait_start < max_wait_time:
            await asyncio.sleep(1.0)
            # 检查任务状态
            updated_info = await registry.get(task_info.task_id)
            if not updated_info:
                logger.warning("Task %s not found during wait", task_info.task_id)
                break
            
            # 如果任务已完成，退出等待（会在后续流程中处理完成事件）
            if updated_info.done or updated_info.cancelled or updated_info.assistant_message_id:
                logger.info("Task %s completed during wait (done=%s, cancelled=%s, assistant_message_id=%s)", 
                          task_info.task_id, updated_info.done, updated_info.cancelled, updated_info.assistant_message_id)
                # 更新 task_info 以便后续处理
                task_info = updated_info
                break
            
            # 如果有新的事件或响应文本，退出等待
            new_events = await registry.get_redis_events(task_info.task_id)
            if new_events or updated_info.response_text:
                logger.info("Task %s generated content during wait (events=%d, response_length=%d)", 
                          task_info.task_id, len(new_events) if new_events else 0, updated_info.response_length)
                # 回放新获取的事件
                if new_events:
                    for event in new_events:
                        yield sse_json_wrapped(event["data"], request_id)
                break
            
            # 更新 task_info
            task_info = updated_info
        
        if not task_info.response_text and not await registry.get_redis_events(task_info.task_id):
            logger.warning("No cached events and no response_text for task %s after waiting", task_info.task_id)


async def _poll_task_updates(
    task_info: ChatTaskInfo,
    request_id: str,
    max_poll_time: float = 300.0  # 最大轮询时间 5 分钟
) -> Any:
    """轮询任务更新并发送新事件"""
    import time
    registry = get_chat_task_registry()
    last_length = task_info.response_length
    start_time = time.time()
    no_update_count = 0  # 连续无更新次数
    max_no_update = 20  # 连续 10 秒无更新则认为任务可能已完成
    
    while not task_info.done and not task_info.cancelled:
        # 超时检查
        if time.time() - start_time > max_poll_time:
            logger.warning("Task %s polling timeout after %d seconds", task_info.task_id, max_poll_time)
            break
        
        await asyncio.sleep(0.5)
        
        updated_info = await registry.get(task_info.task_id)
        if not updated_info:
            logger.warning("Task %s not found during polling", task_info.task_id)
            break
        
        # 更新 task_info 引用
        task_info = updated_info
        
        # 检查任务是否已完成：done 标志、cancelled 标志或 assistant_message_id 存在
        is_completed = updated_info.done or updated_info.cancelled or updated_info.assistant_message_id is not None
        if is_completed:
            if updated_info.assistant_message_id and not updated_info.done:
                logger.info("Task %s appears completed (has assistant_message_id) but done flag not set, marking as done", 
                           updated_info.task_id)
                await registry.mark_done(updated_info.task_id, assistant_message_id=updated_info.assistant_message_id)
                updated_info = await registry.get(task_info.task_id) or updated_info
            
            logger.info("Task %s completed during polling (done=%s, cancelled=%s, assistant_message_id=%s)", 
                       task_info.task_id, updated_info.done, updated_info.cancelled, updated_info.assistant_message_id)
            break
        
        has_update = False
        
        # 发送新的文本增量
        if updated_info.response_length > last_length:
            new_text = updated_info.response_text[last_length:updated_info.response_length]
            yield sse_json_wrapped({
                "type": "text_delta",
                "content": new_text
            }, request_id)
            last_length = updated_info.response_length
            has_update = True
            no_update_count = 0
        
        # 发送新的 Redis 事件
        if registry._use_redis_events:
            new_events = await registry.get_redis_events(task_info.task_id)
            if len(new_events) > task_info.events_count:
                for event in new_events[task_info.events_count:]:
                    yield sse_json_wrapped(event["data"], request_id)
                task_info.events_count = len(new_events)
                has_update = True
                no_update_count = 0
        
        # 如果长时间无更新，检查任务是否实际已完成
        if not has_update:
            no_update_count += 1
            if no_update_count >= max_no_update:
                logger.info("Task %s has no updates for %d seconds, checking if completed", 
                           task_info.task_id, no_update_count * 0.5)
                # 再次检查任务状态（包括检查 assistant_message_id）
                final_check = await registry.get(task_info.task_id)
                if final_check:
                    is_completed = final_check.done or final_check.cancelled or final_check.assistant_message_id is not None
                    
                    # 如果任务有响应内容但没有assistant_message_id，检查数据库中是否有对应的assistant message
                    if not is_completed and final_check.response_length > 0 and final_check.user_message_id:
                        try:
                            from api.routers.rag_inference_handlers import get_message_handler
                            from framework.thread_pool import get_thread_pool
                            
                            # 检查数据库中是否有对应的assistant message（在user_message之后创建的）
                            message_handler = get_message_handler()
                            messages = await get_thread_pool().run_blocking(
                                message_handler.list_messages_by_session,
                                final_check.session_id,
                                limit=50  # 检查最近50条消息
                            )
                            
                            # 查找user_message之后的assistant message
                            # 消息按 created_at 升序排列（oldest first），所以 assistant message 在 user_message 之后
                            user_message_index = None
                            for i, msg in enumerate(messages):
                                if str(msg.id) == str(final_check.user_message_id):
                                    user_message_index = i
                                    break
                            
                            # 如果找到user_message，检查它之后的assistant message
                            if user_message_index is not None:
                                # 在user_message之后（索引更大）查找assistant message
                                for i in range(user_message_index + 1, len(messages)):
                                    msg = messages[i]
                                    if msg.content.get("role") == "assistant":
                                        logger.info("Task %s has response but no assistant_message_id, found assistant message %s in database, marking as done", 
                                                   final_check.task_id, msg.id)
                                        await registry.mark_done(final_check.task_id, assistant_message_id=msg.id)
                                        final_check = await registry.get(task_info.task_id) or final_check
                                        is_completed = True
                                        break
                        except Exception as e:
                            logger.warning("Failed to check database for assistant message for task %s: %s", 
                                         final_check.task_id, e)
                    
                    if is_completed:
                        if final_check.assistant_message_id and not final_check.done:
                            logger.info("Task %s appears completed (has assistant_message_id) but done flag not set, marking as done", 
                                       final_check.task_id)
                            await registry.mark_done(final_check.task_id, assistant_message_id=final_check.assistant_message_id)
                            final_check = await registry.get(task_info.task_id) or final_check
                        
                        logger.info("Task %s is actually completed (done=%s, assistant_message_id=%s)", 
                                   task_info.task_id, final_check.done, final_check.assistant_message_id)
                        break
                # 如果确实没有更新，继续等待（可能是任务在处理中但暂时没有输出）
                no_update_count = 0


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
