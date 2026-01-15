"""SSE event generator for stream chat."""
import json
import uuid
import asyncio
import time
import logging
from datetime import datetime
from typing import Any, AsyncGenerator
from api.sse import (
    delta_envelope,
    iter_text_deltas,
    new_chatcmpl_id,
    now_epoch_seconds,
    openai_chat_completion_chunk,
    sse_done,
    sse_json_wrapped,
)
from api.routers.rag_inference_models import build_stream_chat_payload
from api.routers.rag_inference_handlers import get_rag_inference_handler
from .history_manager import create_user_message, load_and_process_history
from .deepsearch_handler import process_deepsearch
from .stream_processor import start_stream_processing
from .task_registry import get_chat_task_registry
from encapsulation.data_model.orm_models import ChatMessageStatus
from framework.thread_pool import get_thread_pool
from api.routers.rag_inference_handlers import get_message_handler, get_session_handler
from .response_builder import (
    generate_mindmap_if_needed,
    ensure_non_empty_response,
    build_sources_and_evidence,
    create_assistant_message,
    generate_and_update_title,
    build_evidence_for_payload,
    convert_evidence_chunks_to_chunks,
    build_deepsearch_sources_for_frontend,
)

logger = logging.getLogger(__name__)


def _create_progress_emitter(
    request_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop
) -> callable:
    """Create progress emitter function."""
    progress_seq = 0
    
    def emit_progress(payload: dict[str, Any]) -> None:
        nonlocal progress_seq
        progress_seq += 1
        envelope = dict(payload or {})
        envelope.setdefault("v", 1)
        envelope.setdefault("type", "progress")
        envelope.setdefault("ts_ms", int(time.time() * 1000))
        envelope.setdefault("request_id", request_id)
        envelope.setdefault("seq", progress_seq)
        asyncio.run_coroutine_threadsafe(
            queue.put({"kind": "progress", "payload": envelope}),
            loop
        )
    
    return emit_progress


async def _yield_progress_event(
    payload: dict[str, Any],
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield progress event as SSE."""
    tool_calls = [{
        "index": 0,
        "id": f"call_progress_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": "rag_arc_progress",
            "arguments": json.dumps(
                payload or {},
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ),
        },
    }]
    yield sse_json_wrapped(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(role=None, tool_calls=tool_calls),
        ),
        request_id=request_id
    )


async def _yield_token_events(
    text: str,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    task_info: Any = None
) -> AsyncGenerator[str, None]:
    """Yield token events as SSE."""
    if not text:
        return
    
    for delta_piece in iter_text_deltas(text):
        response_parts.append(delta_piece)
        event = sse_json_wrapped(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(role=None, content=delta_piece),
            ),
            request_id=request_id
        )
        
        # 缓存事件到任务注册表
        if task_info:
            try:
                registry = get_chat_task_registry()
                event_data = {
                    "delta": {"content": delta_piece}
                }
                await registry.append_event(task_info.task_id, event_data, event_type="data")
            except Exception:
                pass
        
        yield event
        await asyncio.sleep(0)


async def _yield_deepsearch_answer_stream(
    answer: str,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    task_info: Any = None
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch answer as streaming token events."""
    async for event in _yield_token_events(
        answer,
        chunk_id,
        model_name,
        created,
        request_id,
        response_parts,
        task_info
    ):
        yield event


async def _process_queue_events(
    queue: asyncio.Queue,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    stream_error: list[Exception | None]
) -> AsyncGenerator[str, None]:
    """Process events from queue."""
    while True:
        item = await queue.get()
        if item is None:
            break
        
        if isinstance(item, dict) and item.get("kind") == "progress":
            async for event in _yield_progress_event(
                item.get("payload") or {},
                chunk_id,
                model_name,
                created,
                request_id
            ):
                yield event
            continue
        
        if isinstance(item, dict) and item.get("kind") == "token":
            async for event in _yield_token_events(
                str(item.get("text") or ""),
                chunk_id,
                model_name,
                created,
                request_id,
                response_parts
            ):
                yield event
            continue


async def _process_queue_events_with_cache(
    queue: asyncio.Queue,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    stream_error: list[Exception | None],
    task_info: Any
) -> AsyncGenerator[str, None]:
    """Process events from queue with task event caching."""
    while True:
        item = await queue.get()
        if item is None:
            break
        
        if isinstance(item, dict) and item.get("kind") == "progress":
            async for event in _yield_progress_event(
                item.get("payload") or {},
                chunk_id,
                model_name,
                created,
                request_id
            ):
                # 缓存进度事件
                if task_info:
                    try:
                        registry = get_chat_task_registry()
                        try:
                            if '\n\n' in event:
                                event_data = json.loads(event.split('\n\n', 1)[-1])
                            else:
                                event_data = {"raw": event}
                        except json.JSONDecodeError:
                            event_data = {"raw": event}
                        await registry.append_event(task_info.task_id, event_data, event_type="progress")
                    except Exception:
                        pass
                yield event
            continue
        
        if isinstance(item, dict) and item.get("kind") == "token":
            async for event in _yield_token_events(
                str(item.get("text") or ""),
                chunk_id,
                model_name,
                created,
                request_id,
                response_parts,
                task_info
            ):
                yield event
            continue


async def _yield_error_event(
    error: Exception,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield error event."""
    yield sse_json_wrapped(
        {"error": {"message": str(error)}},
        request_id=request_id,
        code=500,
        message="error"
    )
    yield sse_done()


async def _yield_title_event(
    title: str,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield title event."""
    yield sse_json_wrapped(
        {"type": "title", "title": title},
        request_id=request_id
    )


async def _yield_sources_event(
    sources_for_frontend: list,
    citation_key_map: dict[int, int],
    session_id: uuid.UUID,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield sources event."""
    yield sse_json_wrapped(
        {
            "type": "sources",
            "sources": [s.model_dump() for s in sources_for_frontend],
            "citation_key_map": {str(k): v for k, v in citation_key_map.items()},
            "id": str(session_id)
        },
        request_id=request_id
    )


async def _yield_payload_event(
    assistant_message: Any,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield payload event."""
    payload, subgraph_for_outer = build_stream_chat_payload(
        assistant_message,
        assistant_message.source_file_ids or [],
        subgraph=None,
        evidence=None,
    )
    
    chunk_data = openai_chat_completion_chunk(
        chunk_id=chunk_id,
        model=model_name,
        created=created,
        delta=delta_envelope(role=None, tool_calls=[{
            "index": 0,
            "id": f"call_{assistant_message.id}",
            "type": "function",
            "function": {
                "name": "rag_arc_payload",
                "arguments": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":")
                ),
            },
        }]),
    )
    
    # Add subgraph at outer level if present
    if subgraph_for_outer is not None:
        chunk_data["subgraph"] = subgraph_for_outer
    
    yield sse_json_wrapped(
        chunk_data,
        request_id=request_id
    )


async def _yield_finish_event(
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield finish event."""
    yield sse_json_wrapped(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(),
            finish_reason="stop",
        ),
        request_id=request_id
    )
    yield sse_done()


async def generate_sse_events(
    session_id: uuid.UUID,
    query: str,
    effective_owner: Any,
    return_subgraph: bool,
    include_evidence: bool,
    enable_web_search: bool,
    enable_deepsearch: bool,
    model_name: str,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Generate SSE events for stream chat."""
    chunk_id = new_chatcmpl_id()
    created = now_epoch_seconds()
    
    # Initial assistant role event
    yield sse_json_wrapped(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(role="assistant", content=""),
        ),
        request_id=request_id
    )
    
    # Create user message and load history
    user_message = await create_user_message(session_id, query)
    history_messages, history_text, first_turn = await load_and_process_history(
        session_id,
        user_message.id
    )
    
    # 创建任务并更新状态
    registry = get_chat_task_registry()
    task_info = None
    try:
        task_info = await registry.create(
            session_id=session_id,
            user_message_id=user_message.id,
            query=query,
            request_id=request_id
        )
        
        # 更新数据库状态
        await get_thread_pool().run_blocking(
            get_message_handler().update_message,
            user_message.id,
            {
                "status": ChatMessageStatus.PROCESSING,
                "task_id": task_info.task_id,
                "request_id": request_id
            }
        )
        
        await get_thread_pool().run_blocking(
            get_session_handler().update_session,
            session_id,
            {
                "current_task_id": task_info.task_id,
                "current_task_status": "PROCESSING",
                "current_task_started_at": datetime.now()
            }
        )
    except ValueError as e:
        # 已有任务在进行中，抛出异常让上层处理
        from fastapi import HTTPException, status
        logger.warning("Task creation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "Another task is in progress",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error("Failed to create task: %s", e)
        # 继续执行，但不进行任务管理
    
    # Initialize queues and state
    response_parts: list[str] = []
    queue: asyncio.Queue[object | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stream_error: list[Exception | None] = [None]
    prepared: dict[str, Any] = {}
    rag_inference_handler = get_rag_inference_handler()
    
    emit_progress = _create_progress_emitter(request_id, queue, loop)
    
    # Process DeepSearch if enabled
    deepsearch_result = None
    deepsearch_trace_file_path = None
    deepsearch_sources_for_frontend = None
    deepsearch_citation_key_map: dict[int, int] = {}
    if enable_deepsearch:
        deepsearch_result_container, trace_file_path_container, deepsearch_gen = await process_deepsearch(
            query,
            str(effective_owner),
            request_id,
            chunk_id,
            model_name,
            created,
            loop
        )
        # 先迭代生成器以获取进度事件，生成器完成后 deepsearch_result_container[0] 会被设置
        async for event in deepsearch_gen:
            # 检查任务是否被取消
            if task_info:
                updated_info = await registry.get(task_info.task_id)
                if updated_info and updated_info.cancelled:
                    logger.info("Task cancelled during DeepSearch, stopping")
                    await registry.mark_done(task_info.task_id, error="Task cancelled by user")
                    await get_thread_pool().run_blocking(
                        get_message_handler().update_message,
                        user_message.id,
                        {"status": ChatMessageStatus.CANCELLED}
                    )
                    await get_thread_pool().run_blocking(
                        get_session_handler().update_session,
                        session_id,
                        {
                            "current_task_id": None,
                            "current_task_status": None,
                            "current_task_started_at": None
                        }
                    )
                    yield sse_json_wrapped({
                        "type": "task_cancelled",
                        "message": "Task cancelled by user"
                    }, request_id)
                    yield sse_done()
                    return
            
            # 缓存 DeepSearch 事件
            if task_info:
                try:
                    registry = get_chat_task_registry()
                    # 尝试解析事件数据
                    try:
                        if '\n\n' in event:
                            event_data = json.loads(event.split('\n\n', 1)[-1])
                        else:
                            event_data = {"raw": event}
                    except json.JSONDecodeError:
                        event_data = {"raw": event}
                    await registry.append_event(task_info.task_id, event_data, event_type="progress")
                except Exception:
                    pass
            yield event
        
        # 生成器完成后，从容器中获取结果
        deepsearch_result = deepsearch_result_container[0]
        deepsearch_trace_file_path = trace_file_path_container[0]
        
        # 如果 DeepSearch 成功，直接使用其结果，不再调用 RAG 系统
        if deepsearch_result:
            from core.presentation.deepsearch_payload import trim_deepsearch_payload

            trimmed = trim_deepsearch_payload(deepsearch_result, include_evidence=False)
            report = trimmed.get("report") if isinstance(trimmed, dict) else None
            if not isinstance(report, dict):
                report = {}

            # DeepSearch service 返回的结构是: {"plan": ..., "reasoning": ..., "report": ..., "state": ...}
            # answer 在 report 字典中
            raw_answer = report.get("answer") or ""
            
            # 确保 answer 是字符串类型
            if not isinstance(raw_answer, str):
                if isinstance(raw_answer, dict):
                    # 如果是字典，尝试提取文本内容
                    raw_answer = raw_answer.get("text") or raw_answer.get("content") or raw_answer.get("short_answer") or str(raw_answer)
                else:
                    raw_answer = str(raw_answer)
            
            deepsearch_answer = raw_answer.strip() if raw_answer else ""
            # evidences 也在 report 中
            deepsearch_evidences = report.get("evidences") or []
            deepsearch_sources_for_frontend, deepsearch_citation_key_map = build_deepsearch_sources_for_frontend(report)
            
            if deepsearch_answer:
                logger.info("DeepSearch completed, using DeepSearch answer directly (length=%d, type=%s)", 
                          len(deepsearch_answer), type(deepsearch_answer).__name__)
                
                # 流式输出 DeepSearch 答案
                async for event in _yield_deepsearch_answer_stream(
                    deepsearch_answer,
                    chunk_id,
                    model_name,
                    created,
                    request_id,
                    response_parts,
                    task_info
                ):
                    yield event
                
                # 将 DeepSearch evidences 转换为 chunks
                chunks = convert_evidence_chunks_to_chunks(deepsearch_evidences)
                assistant_response = "".join(response_parts)
                subgraph_data = None
                subgraph_info = None
                raw_llm_response = None
                raw_mindmap_response = None

                # 跳过 RAG 系统的 stream_chat 调用，直接构建响应
                # 继续执行后续的响应构建逻辑（mindmap、sources、message 等）
                # 注意：这里直接跳转到后续的响应构建逻辑
            else:
                logger.warning("DeepSearch completed but no answer found, falling back to RAG system")
                deepsearch_result = None  # 标记为失败，使用 RAG 系统
        else:
            logger.warning("DeepSearch failed or returned None, falling back to RAG system")
            deepsearch_result = None
    
    # 如果 DeepSearch 成功，已经设置了所有变量，直接跳转到响应构建
    # 如果 DeepSearch 未启用或失败，使用 RAG 系统
    if not enable_deepsearch or not deepsearch_result:
        # 检查任务是否被取消（在启动 RAG 处理前）
        if task_info:
            updated_info = await registry.get(task_info.task_id)
            if updated_info and updated_info.cancelled:
                logger.info("Task cancelled before RAG processing, stopping")
                await registry.mark_done(task_info.task_id, error="Task cancelled by user")
                await get_thread_pool().run_blocking(
                    get_message_handler().update_message,
                    user_message.id,
                    {"status": ChatMessageStatus.CANCELLED}
                )
                await get_thread_pool().run_blocking(
                    get_session_handler().update_session,
                    session_id,
                    {
                        "current_task_id": None,
                        "current_task_status": None,
                        "current_task_started_at": None
                    }
                )
                yield sse_json_wrapped({
                    "type": "task_cancelled",
                    "message": "Task cancelled by user"
                }, request_id)
                yield sse_done()
                return
        
        # Start stream processing
        start_stream_processing(
            query,
            effective_owner,
            return_subgraph,
            include_evidence,
            history_text,
            enable_web_search,
            queue,
            loop,
            prepared,
            stream_error,
            emit_progress
        )
        
        # Process queue events (with task event caching)
        async for event in _process_queue_events_with_cache(
            queue,
            chunk_id,
            model_name,
            created,
            request_id,
            response_parts,
            stream_error,
            task_info
        ):
            # 检查任务是否被取消
            if task_info:
                updated_info = await registry.get(task_info.task_id)
                if updated_info and updated_info.cancelled:
                    logger.info("Task cancelled during execution, stopping event processing")
                    # 标记为完成（已取消）
                    await registry.mark_done(task_info.task_id, error="Task cancelled by user")
                    await get_thread_pool().run_blocking(
                        get_message_handler().update_message,
                        user_message.id,
                        {"status": ChatMessageStatus.CANCELLED}
                    )
                    await get_thread_pool().run_blocking(
                        get_session_handler().update_session,
                        session_id,
                        {
                            "current_task_id": None,
                            "current_task_status": None,
                            "current_task_started_at": None
                        }
                    )
                    # 发送取消事件
                    yield sse_json_wrapped({
                        "type": "task_cancelled",
                        "message": "Task cancelled by user"
                    }, request_id)
                    yield sse_done()
                    return
            yield event
        
        # Handle errors
        if stream_error[0] is not None:
            # 任务失败，更新状态
            if task_info:
                try:
                    error_msg = str(stream_error[0])
                    await registry.mark_done(task_info.task_id, error=error_msg)
                    await get_thread_pool().run_blocking(
                        get_message_handler().update_message,
                        user_message.id,
                        {"status": ChatMessageStatus.FAILED}
                    )
                    await get_thread_pool().run_blocking(
                        get_session_handler().update_session,
                        session_id,
                        {
                            "current_task_id": None,
                            "current_task_status": None,
                            "current_task_started_at": None
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to update task error status: %s", e)
            
            async for event in _yield_error_event(stream_error[0], request_id):
                yield event
            return
        
        # 检查任务是否被取消（在处理响应前）
        if task_info:
            updated_info = await registry.get(task_info.task_id)
            if updated_info and updated_info.cancelled:
                logger.info("Task cancelled before building response, stopping")
                await registry.mark_done(task_info.task_id, error="Task cancelled by user")
                await get_thread_pool().run_blocking(
                    get_message_handler().update_message,
                    user_message.id,
                    {"status": ChatMessageStatus.CANCELLED}
                )
                await get_thread_pool().run_blocking(
                    get_session_handler().update_session,
                    session_id,
                    {
                        "current_task_id": None,
                        "current_task_status": None,
                        "current_task_started_at": None
                    }
                )
                yield sse_json_wrapped({
                    "type": "task_cancelled",
                    "message": "Task cancelled by user"
                }, request_id)
                yield sse_done()
                return
        
        # Build response
        assistant_response = "".join(response_parts)
        if not assistant_response:
            logger.warning(
                "SSE assistant_response is empty after streaming; "
                "query=%r owner_id=%s history_len=%d prepared_chunks=%d",
                query,
                str(effective_owner),
                len(history_messages),
                len(prepared.get("chunks") or []),
            )
        
        chunks = prepared.get("chunks") or []
        subgraph_data = prepared.get("subgraph_data")
        subgraph_info = prepared.get("subgraph_info")
        raw_llm_response = prepared.get("raw_llm_response")
        raw_mindmap_response = prepared.get("raw_mindmap_response")
    
    # 检查任务是否被取消（在生成 mindmap 前）
    if task_info:
        updated_info = await registry.get(task_info.task_id)
        if updated_info and updated_info.cancelled:
            logger.info("Task cancelled before generating mindmap, stopping")
            await registry.mark_done(task_info.task_id, error="Task cancelled by user")
            await get_thread_pool().run_blocking(
                get_message_handler().update_message,
                user_message.id,
                {"status": ChatMessageStatus.CANCELLED}
            )
            await get_thread_pool().run_blocking(
                get_session_handler().update_session,
                session_id,
                {
                    "current_task_id": None,
                    "current_task_status": None,
                    "current_task_started_at": None
                }
            )
            yield sse_json_wrapped({
                "type": "task_cancelled",
                "message": "Task cancelled by user"
            }, request_id)
            yield sse_done()
            return
    
    # Generate mindmap if needed (keep upstream subgraph when already provided).
    if return_subgraph and subgraph_data is None:
        subgraph_data, raw_mindmap_response = await generate_mindmap_if_needed(
            return_subgraph,
            query,
            assistant_response,
            chunks,
            rag_inference_handler
        )
    
    # Ensure non-empty response
    assistant_response, is_fallback_response = ensure_non_empty_response(assistant_response)
    
    # Build sources and evidence
    if enable_deepsearch and deepsearch_result and deepsearch_sources_for_frontend is not None:
        sources_for_frontend = deepsearch_sources_for_frontend
        citation_key_map = deepsearch_citation_key_map
    else:
        assistant_response, sources_for_frontend, citation_key_map = await build_sources_and_evidence(
            chunks,
            subgraph_data,
            subgraph_info,
            rag_inference_handler,
            assistant_response,
            is_fallback_response
        )
    
    # 再次检查任务是否被取消（在创建消息前）
    if task_info:
        updated_info = await registry.get(task_info.task_id)
        if updated_info and updated_info.cancelled:
            logger.info("Task cancelled before creating assistant message, stopping")
            await registry.mark_done(task_info.task_id, error="Task cancelled by user")
            await get_thread_pool().run_blocking(
                get_message_handler().update_message,
                user_message.id,
                {"status": ChatMessageStatus.CANCELLED}
            )
            await get_thread_pool().run_blocking(
                get_session_handler().update_session,
                session_id,
                {
                    "current_task_id": None,
                    "current_task_status": None,
                    "current_task_started_at": None
                }
            )
            yield sse_json_wrapped({
                "type": "task_cancelled",
                "message": "Task cancelled by user"
            }, request_id)
            yield sse_done()
            return
    
    # Create assistant message
    assistant_message = await create_assistant_message(
        session_id,
        assistant_response,
        sources_for_frontend,
        subgraph_data,
        return_subgraph,
        raw_llm_response,
        raw_mindmap_response,
        deepsearch_trace_file_path=deepsearch_trace_file_path if enable_deepsearch else None
    )
    
    # 任务完成，更新状态
    if task_info:
        try:
            await registry.mark_done(
                task_info.task_id,
                assistant_message_id=assistant_message.id
            )
            
            # 更新数据库状态
            await get_thread_pool().run_blocking(
                get_message_handler().update_message,
                user_message.id,
                {"status": ChatMessageStatus.COMPLETED}
            )
            
            await get_thread_pool().run_blocking(
                get_session_handler().update_session,
                session_id,
                {
                    "current_task_id": None,
                    "current_task_status": None,
                    "current_task_started_at": None
                }
            )
        except Exception as e:
            logger.warning("Failed to update task completion status: %s", e)
    
    # Generate title if first turn
    if first_turn:
        title = await generate_and_update_title(
            first_turn,
            session_id,
            query,
            assistant_response,
            rag_inference_handler
        )
        if title:
            async for event in _yield_title_event(title, request_id):
                yield event
    
    # Yield sources event
    async for event in _yield_sources_event(
        sources_for_frontend,
        citation_key_map,
        session_id,
        request_id
    ):
        yield event
    
    # Build evidence for payload
    evidence = build_evidence_for_payload(
        include_evidence,
        chunks,
        subgraph_data,
        subgraph_info,
        rag_inference_handler
    )
    
    # Update payload with evidence (subgraph is extracted separately)
    payload, subgraph_for_outer = build_stream_chat_payload(
        assistant_message,
        chunks,
        subgraph=subgraph_data if return_subgraph else None,
        evidence=evidence,
    )
    
    # Build the chunk with tool_calls
    chunk_data = openai_chat_completion_chunk(
        chunk_id=chunk_id,
        model=model_name,
        created=created,
        delta=delta_envelope(role=None, tool_calls=[{
            "index": 0,
            "id": f"call_{assistant_message.id}",
            "type": "function",
            "function": {
                "name": "rag_arc_payload",
                "arguments": json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":")
                ),
            },
        }]),
    )
    
    # Add subgraph at outer level (data level) if present
    if subgraph_for_outer is not None:
        chunk_data["subgraph"] = subgraph_for_outer
    
    yield sse_json_wrapped(
        chunk_data,
        request_id=request_id
    )
    
    async for event in _yield_finish_event(chunk_id, model_name, created, request_id):
        yield event
