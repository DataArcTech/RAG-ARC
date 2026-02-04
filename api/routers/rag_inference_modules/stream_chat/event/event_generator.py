"""SSE event generator for stream chat."""
import json
import uuid
import asyncio
import time
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
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
from config.output_limits import CITATION_STREAM_MODE, CHAT_TOP_CHUNKS
from ..utils.history_manager import create_user_message, load_and_process_history
from ..deepsearch.deepsearch_handler import process_deepsearch
from ..rag.stream_processor import start_stream_processing
from ..task.task_registry import get_chat_task_registry
from ..task.task_helpers import (
    create_and_register_task,
    check_and_handle_cancellation,
    yield_cancellation_event,
    mark_task_completed,
    cache_deepsearch_event
)
from ..response.response_finalizer import _build_and_yield_final_response
from encapsulation.data_model.orm_models import ChatMessageStatus
from framework.thread_pool import get_thread_pool
from api.routers.rag_inference_handlers import get_message_handler, get_session_handler
from ..response.response_builder import (
    generate_mindmap_if_needed,
    ensure_non_empty_response,
    build_sources_and_evidence,
    create_assistant_message,
    generate_and_update_title,
    build_evidence_for_payload,
    convert_evidence_chunks_to_chunks,
    build_deepsearch_sources_for_frontend,
)
from ..utils.citation_stream import CitationStreamRenumberer

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
    task_info: Any = None,
    citation_stream: CitationStreamRenumberer | None = None,
) -> AsyncGenerator[str, None]:
    """Yield token events as SSE."""
    if not text:
        return

    text_to_emit = text
    if citation_stream is not None:
        text_to_emit = citation_stream.feed(text)
        if not text_to_emit:
            return

    for delta_piece in iter_text_deltas(text_to_emit):
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
    task_info: Any = None,
    citation_stream: CitationStreamRenumberer | None = None,
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch answer as streaming token events."""
    async for event in _yield_token_events(
        answer,
        chunk_id,
        model_name,
        created,
        request_id,
        response_parts,
        task_info,
        citation_stream,
    ):
        yield event
    if citation_stream is not None:
        flush_text = citation_stream.flush()
        if flush_text:
            async for event in _yield_token_events(
                flush_text,
                chunk_id,
                model_name,
                created,
                request_id,
                response_parts,
                task_info,
                None,
            ):
                yield event


async def _process_queue_events(
    queue: asyncio.Queue,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    stream_error: list[Exception | None],
    citation_stream: CitationStreamRenumberer | None = None,
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
                response_parts,
                None,
                citation_stream,
            ):
                yield event
            continue
    if citation_stream is not None:
        flush_text = citation_stream.flush()
        if flush_text:
            async for event in _yield_token_events(
                flush_text,
                chunk_id,
                model_name,
                created,
                request_id,
                response_parts,
                None,
                None,
            ):
                yield event


async def _process_queue_events_with_cache(
    queue: asyncio.Queue,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    response_parts: list[str],
    stream_error: list[Exception | None],
    task_info: Any,
    citation_stream: CitationStreamRenumberer | None = None,
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
                task_info,
                citation_stream,
            ):
                yield event
            continue
    if citation_stream is not None:
        flush_text = citation_stream.flush()
        if flush_text:
            async for event in _yield_token_events(
                flush_text,
                chunk_id,
                model_name,
                created,
                request_id,
                response_parts,
                task_info,
                None,
            ):
                yield event


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


async def _ensure_finalization_after_queue(
    response_parts: list[str],
    chunks: list,
    subgraph_data: Optional[Any],
    subgraph_info: Optional[Any],
    raw_llm_response: Optional[Any],
    raw_mindmap_response: Optional[Any],
    enable_deepsearch: bool,
    deepsearch_result: Optional[Any],
    deepsearch_sources_for_frontend: Optional[Any],
    deepsearch_citation_key_map: Optional[dict],
    return_subgraph: bool,
    query: str,
    session_id: Any,
    first_turn: bool,
    include_evidence: bool,
    task_info: Optional[Any],
    user_message: Any,
    rag_inference_handler: Any,
    deepsearch_trace_file_path: Optional[str],
    assistant_response_override: Optional[str] = None,
    citation_key_map_override: Optional[dict[int, int]] = None,
    assistant_response_renumbered: bool = False,
) -> None:
    """在队列处理完成后，确保最终化逻辑执行（后台任务）"""
    try:
        assistant_response = assistant_response_override or "".join(response_parts)
        if not assistant_response:
            return
        
        from ..response.response_finalizer import _ensure_finalization
        renumbered = assistant_response_renumbered and assistant_response_override is None
        await _ensure_finalization(
            assistant_response=assistant_response,
            chunks=chunks,
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            raw_llm_response=raw_llm_response,
            raw_mindmap_response=raw_mindmap_response,
            enable_deepsearch=enable_deepsearch,
            deepsearch_result=deepsearch_result,
            deepsearch_sources_for_frontend=deepsearch_sources_for_frontend,
            deepsearch_citation_key_map=deepsearch_citation_key_map,
            return_subgraph=return_subgraph,
            query=query,
            session_id=session_id,
            first_turn=first_turn,
            include_evidence=include_evidence,
            task_info=task_info,
            user_message=user_message,
            rag_inference_handler=rag_inference_handler,
            deepsearch_trace_file_path=deepsearch_trace_file_path,
            citation_key_map_override=citation_key_map_override,
            assistant_response_renumbered=renumbered,
        )
        logger.info("Background finalization completed for task %s", task_info.task_id if task_info else "unknown")
    except Exception as e:
        logger.error("Failed to ensure finalization in background task: %s", e, exc_info=True)


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
    request_id: str,
    task_info: Any = None
) -> AsyncGenerator[str, None]:
    """Yield sources event (SSE 倒数第三个事件流，sources 部分)."""
    sources_data = [s.model_dump() for s in sources_for_frontend]
    payload = {
        "type": "sources",
        "sources": sources_data,
        "citation_key_map": {str(k): v for k, v in citation_key_map.items()},
        "id": str(session_id)
    }

    # 存储sources事件到Redis，供恢复使用
    if task_info:
        try:
            registry = get_chat_task_registry()
            await registry.append_event(task_info.task_id, payload, event_type="sources")
        except Exception as e:
            logger.warning("Failed to store sources event to Redis: %s", e)
    
    # Log the complete SSE sources payload for debugging
    logger.info(
        "SSE sources event payload: sources_count=%d sources_keys=%s citation_key_map=%s session_id=%s",
        len(sources_data),
        [s.get("key") for s in sources_data],
        payload.get("citation_key_map"),
        str(session_id)
    )
    
    # Log full payload (truncated if too large)
    payload_str = json.dumps(payload, ensure_ascii=False)
    if len(payload_str) > 2000:
        logger.info("SSE sources event payload (truncated): %s...", payload_str[:2000])
    else:
        logger.info("SSE sources event payload (full): %s", payload_str)
    
    yield sse_json_wrapped(payload, request_id=request_id)


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
    from .event_initializer import (
        _initialize_chat_context,
        initialize_streaming_state,
        yield_initial_assistant_event
    )
    
    # 生成初始 assistant role 事件
    chunk_id = new_chatcmpl_id()
    created = now_epoch_seconds()
    async for event in yield_initial_assistant_event(chunk_id, model_name, created, request_id):
        yield event
    
    # 初始化聊天上下文
    user_message, history_messages, history_text, first_turn, task_info = await _initialize_chat_context(
        session_id, query, request_id
    )
    
    # 初始化流式处理状态
    loop = asyncio.get_running_loop()
    response_parts, queue, stream_error, prepared, rag_inference_handler, emit_progress = initialize_streaming_state(
        request_id, loop
    )

    citation_stream = None
    if CITATION_STREAM_MODE == "appearance":
        try:
            max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))
        except ValueError:
            max_sources = 5
        # LLM keys are assigned from the chunk list passed to the model (1..N).
        # Keep the streaming remap bounded so hallucinated keys don't create "missing source" citations.
        cap = CHAT_TOP_CHUNKS if isinstance(CHAT_TOP_CHUNKS, int) and CHAT_TOP_CHUNKS > 0 else max_sources
        citation_stream = CitationStreamRenumberer(max_key=min(max_sources, cap))
    
    # Process DeepSearch if enabled
    deepsearch_result = None
    deepsearch_trace_file_path = None
    deepsearch_sources_for_frontend = None
    deepsearch_citation_key_map: dict[int, int] = {}
    chunks = []
    assistant_response = ""
    subgraph_data = None
    subgraph_info = None
    raw_llm_response = None
    raw_mindmap_response = None
    
    if enable_deepsearch:
        from ..deepsearch.deepsearch_processor import process_deepsearch_with_events
        
        async for event in process_deepsearch_with_events(
            query, effective_owner, request_id, chunk_id, model_name, created,
            loop, task_info, session_id, user_message.id, response_parts, citation_stream
        ):
            if isinstance(event, tuple):
                # 最后返回结果元组
                deepsearch_result, deepsearch_trace_file_path, deepsearch_sources_for_frontend, \
                deepsearch_citation_key_map, chunks, assistant_response = event
                break
            yield event
    
    # 如果 DeepSearch 成功，已经设置了所有变量，直接跳转到响应构建
    # 如果 DeepSearch 未启用或失败，使用 RAG 系统
    if not enable_deepsearch or not deepsearch_result:
        from ..rag.rag_processor import process_rag_streaming, process_rag_queue_events, handle_rag_streaming_error
        
        # 启动 RAG 流处理
        if not await process_rag_streaming(
            query, effective_owner, return_subgraph, include_evidence, history_text,
            enable_web_search, queue, loop, prepared, stream_error, emit_progress,
            chunk_id, model_name, created, request_id, response_parts,
            task_info, session_id, user_message.id
        ):
            return  # 被取消
        
        # 创建最终化回调函数，在后台线程完成时自动执行
        async def finalization_callback():
            """后台线程完成时调用的最终化函数"""
            try:
                # 等待一小段时间，确保所有token都已处理
                await asyncio.sleep(0.5)
                
                # 检查是否有错误
                if stream_error[0]:
                    return
                
                # 检查是否有响应内容
                assistant_response_override = prepared.get("assistant_response")
                if not (assistant_response_override or "".join(response_parts)):
                    return
                # 需要 subgraph 但 prepared 里还没有时，不在此处落库，由主协程在生成 mindmap 后再创建，避免落库 subgraph_data=null
                if return_subgraph and prepared.get("subgraph_data") is None:
                    return
                # 从prepared中获取chunks等变量
                final_chunks = prepared.get("chunks") or []
                final_subgraph_data = prepared.get("subgraph_data")
                final_subgraph_info = prepared.get("subgraph_info")
                final_raw_llm_response = prepared.get("raw_llm_response")
                final_raw_mindmap_response = prepared.get("raw_mindmap_response")
                
                # 执行最终化
                await _ensure_finalization_after_queue(
                    response_parts, final_chunks, final_subgraph_data, final_subgraph_info,
                    final_raw_llm_response, final_raw_mindmap_response, enable_deepsearch,
                    deepsearch_result, deepsearch_sources_for_frontend,
                    deepsearch_citation_key_map, return_subgraph, query,
                    session_id, first_turn, include_evidence, task_info,
                    user_message,
                    rag_inference_handler,
                    deepsearch_trace_file_path,
                    assistant_response_override=assistant_response_override,
                    citation_key_map_override=(citation_stream.key_map if citation_stream else None),
                    assistant_response_renumbered=bool(citation_stream),
                )
                logger.info("Background finalization completed for task %s", task_info.task_id if task_info else "unknown")
            except Exception as e:
                logger.error("Error in finalization callback: %s", e, exc_info=True)
        
        # 将回调函数存储到prepared字典，供后台线程调用
        prepared["finalization_callback"] = finalization_callback
        
        # 处理队列事件
        try:
            async for event in process_rag_queue_events(
                queue, chunk_id, model_name, created, request_id, response_parts,
                stream_error, task_info, session_id, user_message.id, citation_stream
            ):
                yield event
        except Exception as e:
            # 即使客户端断开导致yield失败，后台线程也会在完成时执行最终化
            logger.warning("Error during queue event processing (client may have disconnected): %s", e)
        
        # 处理错误
        async for event in handle_rag_streaming_error(
            stream_error, task_info, session_id, user_message.id, request_id
        ):
            yield event
            return
        
        # 检查取消（在处理响应前）
        if await check_and_handle_cancellation(task_info, session_id, user_message.id):
            async for cancel_event in yield_cancellation_event(request_id):
                yield cancel_event
            return
        
        # 构建响应
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
        
        # 记录从 prepared 中获取的数据
        logger.info(
            "Retrieved from prepared: chunks_count=%d, has_subgraph_data=%s, return_subgraph=%s",
            len(chunks),
            subgraph_data is not None,
            return_subgraph
        )
    
    # 检查任务是否被取消（在生成 mindmap 前）
    if await check_and_handle_cancellation(task_info, session_id, user_message.id):
        async for cancel_event in yield_cancellation_event(request_id):
            yield cancel_event
        return
    
    # 构建最终响应
    # 使用try-except确保即使客户端断开，最终化逻辑也能执行
    try:
        async for event in _build_and_yield_final_response(
            assistant_response=assistant_response,
            chunks=chunks,
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            raw_llm_response=raw_llm_response,
            raw_mindmap_response=raw_mindmap_response,
            enable_deepsearch=enable_deepsearch,
            deepsearch_result=deepsearch_result,
            deepsearch_sources_for_frontend=deepsearch_sources_for_frontend,
            deepsearch_citation_key_map=deepsearch_citation_key_map,
            return_subgraph=return_subgraph,
            query=query,
            session_id=session_id,
            first_turn=first_turn,
            include_evidence=include_evidence,
            task_info=task_info,
            user_message=user_message,
            rag_inference_handler=rag_inference_handler,
            chunk_id=chunk_id,
            model_name=model_name,
            created=created,
            request_id=request_id,
            deepsearch_trace_file_path=deepsearch_trace_file_path,
            emit_final_text=(CITATION_STREAM_MODE != "appearance"),
            citation_key_map_override=(citation_stream.key_map if citation_stream else None),
            assistant_response_renumbered=bool(citation_stream),
        ):
            yield event
    except Exception as e:
        # 即使客户端断开导致yield失败，也要确保最终化逻辑执行
        logger.warning("Error during final response building (client may have disconnected): %s", e)
        # 直接调用最终化逻辑，不通过yield
        try:
            from ..response.response_finalizer import _ensure_finalization
            await _ensure_finalization(
                assistant_response=assistant_response,
                chunks=chunks,
                subgraph_data=subgraph_data,
                subgraph_info=subgraph_info,
                raw_llm_response=raw_llm_response,
                raw_mindmap_response=raw_mindmap_response,
                enable_deepsearch=enable_deepsearch,
                deepsearch_result=deepsearch_result,
                deepsearch_sources_for_frontend=deepsearch_sources_for_frontend,
                deepsearch_citation_key_map=deepsearch_citation_key_map,
                return_subgraph=return_subgraph,
                query=query,
                session_id=session_id,
                first_turn=first_turn,
                include_evidence=include_evidence,
                task_info=task_info,
                user_message=user_message,
                rag_inference_handler=rag_inference_handler,
                deepsearch_trace_file_path=deepsearch_trace_file_path,
                citation_key_map_override=(citation_stream.key_map if citation_stream else None),
                assistant_response_renumbered=bool(citation_stream),
            )
        except Exception as finalize_error:
            logger.error("Failed to ensure finalization: %s", finalize_error, exc_info=True)
