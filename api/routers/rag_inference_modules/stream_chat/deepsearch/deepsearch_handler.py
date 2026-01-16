"""DeepSearch handling for stream chat."""
import json
import uuid
import asyncio
import time
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, AsyncGenerator, List
from framework.register import Register
from api.sse import (
    delta_envelope,
    openai_chat_completion_chunk,
    sse_json_wrapped,
)
from core.deepsearch.trace import TraceEmitter, TraceEvent, set_trace_emitter, reset_trace_emitter
from api.routers.deepsearch_weaver_render import render_trace_payload, weaver_block

logger = logging.getLogger(__name__)


def _generate_trace_message(trace_event: TraceEvent) -> str:
    """Generate a human-readable message from trace event."""
    tag = trace_event.tag
    meta = trace_event.meta or {}
    
    if tag == "think":
        stage = meta.get("stage", "")
        if stage == "plan":
            return "正在生成搜索计划..."
        elif stage == "question_classification":
            return "正在分析问题类型..."
        elif stage == "think_init":
            return "正在初始化思考..."
        elif stage == "reflection":
            step_id = meta.get("step_id", "")
            return f"正在反思步骤 {step_id} 的结果..."
        else:
            return "正在思考..."
    
    elif tag == "write_outline":
        return "已生成搜索计划大纲"
    
    elif tag == "tool_call":
        tool_name = meta.get("tool_name", "")
        plan_step = meta.get("plan_step", "")
        
        # 简化工具名称显示
        if tool_name == "graph.think":
            return "正在进行图谱推理..."
        elif tool_name == "graph_adapter.query":
            return "正在查询知识图谱..."
        elif tool_name.startswith("graph"):
            return "正在执行图谱操作..."
        elif tool_name.startswith("web") or tool_name.startswith("search"):
            return "正在进行网络搜索..."
        elif tool_name.startswith("text"):
            return "正在处理文本..."
        else:
            if plan_step:
                return f"正在执行步骤 {plan_step}..."
            return f"正在调用工具 {tool_name}..."
    
    elif tag == "tool_response":
        tool_name = meta.get("tool_name", "")
        ok = meta.get("ok", True)
        
        if not ok:
            return f"工具 {tool_name} 调用失败"
        
        if tool_name == "graph_adapter.query":
            evidence_count = meta.get("evidence_count", 0)
            return f"已从知识图谱获取 {evidence_count} 条证据"
        elif tool_name.startswith("graph"):
            return "图谱操作完成"
        elif tool_name.startswith("web") or tool_name.startswith("search"):
            return "网络搜索完成"
        else:
            return f"工具 {tool_name} 调用完成"
    
    elif tag == "write":
        return "正在写入内容..."
    
    elif tag == "progress":
        return "处理中..."
    
    elif tag == "terminate":
        return "处理终止"
    
    else:
        return f"DeepSearch 执行中 ({tag})..."


def _save_trace_events_to_file(
    trace_events: List[TraceEvent],
    request_id: str,
    run_id: str,
    query: str,
    deepsearch_result: dict[str, Any] | None
) -> str | None:
    """Save trace events to a JSON file.
    
    Returns:
        Path to the saved file, or None if saving failed.
    """
    try:
        # 确定存储目录
        # 优先使用环境变量 DEEPSEARCH_TRACE_STORAGE_PATH
        # 如果没有设置，默认使用 ./local/deepsearch_traces（与 artifacts 目录同级）
        base_dir = os.getenv("DEEPSEARCH_TRACE_STORAGE_PATH", "./local/deepsearch_traces")
        storage_dir = Path(base_dir).expanduser()
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名：使用 run_id 或 request_id，加上时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = run_id or request_id
        filename = f"trace_{file_id}_{timestamp}.json"
        file_path = storage_dir / filename
        
        rendered_result = deepsearch_result
        try:
            if isinstance(deepsearch_result, dict):
                report = deepsearch_result.get("report")
                if isinstance(report, dict):
                    raw_answer = report.get("answer")
                    if isinstance(raw_answer, str) and raw_answer.strip():
                        from core.deepsearch.report.sup_citations import format_answer_with_references

                        structured = report.get("structured_report")
                        citations = structured.get("citations") if isinstance(structured, dict) else None
                        evidences = report.get("evidences") if isinstance(report.get("evidences"), list) else None

                        converted, sources, citation_key_map = format_answer_with_references(
                            raw_answer,
                            citations=citations if isinstance(citations, list) else [],
                            evidences=evidences,
                        )
                        rendered_result = json.loads(json.dumps(deepsearch_result, ensure_ascii=False, default=str))
                        rendered_report = rendered_result.get("report") if isinstance(rendered_result, dict) else None
                        if isinstance(rendered_report, dict):
                            rendered_report.setdefault("answer_raw", raw_answer)
                            rendered_report["answer"] = converted
                            if sources:
                                rendered_report["sources"] = sources
                            if citation_key_map:
                                rendered_report["citation_key_map"] = citation_key_map
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to render DeepSearch answer for trace storage: %s", exc)
            rendered_result = deepsearch_result

        # 准备要保存的数据
        trace_data = {
            "metadata": {
                "request_id": request_id,
                "run_id": run_id or request_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "total_events": len(trace_events),
            },
            "deepsearch_result": rendered_result,
            "trace_events": [
                {
                    "tag": event.tag,
                    "content": event.content,
                    "meta": event.meta,
                }
                for event in trace_events
            ]
        }
        
        # 保存为 JSON 文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                trace_data,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        
        logger.info(
            "Saved %d trace events to %s (request_id=%s, run_id=%s)",
            len(trace_events),
            file_path,
            request_id,
            run_id or request_id
        )
        return str(file_path)
    except Exception as e:
        logger.error("Failed to save trace events to file: %s", e, exc_info=True)
        return None


def load_trace_events_from_file(trace_file_path: str) -> dict[str, Any] | None:
    """Load trace events from a JSON file.
    
    Args:
        trace_file_path: Path to the trace events JSON file
        
    Returns:
        Dictionary containing trace data, or None if loading failed.
    """
    try:
        file_path = Path(trace_file_path).expanduser()
        if not file_path.exists():
            logger.warning("Trace file not found: %s", trace_file_path)
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        
        logger.debug("Loaded trace events from %s", trace_file_path)
        return trace_data
    except Exception as e:
        logger.error("Failed to load trace events from file %s: %s", trace_file_path, e, exc_info=True)
        return None


class InMemoryTraceEmitter:
    """In-memory trace emitter that collects events in a queue for real-time processing and storage."""
    
    def __init__(self, trace_queue: asyncio.Queue[TraceEvent | None]):
        self.trace_queue = trace_queue
        self.collected_events: List[TraceEvent] = []  # 收集所有 trace events 用于存储
    
    async def emit(self, event: TraceEvent) -> None:
        """Emit a trace event to the queue and collect it for storage."""
        try:
            await self.trace_queue.put(event)
            # 同时收集事件用于后续存储
            self.collected_events.append(event)
        except Exception as e:
            logger.error("Error emitting trace event: %s", e, exc_info=True)
    
    def get_all_events(self) -> List[TraceEvent]:
        """Get all collected trace events."""
        return self.collected_events.copy()


def _build_progress_info(stage: str, metadata: dict, state: Any, request_id: str) -> dict[str, Any]:
    """Build progress info based on DeepSearch stage."""
    progress_info = {
        "stage": "deepsearch",
        "deepsearch_stage": stage,
        "status": "running",
    }
    
    if stage == "planned":
        progress_info["message"] = "正在生成搜索计划..."
        if "step_count" in metadata:
            progress_info["plan_steps_count"] = metadata.get("step_count")
        if hasattr(state, "plan_steps") and state.plan_steps:
            progress_info["plan_steps"] = state.plan_steps
        if hasattr(state, "plan_metadata") and state.plan_metadata:
            progress_info["plan_metadata"] = state.plan_metadata
    elif stage == "reasoned":
        progress_info["message"] = "正在进行图谱推理..."
        if hasattr(state, "reasoning_trace") and state.reasoning_trace:
            progress_info["reasoning_trace"] = state.reasoning_trace
            reasoning_steps = state.reasoning_trace.get("reasoning_steps", [])
            if reasoning_steps:
                progress_info["reasoning_steps"] = reasoning_steps
                progress_info["reasoning_steps_count"] = len(reasoning_steps)
            tool_results = state.reasoning_trace.get("tool_results", [])
            if tool_results:
                progress_info["tool_results"] = tool_results
                progress_info["tool_calls_count"] = len(tool_results)
    elif stage == "gap_evaluated":
        progress_info["message"] = "正在检测知识缺口..."
        if hasattr(state, "gap_result") and state.gap_result:
            progress_info["gap_result"] = state.gap_result
        if "should_trigger_external" in metadata:
            progress_info["should_trigger_external"] = metadata.get("should_trigger_external")
    elif stage == "external_invoked":
        progress_info["message"] = "正在进行外部搜索..."
        if hasattr(state, "external_calls") and state.external_calls:
            progress_info["external_calls"] = state.external_calls
            progress_info["external_calls_count"] = len(state.external_calls)
    elif stage == "reported":
        progress_info["message"] = "正在生成报告..."
        if hasattr(state, "report_payload") and state.report_payload:
            progress_info["report_payload"] = state.report_payload
    elif stage == "quality_gated":
        progress_info["message"] = "正在进行质量检查..."
        if hasattr(state, "quality_gates") and state.quality_gates:
            progress_info["quality_gates"] = state.quality_gates
            progress_info["quality_gates_count"] = len(state.quality_gates)
    elif stage == "done":
        progress_info["status"] = "completed"
        progress_info["message"] = "DeepSearch 完成"
        if hasattr(state, "run_id"):
            progress_info["run_id"] = state.run_id
    elif stage == "failed":
        progress_info["status"] = "failed"
        progress_info["message"] = "DeepSearch 执行失败"
        if hasattr(state, "errors") and state.errors:
            progress_info["errors"] = state.errors
    elif stage == "created":
        progress_info["message"] = "DeepSearch 初始化..."
        if hasattr(state, "run_id"):
            progress_info["run_id"] = state.run_id
    else:
        progress_info["message"] = f"DeepSearch 执行中（阶段: {stage}）..."
    
    if "plan_id" in metadata:
        progress_info["plan_id"] = metadata.get("plan_id")
    
    return progress_info


def _create_stage_listener(
    request_id: str,
    loop: asyncio.AbstractEventLoop,
    deepsearch_progress_queue: asyncio.Queue,
    emit_deepsearch_progress: callable
) -> callable:
    """Create DeepSearch stage listener."""
    def deepsearch_stage_listener(record: dict[str, Any], state: Any) -> None:
        try:
            stage = record.get("stage", "unknown")
            metadata = record.get("metadata", {})
            logger.info("DeepSearch stage listener: stage=%s, request_id=%s", stage, request_id)
            progress_info = _build_progress_info(stage, metadata, state, request_id)
            asyncio.run_coroutine_threadsafe(
                emit_deepsearch_progress(progress_info),
                loop
            )
        except Exception as e:
            logger.error("DeepSearch stage listener error: %s", e, exc_info=True)
    return deepsearch_stage_listener


async def _yield_deepsearch_progress(
    progress_payload: dict[str, Any],
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch progress as SSE event."""
    tool_calls = [{
        "index": 0,
        "id": f"call_deepsearch_progress_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": "rag_arc_progress",
            "arguments": json.dumps(
                progress_payload,
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


async def _yield_trace_event(
    trace_event: TraceEvent,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    run_id: str
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch trace event as SSE event (weaver format)."""
    # 渲染 trace event
    rendered_tag, rendered_content = render_trace_payload(
        trace_tag=trace_event.tag,
        content=trace_event.content,
        run_id=run_id
    )
    
    # 包装成 weaver block
    weaver_content = weaver_block(rendered_tag, rendered_content)
    
    # 生成人类可读的 message
    message = _generate_trace_message(trace_event)
    
    # 作为 tool_call 发送
    tool_calls = [{
        "index": 0,
        "id": f"call_deepsearch_trace_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": "rag_arc_trace",
            "arguments": json.dumps(
                {
                    "tag": rendered_tag,
                    "content": weaver_content,
                    "message": message,
                    "meta": trace_event.meta,
                },
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


async def _handle_deepsearch_task_completion(
    deepsearch_task: asyncio.Task,
    trace_emitter: InMemoryTraceEmitter,
    deepsearch_result_container: list[dict[str, Any] | None],
    trace_file_path_container: list[str | None],
    run_id: str,
    request_id: str,
    query: str,
    chunk_id: str,
    model_name: str,
    created: int,
) -> tuple[str, AsyncGenerator[str, None]]:
    """处理 DeepSearch 任务完成后的逻辑（成功或失败）。
    
    Returns:
        Tuple of (effective_run_id, error_events_generator)
    """
    effective_run_id = run_id or request_id
    result = None
    
    try:
        result = await deepsearch_task
        deepsearch_result_container[0] = result
        
        # 从 result 中提取 run_id（如果还没有）
        if not run_id and isinstance(result, dict):
            state = result.get("state") or {}
            if isinstance(state, dict):
                run_id = str(state.get("run_id", request_id))
        
        effective_run_id = run_id or request_id
        logger.info("DeepSearch service returned (request_id=%s, run_id=%s)", request_id, effective_run_id)
        
        # 保存所有 trace events 到文件
        all_trace_events = trace_emitter.get_all_events()
        if all_trace_events:
            saved_path = _save_trace_events_to_file(
                all_trace_events,
                request_id,
                effective_run_id,
                query,
                result
            )
            if saved_path:
                trace_file_path_container[0] = saved_path
                logger.info("Trace events saved to: %s", saved_path)
        
        # 返回空的错误事件生成器（成功时没有错误）
        async def empty_error_gen():
            return
            yield
        return effective_run_id, empty_error_gen()
        
    except Exception as e:
        # DeepSearch 任务失败，记录错误并设置结果为 None
        logger.error("DeepSearch task failed (request_id=%s): %s", request_id, e, exc_info=True)
        deepsearch_result_container[0] = None
        
        # 发送错误事件给前端
        error_progress = {
            "stage": "deepsearch",
            "deepsearch_stage": "failed",
            "status": "failed",
            "message": f"DeepSearch 执行失败: {str(e)}",
            "error": str(e),
        }
        
        async def error_events_gen():
            async for event in _yield_deepsearch_progress(
                error_progress,
                chunk_id,
                model_name,
                created,
                request_id
            ):
                yield event
        
        effective_run_id = run_id or request_id
        logger.warning("DeepSearch failed, will fall back to RAG system (request_id=%s)", request_id)
        
        # 即使失败，也保存已收集的 trace events
        all_trace_events = trace_emitter.get_all_events()
        if all_trace_events:
            saved_path = _save_trace_events_to_file(
                all_trace_events,
                request_id,
                effective_run_id,
                query,
                None  # 没有结果
            )
            if saved_path:
                trace_file_path_container[0] = saved_path
                logger.info("Trace events saved (failed run) to: %s", saved_path)
        
        return effective_run_id, error_events_gen()


async def _consume_remaining_events(
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None],
    trace_queue: asyncio.Queue[TraceEvent | None],
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    effective_run_id: str,
) -> AsyncGenerator[str, None]:
    """消费剩余的 progress 和 trace 事件。"""
    deadline = time.time() + 1.0
    while time.time() < deadline:
        try:
            # 处理剩余的 progress 事件
            try:
                progress_payload = await asyncio.wait_for(
                    deepsearch_progress_queue.get(),
                    timeout=0.05
                )
                if progress_payload is not None:
                    async for event in _yield_deepsearch_progress(
                        progress_payload,
                        chunk_id,
                        model_name,
                        created,
                        request_id
                    ):
                        yield event
            except asyncio.TimeoutError:
                pass
            
            # 处理剩余的 trace 事件
            try:
                trace_event = await asyncio.wait_for(
                    trace_queue.get(),
                    timeout=0.05
                )
                if trace_event is not None:
                    async for event in _yield_trace_event(
                        trace_event,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        effective_run_id
                    ):
                        yield event
            except asyncio.TimeoutError:
                pass
            
            # 如果两个队列都为空，退出
            if deepsearch_progress_queue.empty() and trace_queue.empty():
                break
                
        except Exception as e:
            logger.error("Error consuming remaining events: %s", e, exc_info=True)
            break


async def _process_event_loop(
    deepsearch_task: asyncio.Task,
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None],
    trace_queue: asyncio.Queue[TraceEvent | None],
    run_id: list[str],  # 使用列表以便修改
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """处理事件循环，等待并处理 progress 和 trace 事件。"""
    deepsearch_progress_task = None
    trace_task = None
    
    while not deepsearch_task.done():
        # 处理 progress 事件
        if deepsearch_progress_task is None or deepsearch_progress_task.done():
            deepsearch_progress_task = asyncio.create_task(
                deepsearch_progress_queue.get()
            )
        
        # 处理 trace 事件
        if trace_task is None or trace_task.done():
            trace_task = asyncio.create_task(
                trace_queue.get()
            )
        
        # 等待任一任务完成
        done, pending = await asyncio.wait(
            [deepsearch_task, deepsearch_progress_task, trace_task],
            timeout=0.1,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # 处理 progress 事件
        if deepsearch_progress_task in done:
            try:
                progress_payload = await deepsearch_progress_task
                deepsearch_progress_task = None
                if progress_payload is not None:
                    # 从 progress 中提取 run_id（如果存在）
                    if not run_id[0] and "run_id" in progress_payload:
                        run_id[0] = str(progress_payload.get("run_id", ""))
                    
                    async for event in _yield_deepsearch_progress(
                        progress_payload,
                        chunk_id,
                        model_name,
                        created,
                        request_id
                    ):
                        yield event
            except Exception as e:
                logger.error("Error processing DeepSearch progress: %s", e, exc_info=True)
                deepsearch_progress_task = None
        
        # 处理 trace 事件
        if trace_task in done:
            try:
                trace_event = await trace_task
                trace_task = None
                if trace_event is not None:
                    # 使用 request_id 作为 run_id（如果没有从 progress 中获取到）
                    effective_run_id = run_id[0] or request_id
                    async for event in _yield_trace_event(
                        trace_event,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        effective_run_id
                    ):
                        yield event
            except Exception as e:
                logger.error("Error processing DeepSearch trace: %s", e, exc_info=True)
                trace_task = None
        
        # 如果 DeepSearch 任务完成，退出循环
        if deepsearch_task in done:
            if deepsearch_progress_task and not deepsearch_progress_task.done():
                deepsearch_progress_task.cancel()
                try:
                    await deepsearch_progress_task
                except asyncio.CancelledError:
                    pass
            if trace_task and not trace_task.done():
                trace_task.cancel()
                try:
                    await trace_task
                except asyncio.CancelledError:
                    pass
            break


async def _create_empty_generator() -> AsyncGenerator[str, None]:
    """Create an empty async generator for error paths.

    `deepsearch_processor.process_deepsearch_with_events` always `async for`-iterates
    the returned generator. Returning a sync generator here breaks SSE streaming.
    """
    if False:  # pragma: no cover
        yield ""
    return


def _handle_deepsearch_error(
    error: Exception,
    error_type: type,
    trace_token: Any | None = None
) -> tuple[list[dict[str, Any] | None], list[str | None], AsyncGenerator[str, None]]:
    """处理 DeepSearch 初始化错误。
    
    Returns:
        Tuple of (deepsearch_result_container, trace_file_path_container, empty_generator)
    """
    if error_type == KeyError:
        logger.warning("DeepSearch service not available: %s", error)
    else:
        logger.error("DeepSearch failed: %s", error, exc_info=True)
    
    # 确保重置 trace emitter
    if trace_token is not None:
        try:
            reset_trace_emitter(trace_token)
        except Exception:
            pass
    
    return [None], [None], _create_empty_generator()


def _initialize_deepsearch_processing(
    query: str,
    effective_owner: str,
    request_id: str,
    loop: asyncio.AbstractEventLoop,
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None],
    trace_queue: asyncio.Queue[TraceEvent | None],
    emit_deepsearch_progress: callable,
) -> tuple[Any, InMemoryTraceEmitter, Any, asyncio.Task]:
    """初始化 DeepSearch 处理环境。
    
    Returns:
        Tuple of (deepsearch_service, trace_emitter, trace_token, deepsearch_task)
    """
    registrator = Register()
    deepsearch_service = registrator.get_object("deepsearch_service")
    logger.info("DeepSearch service found, running...")
    
    # 创建 InMemoryTraceEmitter 并设置 trace emitter
    trace_emitter = InMemoryTraceEmitter(trace_queue)
    trace_token = set_trace_emitter(trace_emitter)
    
    stage_listener = _create_stage_listener(
        request_id,
        loop,
        deepsearch_progress_queue,
        emit_deepsearch_progress
    )
    
    # 启动 DeepSearch 任务
    deepsearch_task = asyncio.create_task(
        deepsearch_service.run(
            question=query,
            owner_id=str(effective_owner),
            stage_listener=stage_listener,
        )
    )
    
    return deepsearch_service, trace_emitter, trace_token, deepsearch_task


async def _create_progress_generator(
    deepsearch_task: asyncio.Task,
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None],
    trace_queue: asyncio.Queue[TraceEvent | None],
    trace_emitter: InMemoryTraceEmitter,
    deepsearch_result_container: list[dict[str, Any] | None],
    trace_file_path_container: list[str | None],
    run_id_list: list[str],
    trace_token: Any,
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    query: str,
) -> AsyncGenerator[str, None]:
    """创建 progress 事件生成器。"""
    # 处理事件循环
    async for event in _process_event_loop(
        deepsearch_task,
        deepsearch_progress_queue,
        trace_queue,
        run_id_list,
        chunk_id,
        model_name,
        created,
        request_id,
    ):
        yield event
    
    # 处理任务完成
    effective_run_id, error_events_gen = await _handle_deepsearch_task_completion(
        deepsearch_task,
        trace_emitter,
        deepsearch_result_container,
        trace_file_path_container,
        run_id_list[0],
        request_id,
        query,
        chunk_id,
        model_name,
        created,
    )
    
    # 发送错误事件（如果有）
    async for event in error_events_gen:
        yield event
    
    # 消费剩余事件
    async for event in _consume_remaining_events(
        deepsearch_progress_queue,
        trace_queue,
        chunk_id,
        model_name,
        created,
        request_id,
        effective_run_id,
    ):
        yield event
    
    # 重置 trace emitter
    try:
        reset_trace_emitter(trace_token)
    except Exception as e:
        logger.warning("Error resetting trace emitter: %s", e)


async def process_deepsearch(
    query: str,
    effective_owner: str,
    request_id: str,
    chunk_id: str,
    model_name: str,
    created: int,
    loop: asyncio.AbstractEventLoop
) -> tuple[list[dict[str, Any] | None], list[str | None], AsyncGenerator[str, None]]:
    """Process DeepSearch and yield progress events.
    
    Returns:
        Tuple of (deepsearch_result_container, trace_file_path_container, progress_generator)
        - deepsearch_result_container: list with one element that will be set to the result
        - trace_file_path_container: list with one element that will be set to the trace file path
        - progress_generator: async generator that yields SSE events
    """
    # 初始化队列和容器
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    trace_queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
    progress_seq = 0
    deepsearch_result_container: list[dict[str, Any] | None] = [None]
    trace_file_path_container: list[str | None] = [None]
    run_id_list: list[str] = [""]
    
    # 创建 emit_deepsearch_progress 函数
    async def emit_deepsearch_progress(payload: dict[str, Any]) -> None:
        nonlocal progress_seq
        progress_seq += 1
        envelope = dict(payload or {})
        envelope.setdefault("v", 1)
        envelope.setdefault("type", "progress")
        envelope.setdefault("ts_ms", int(time.time() * 1000))
        envelope.setdefault("request_id", request_id)
        envelope.setdefault("seq", progress_seq)
        await deepsearch_progress_queue.put(envelope)
    
    # 初始化 DeepSearch 处理环境
    trace_token = None
    try:
        _, trace_emitter, trace_token, deepsearch_task = _initialize_deepsearch_processing(
            query,
            effective_owner,
            request_id,
            loop,
            deepsearch_progress_queue,
            trace_queue,
            emit_deepsearch_progress,
        )
        
        # 创建并返回 progress generator
        progress_gen = _create_progress_generator(
            deepsearch_task,
            deepsearch_progress_queue,
            trace_queue,
            trace_emitter,
            deepsearch_result_container,
            trace_file_path_container,
            run_id_list,
            trace_token,
            chunk_id,
            model_name,
            created,
            request_id,
            query,
        )
        
        return deepsearch_result_container, trace_file_path_container, progress_gen
        
    except KeyError as e:
        return _handle_deepsearch_error(e, KeyError, trace_token)
    except Exception as e:
        return _handle_deepsearch_error(e, type(e), trace_token)
