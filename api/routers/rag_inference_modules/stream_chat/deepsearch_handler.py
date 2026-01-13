"""DeepSearch handling for stream chat."""
import json
import uuid
import asyncio
import time
import logging
from typing import Any, AsyncGenerator
from framework.register import Register
from api.sse import (
    delta_envelope,
    openai_chat_completion_chunk,
    sse_json_wrapped,
)

logger = logging.getLogger(__name__)


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


async def process_deepsearch(
    query: str,
    effective_owner: str,
    request_id: str,
    chunk_id: str,
    model_name: str,
    created: int,
    loop: asyncio.AbstractEventLoop
) -> tuple[str, AsyncGenerator[str, None]]:
    """Process DeepSearch and yield progress events.
    
    Returns:
        Tuple of (enhanced_query, progress_generator)
    """
    enhanced_query = query
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    progress_seq = 0
    
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
    
    try:
        registrator = Register()
        deepsearch_service = registrator.get_object("deepsearch_service")
        logger.info("DeepSearch service found, running...")
        
        stage_listener = _create_stage_listener(
            request_id,
            loop,
            deepsearch_progress_queue,
            emit_deepsearch_progress
        )
        
        deepsearch_task = asyncio.create_task(
            deepsearch_service.run(
                question=query,
                owner_id=str(effective_owner),
                stage_listener=stage_listener,
            )
        )
        
        # Process progress events during execution
        async def progress_generator() -> AsyncGenerator[str, None]:
            deepsearch_progress_task = None
            while not deepsearch_task.done():
                if deepsearch_progress_task is None or deepsearch_progress_task.done():
                    deepsearch_progress_task = asyncio.create_task(
                        deepsearch_progress_queue.get()
                    )
                
                done, pending = await asyncio.wait(
                    [deepsearch_task, deepsearch_progress_task],
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if deepsearch_progress_task in done:
                    try:
                        progress_payload = await deepsearch_progress_task
                        deepsearch_progress_task = None
                        if progress_payload is not None:
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
                
                if deepsearch_task in done:
                    if deepsearch_progress_task and not deepsearch_progress_task.done():
                        deepsearch_progress_task.cancel()
                        try:
                            await deepsearch_progress_task
                        except asyncio.CancelledError:
                            pass
                    break
            
            # Get result and consume remaining progress
            deepsearch_result = await deepsearch_task
            logger.info("DeepSearch service returned (request_id=%s)", request_id)
            
            deadline = time.time() + 1.0
            while time.time() < deadline:
                try:
                    progress_payload = await asyncio.wait_for(
                        deepsearch_progress_queue.get(),
                        timeout=0.1
                    )
                    if progress_payload is None:
                        break
                    async for event in _yield_deepsearch_progress(
                        progress_payload,
                        chunk_id,
                        model_name,
                        created,
                        request_id
                    ):
                        yield event
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    break
                except Exception as e:
                    logger.error("Error consuming remaining progress: %s", e, exc_info=True)
                    break
            
            deepsearch_answer = (
                deepsearch_result.get("answer") or
                deepsearch_result.get("report") or
                ""
            )
            if deepsearch_answer:
                nonlocal enhanced_query
                enhanced_query = f"{query}\n\n[DeepSearch 增强上下文]\n{deepsearch_answer}"
                logger.info("DeepSearch completed, enhanced query length: %d", len(enhanced_query))
        
        return enhanced_query, progress_generator()
    except KeyError as e:
        logger.warning("DeepSearch service not available: %s", e)
        async def empty_gen():
            return
            yield
        return enhanced_query, empty_gen()
    except Exception as e:
        logger.error("DeepSearch failed: %s", e, exc_info=True)
        async def empty_gen():
            return
            yield
        return enhanced_query, empty_gen()
