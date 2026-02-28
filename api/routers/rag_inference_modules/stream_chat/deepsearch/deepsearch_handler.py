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
from core.deepsearch.utils.llm_envelope import try_parse_llm_envelope
from core.constants.io_namespaces import DEEPSEARCH_TRACES_NAMESPACE
from api.routers.deepsearch_weaver_render import render_trace_payload
from framework.virtual_paths import IO_PATH_PREFIX, io_key

logger = logging.getLogger(__name__)


def _try_parse_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_envelope_thinking(raw: str) -> str | None:
    payload = try_parse_llm_envelope(raw)
    if not isinstance(payload, dict):
        return None
    thinking = str(payload.get("thinking") or "").strip()
    return thinking or None


def _extract_reasoning_lines(raw: str) -> str | None:
    parsed = _try_parse_json(raw)
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    lines = []
    for line in str(raw or "").splitlines():
        if "reasoning=" in line:
            lines.append(line.split("reasoning=", 1)[1].strip())
    if lines:
        return " / ".join([line for line in lines if line])
    text = str(raw or "").strip()
    return text or None


def _extract_tool_response_thinking(payload: dict[str, Any]) -> str | None:
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        summary = result.get("summary")
        if isinstance(summary, str) and summary.strip():
            return _extract_envelope_thinking(summary) or _extract_reasoning_lines(summary)
        think_notes = result.get("think_notes")
        if isinstance(think_notes, list):
            for note in reversed(think_notes):
                if not isinstance(note, dict):
                    continue
                reasoning = note.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
    return None


def _extract_tool_call_hint(payload: dict[str, Any]) -> str | None:
    extra = payload.get("extra") if isinstance(payload, dict) else None
    if isinstance(extra, dict):
        for key in ("rationale", "purpose", "query", "focus_query", "question"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    tool_name = payload.get("tool_name") if isinstance(payload, dict) else None
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()
    return None


def _format_tool_call_label(payload: dict[str, Any]) -> str | None:
    """Format a tool call as tool_name(key=val, ...) compact label for frontend display."""
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip()
    if not tool_name:
        return None
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    tool_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
    args = {**extra, **tool_args}
    parts = []
    for key in ("query", "focus_query", "question", "file_id", "pages"):
        val = args.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            display = (val[:40] + "...") if len(val) > 40 else val
            parts.append(f'{key}="{display}"')
        elif isinstance(val, list):
            raw = str(val)
            display = (raw[:40] + "...") if len(raw) > 40 else raw
            parts.append(f"{key}={display}")
        if len(parts) >= 2:
            break
    if parts:
        return f"{tool_name}({', '.join(parts)})"
    return tool_name


def _first_nonempty_line(text: str | None) -> str | None:
    for line in str(text or "").splitlines():
        token = line.strip()
        if token:
            return token
    return None


def _build_trace_message(trace_event: TraceEvent, rendered_content: str | None = None) -> str | None:
    content = str(getattr(trace_event, "content", "") or "").strip()
    if trace_event.tag == "think" and content:
        return _extract_envelope_thinking(content) or _extract_reasoning_lines(content)

    parsed = _try_parse_json(content)
    if trace_event.tag == "tool_response" and isinstance(parsed, dict):
        message = _extract_tool_response_thinking(parsed)
        if message:
            return message
    if trace_event.tag == "tool_call" and isinstance(parsed, dict):
        label = _format_tool_call_label(parsed)
        if label:
            return label
        message = _extract_tool_call_hint(parsed)
        if message:
            return message

    if content:
        message = _extract_envelope_thinking(content)
        if message:
            return message

    return _first_nonempty_line(rendered_content) or _first_nonempty_line(content)

def _collect_plan_step_texts(plan_steps: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(plan_steps, list):
        return texts
    for step in plan_steps:
        if not isinstance(step, dict):
            continue
        for key in ("description", "text", "title"):
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                break
    return texts


def _extract_latest_think_note(state: Any) -> str | None:
    trace = getattr(state, "reasoning_trace", None)
    trace = trace if isinstance(trace, dict) else {}
    notes = trace.get("think_notes") if isinstance(trace, dict) else None
    if not isinstance(notes, list):
        return None
    for note in reversed(notes):
        if not isinstance(note, dict):
            continue
        reasoning = note.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    return None


def _extract_latest_reasoning_step(state: Any) -> str | None:
    trace = getattr(state, "reasoning_trace", None)
    trace = trace if isinstance(trace, dict) else {}
    steps = trace.get("reasoning_steps") if isinstance(trace, dict) else None
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        candidate = step.get("output_summary") or step.get("description")
        if isinstance(candidate, str) and candidate.strip():
            return _extract_envelope_thinking(candidate) or _extract_reasoning_lines(candidate)
    return None


def _extract_tool_result_thinking(state: Any) -> str | None:
    trace = getattr(state, "reasoning_trace", None)
    trace = trace if isinstance(trace, dict) else {}
    tool_results = trace.get("tool_results") if isinstance(trace, dict) else None
    if not isinstance(tool_results, list):
        return None
    for entry in reversed(tool_results):
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        summary = result.get("summary")
        if isinstance(summary, str) and summary.strip():
            return _extract_envelope_thinking(summary) or _extract_reasoning_lines(summary)
        think_notes = result.get("think_notes")
        if isinstance(think_notes, list):
            for note in reversed(think_notes):
                if not isinstance(note, dict):
                    continue
                reasoning = note.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
    return None


def _extract_report_summary(state: Any) -> str | None:
    report = getattr(state, "report_payload", None)
    if not isinstance(report, dict):
        return None
    answer = report.get("answer")
    if isinstance(answer, str) and answer.strip():
        return _first_nonempty_line(answer)
    return None


def _extract_error_summary(state: Any) -> str | None:
    errors = getattr(state, "errors", None)
    if not isinstance(errors, list):
        return None
    for entry in reversed(errors):
        if not isinstance(entry, dict):
            continue
        for key in ("reason", "message"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _build_progress_message(stage: str, metadata: dict, state: Any) -> str | None:
    # For reported/done stages, prefer the report summary so that the
    # progress message reflects the actual answer rather than the last
    # think-note reasoning.  Previously the think-note candidates always
    # won (they're never empty), meaning the progress "done" event
    # carried raw thinking text — which frontends could mistake for the
    # final answer.
    if stage in {"reported", "done"}:
        report_summary = _extract_report_summary(state)
        if report_summary:
            return report_summary

    for candidate in (
        _extract_latest_think_note(state),
        _extract_tool_result_thinking(state),
        _extract_latest_reasoning_step(state),
    ):
        if candidate:
            return candidate

    if stage == "planned":
        plan_texts = _collect_plan_step_texts(getattr(state, "plan_steps", None))
        if plan_texts:
            return " / ".join(plan_texts[:2])

    if stage == "failed":
        error_summary = _extract_error_summary(state)
        if error_summary:
            return error_summary

    if isinstance(metadata, dict):
        for key in ("plan_id", "step_id", "stage"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None



def _save_trace_events_to_file(
    trace_events: List[TraceEvent],
    request_id: str,
    run_id: str,
    query: str,
    deepsearch_result: dict[str, Any] | None
) -> str | None:
    """Save trace events to a JSON file.

    Returns:
        An IO reference (preferred) or filesystem path (legacy), or None if saving failed.
    """
    try:
        def _require_io_manager():
            from app_registration import registrator

            io_manager = registrator.get_object("io_manager")
            if io_manager is None:
                raise RuntimeError("io_manager is required for trace storage")
            return io_manager

        # 生成文件名：使用 run_id 或 request_id，加上时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = run_id or request_id
        filename = f"trace_{file_id}_{timestamp}.json"

        rendered_result = deepsearch_result
        try:
            if isinstance(deepsearch_result, dict):
                report = deepsearch_result.get("report")
                if isinstance(report, dict):
                    raw_answer = report.get("answer")
                    if isinstance(raw_answer, str) and raw_answer.strip():
                        # Keep DeepSearch citations consistent with the rag_inference chain:
                        # answer already uses <sup>k</sup>; we just normalize punctuation and emit sources mapping.
                        from core.deepsearch.report.sup_citations import (
                            build_source_entries,
                            normalize_sup_punctuation,
                        )

                        structured = report.get("structured_report")
                        source_key_map = structured.get("source_key_map") if isinstance(structured, dict) else None
                        evidences = report.get("evidences") if isinstance(report.get("evidences"), list) else None

                        normalized = normalize_sup_punctuation(raw_answer)
                        rendered_result = json.loads(json.dumps(deepsearch_result, ensure_ascii=False, default=str))
                        rendered_report = rendered_result.get("report") if isinstance(rendered_result, dict) else None
                        if isinstance(rendered_report, dict):
                            rendered_report.setdefault("answer_raw", raw_answer)
                            rendered_report["answer"] = normalized
                            if isinstance(source_key_map, dict) and source_key_map:
                                pairs = []
                                for key, ev_id in source_key_map.items():
                                    try:
                                        num = int(str(key).strip())
                                    except ValueError:
                                        continue
                                    ev_id = str(ev_id).strip()
                                    if not ev_id:
                                        continue
                                    pairs.append((num, ev_id))
                                if pairs:
                                    pairs.sort(key=lambda item: item[0])
                                    ordered_ids = [ev_id for _, ev_id in pairs]
                                    id_to_num = {ev_id: num for num, ev_id in pairs}
                                    evidence_lookup = {}
                                    for item in evidences or []:
                                        if not isinstance(item, dict):
                                            continue
                                        chunk_id = str(item.get("chunk_id") or "").strip()
                                        if not chunk_id:
                                            continue
                                        evidence_lookup[chunk_id] = item
                                    sources = build_source_entries(
                                        ordered_ids=ordered_ids,
                                        evidence_lookup=evidence_lookup,
                                        id_to_num=id_to_num,
                                    )
                                    if sources:
                                        rendered_report["sources"] = sources
                                    rendered_report["citation_key_map"] = id_to_num
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

        trace_root = str(os.getenv("DEEPSEARCH_TRACE_STORAGE_PATH", f"{IO_PATH_PREFIX}{DEEPSEARCH_TRACES_NAMESPACE}") or "").strip()
        if not trace_root:
            raise ValueError("DEEPSEARCH_TRACE_STORAGE_PATH must not be empty")

        # Preferred: io:// storage via IOManager (works across LocalDB/MinIO).
        if trace_root.startswith(IO_PATH_PREFIX):
            io_manager = _require_io_manager()
            root_key = io_key(trace_root)
            if not root_key:
                raise ValueError("DEEPSEARCH_TRACE_STORAGE_PATH must not be empty")
            namespace, prefix = (root_key.split("/", 1) + [""])[:2]
            namespace = namespace or DEEPSEARCH_TRACES_NAMESPACE
            key = f"{prefix}/{file_id}/{filename}".lstrip("/") if prefix else f"{file_id}/{filename}"

            put = io_manager.put_json(
                namespace=namespace,
                key=key,
                payload=trace_data,
            )

            logger.info(
                "Saved %d trace events to %s (request_id=%s, run_id=%s)",
                len(trace_events),
                put.ref,
                request_id,
                run_id or request_id,
            )
            return str(put.ref)

        # Legacy: local filesystem path support for unit tests / dev scripts.
        out_dir = Path(trace_root).expanduser().resolve() / str(file_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / filename).resolve()
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(trace_data, handle, ensure_ascii=False, indent=2, default=str)
        logger.info(
            "Saved %d trace events to %s (request_id=%s, run_id=%s)",
            len(trace_events),
            out_path,
            request_id,
            run_id or request_id,
        )
        return str(out_path)
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
        token = str(trace_file_path or "").strip()
        if not token:
            return None

        if token.startswith("io://"):
            from app_registration import registrator

            io_manager = registrator.get_object("io_manager")
            if io_manager is None:
                raise RuntimeError("io_manager is required for trace retrieval")
            payload = io_manager.get_json(token)
            if payload is None:
                logger.warning("Trace ref not found: %s", token)
            return payload

        path = Path(token)
        if not path.exists() or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
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
        if "step_count" in metadata:
            progress_info["plan_steps_count"] = metadata.get("step_count")
        if hasattr(state, "plan_steps") and state.plan_steps:
            progress_info["plan_steps"] = state.plan_steps
        if hasattr(state, "plan_metadata") and state.plan_metadata:
            progress_info["plan_metadata"] = state.plan_metadata
    elif stage == "reasoned":
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
    elif stage == "reported":
        if hasattr(state, "report_payload") and state.report_payload:
            progress_info["report_payload"] = state.report_payload
    elif stage == "done":
        progress_info["status"] = "completed"
        # Use "completed" instead of "done" to avoid collision with the
        # canonical done signal emitted by deepsearch_processor
        # (_yield_deepsearch_done_signal).  The processor's signal is the
        # one the frontend should act on to collapse thinking and start
        # rendering content deltas.  If we also emit deepsearch_stage="done"
        # here, the frontend may prematurely use this event's `message`
        # (which contains the last think-note reasoning, not the report)
        # as the answer.
        progress_info["deepsearch_stage"] = "completed"
        if hasattr(state, "run_id"):
            progress_info["run_id"] = state.run_id
    elif stage == "failed":
        progress_info["status"] = "failed"
        if hasattr(state, "errors") and state.errors:
            progress_info["errors"] = state.errors
    elif stage == "created":
        if hasattr(state, "run_id"):
            progress_info["run_id"] = state.run_id

    message = _build_progress_message(stage, metadata, state)
    progress_info["message"] = message or str(stage or "deepsearch")

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
    request_id: str,
    index: int = 0,
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch progress as SSE event."""
    tool_calls = [{
        "index": index,
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
    run_id: str,
    index: int = 0,
) -> AsyncGenerator[str, None]:
    """Yield DeepSearch trace event as SSE event (rag_arc_trace format)."""
    # tool_response events contain raw retrieval results - not shown to user
    if trace_event.tag == "tool_response":
        return

    # 渲染 trace event
    rendered_tag, rendered_content = render_trace_payload(
        trace_tag=trace_event.tag,
        content=trace_event.content,
        run_id=run_id
    )

    # 生成人类可读的 message（优先使用动态推理内容）
    message = _build_trace_message(trace_event, rendered_content=rendered_content) or str(rendered_tag or trace_event.tag or "trace")

    # 作为 tool_call 发送
    tool_calls = [{
        "index": index,
        "id": f"call_deepsearch_trace_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": "rag_arc_trace",
            "arguments": json.dumps(
                {
                    "message": message,
                    "content": rendered_content,
                    "tag": rendered_tag,
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


async def _yield_deepsearch_done_signal(
    chunk_id: str,
    model_name: str,
    created: int,
    request_id: str,
    index: int = 0,
) -> AsyncGenerator[str, None]:
    """Yield rag_arc_progress done signal to trigger frontend collapse of thinking steps."""
    tool_calls = [{
        "index": index,
        "id": f"call_deepsearch_done_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": "rag_arc_progress",
            "arguments": json.dumps({"deepsearch_stage": "done"}, ensure_ascii=False),
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
    trace_call_index: list[int],
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
            idx = trace_call_index[0]
            trace_call_index[0] += 1
            async for event in _yield_deepsearch_progress(
                error_progress,
                chunk_id,
                model_name,
                created,
                request_id,
                index=idx,
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
    trace_call_index: list[int],
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
                    idx = trace_call_index[0]
                    trace_call_index[0] += 1
                    async for event in _yield_deepsearch_progress(
                        progress_payload,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        index=idx,
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
                    idx = trace_call_index[0]
                    trace_call_index[0] += 1
                    async for event in _yield_trace_event(
                        trace_event,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        effective_run_id,
                        index=idx,
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
    trace_call_index: list[int],
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
                    idx = trace_call_index[0]
                    trace_call_index[0] += 1
                    async for event in _yield_deepsearch_progress(
                        progress_payload,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        index=idx,
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
                    idx = trace_call_index[0]
                    trace_call_index[0] += 1
                    async for event in _yield_trace_event(
                        trace_event,
                        chunk_id,
                        model_name,
                        created,
                        request_id,
                        effective_run_id,
                        index=idx,
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
) -> tuple[list[dict[str, Any] | None], list[str | None], AsyncGenerator[str, None], list[int]]:
    """处理 DeepSearch 初始化错误。

    Returns:
        Tuple of (deepsearch_result_container, trace_file_path_container, empty_generator, trace_call_index)
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
    
    return [None], [None], _create_empty_generator(), [0]


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
    trace_call_index: list[int],
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
        trace_call_index,
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
        trace_call_index,
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
        trace_call_index,
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
) -> tuple[list[dict[str, Any] | None], list[str | None], AsyncGenerator[str, None], list[int]]:
    """Process DeepSearch and yield progress events.

    Returns:
        Tuple of (deepsearch_result_container, trace_file_path_container, progress_generator, trace_call_index)
        - deepsearch_result_container: list with one element that will be set to the result
        - trace_file_path_container: list with one element that will be set to the trace file path
        - progress_generator: async generator that yields SSE events
        - trace_call_index: shared counter tracking the next available tool_call index
    """
    # 初始化队列和容器
    deepsearch_progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    trace_queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
    progress_seq = 0
    deepsearch_result_container: list[dict[str, Any] | None] = [None]
    trace_file_path_container: list[str | None] = [None]
    run_id_list: list[str] = [""]
    trace_call_index: list[int] = [0]

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
            trace_call_index,
        )

        return deepsearch_result_container, trace_file_path_container, progress_gen, trace_call_index
        
    except KeyError as e:
        return _handle_deepsearch_error(e, KeyError, trace_token)
    except Exception as e:
        return _handle_deepsearch_error(e, type(e), trace_token)
