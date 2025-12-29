import uuid
import asyncio
import json
import os
import time
from typing import Annotated, Any, Dict, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User
from framework.register import Register
from core.utils.owner_guard import is_admin_owner
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from application.rag_inference.module import RAGInference
from application.deepsearch.trace_emitter import make_inprocess_trace_emitter
from api.deepsearch.tasks import TASKS, format_sse, new_run_id
from api.sse import (
    delta_envelope,
    new_chatcmpl_id,
    now_epoch_seconds,
    openai_chat_completion_chunk,
    sse_text,
    sse_done,
    sse_json,
)
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from core.deepsearch.trace import emit_trace, reset_trace_emitter, set_trace_emitter, with_trace_protocol
from core.deepsearch.tooling.all_tools import render_all_tools_block

router = APIRouter(prefix="/deepsearch", tags=["deepsearch"])
registrator = Register()


def _get_task_queue() -> RedisTaskQueue:
    # Avoid import-time singletons so tests/runtime can update env settings.
    return RedisTaskQueue.from_env()

def _use_celery() -> bool:
    return os.getenv("TASK_QUEUE_MODE", "inprocess").lower() == "celery"

def _assert_task_owner(task_run: Dict[str, Any], *, user_id: uuid.UUID) -> None:
    if is_admin_owner(user_id):
        return
    raw = task_run.get("owner_id")
    if not raw:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")
    try:
        owner_uuid = uuid.UUID(str(raw))
    except Exception:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id") from None
    if owner_uuid != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")


class DeepSearchRequest(BaseModel):
    question: str = Field(..., description="User question, must be a non-empty string")
    owner_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional owner override, limited to admin users",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata merged into DeepSearch state",
    )
    include_evidence: bool = Field(
        default=False,
        description="When true, attach chunk/graph summaries to the response.",
    )


def _get_deepsearch_service():
    try:
        return registrator.get_object("deepsearch_service")
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DeepSearch service is not initialized. Check DEEPSEARCH_SERVICE_CONFIG_PATH.",
        ) from exc


def _get_graph_store() -> Any | None:
    try:
        rag = registrator.get_object("rag_inference")
    except KeyError:
        return None
    if isinstance(rag, RAGInference):
        try:
            return rag.get_graph_store()
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _stage_progress(stage: str) -> Dict[str, Any]:
    order = [
        "created",
        "planned",
        "reasoned",
        "gap_evaluated",
        "external_invoked",
        "reported",
        "failed",
    ]
    normalized = (stage or "").strip().lower() or "unknown"
    try:
        idx = order.index(normalized)
        pct = int((idx / max(1, len(order) - 1)) * 100)
    except ValueError:
        idx = 0
        pct = 0
    return {"stage": normalized, "step_index": idx, "step_total": len(order), "percent": pct}

async def _stream_events(run_id: str, *, last_event_id: int = -1):
    info = await TASKS.get(run_id)
    if not info:
        yield format_sse(event="error", data=with_trace_protocol({"run_id": run_id, "message": "run_id not found"}, run_id=run_id))
        yield sse_done()
        return

    cursor = max(-1, last_event_id)
    while True:
        await asyncio.sleep(0)
        async with info.cond:
            while cursor + 1 >= len(info.events) and not info.done:
                try:
                    await asyncio.wait_for(info.cond.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield format_sse(event="heartbeat", data=with_trace_protocol({"run_id": run_id}, run_id=run_id))
            pending = info.events[cursor + 1 :]

        for event in pending:
            cursor = int(event.get("id", cursor + 1))
            payload = event.get("payload") or {}
            yield format_sse(
                event=event.get("type") or "message",
                event_id=cursor,
                data={"run_id": run_id, **payload},
            )

        if info.done and cursor + 1 >= len(info.events):
            done_payload = {
                "run_id": run_id,
                "done": True,
                "error": info.error,
                "progress": info.last_progress.get("progress") if isinstance(info.last_progress, dict) else None,
            }
            yield format_sse(event="done", data=with_trace_protocol(done_payload, run_id=run_id), event_id=cursor + 1)
            yield sse_done()
            return


_WEAVER_ALLOWED_TAGS = {
    "all_tools",
    "think",
    "write_outline",
    "tool_call",
    "tool_response",
    "write",
    "progress",
    "terminate",
}

def _truncate_text(value: str, *, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _try_parse_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _render_tool_call(payload: Dict[str, Any]) -> str:
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip() or "unknown"
    call_id = str(payload.get("call_id") or "").strip()
    plan_step = str(payload.get("plan_step") or "").strip()
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    query = ""
    for key in ("query", "focus_query", "question", "prompt"):
        if isinstance(extra.get(key), str) and extra[key].strip():
            query = extra[key].strip()
            break
    channel = ""
    if isinstance(extra.get("channel"), str):
        channel = extra.get("channel") or ""

    lines: list[str] = []
    header = f"tool={tool_name}"
    if call_id:
        header += f" call_id={call_id}"
    if plan_step:
        header += f" plan_step={plan_step}"
    if channel:
        header += f" channel={channel}"
    lines.append(header)
    descriptor = payload.get("descriptor") if isinstance(payload.get("descriptor"), dict) else {}
    if descriptor:
        profile = descriptor.get("profile")
        determinism = descriptor.get("determinism")
        about = str(descriptor.get("description") or "").strip()
        if profile or determinism:
            lines.append(
                "tool_meta="
                + json.dumps(
                    {"profile": profile, "determinism": determinism},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
        if about:
            lines.append("about=" + _truncate_text(about, limit=200))
    if query:
        lines.append("query=" + _truncate_text(query, limit=280))
    routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
    if routing:
        lines.append(
            "routing="
            + json.dumps(
                {
                    "can_route_remote": routing.get("can_route_remote"),
                    "prefer_remote": routing.get("prefer_remote"),
                    "has_local": routing.get("has_local"),
                    "default_mcp_server": routing.get("default_mcp_server"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    coverage = payload.get("coverage_metrics") if isinstance(payload.get("coverage_metrics"), dict) else {}
    if coverage:
        lines.append(
            "coverage="
            + json.dumps(
                {
                    "evidence_count": coverage.get("evidence_count"),
                    "coverage_ratio": coverage.get("coverage_ratio"),
                    "coverage_score": coverage.get("coverage_score"),
                    "completed_steps": coverage.get("completed_steps"),
                    "total_steps": coverage.get("total_steps"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    return "\n".join([ln for ln in lines if ln is not None and str(ln).strip()])


def _evidence_preview(evidence: Dict[str, Any]) -> str:
    chunk_id = str(evidence.get("chunk_id") or evidence.get("evidence_id") or "").strip()
    source = str(evidence.get("source") or "").strip()
    score = evidence.get("score")
    prefix = f"- {chunk_id}" if chunk_id else "- (no_chunk_id)"
    if source:
        prefix += f" source={source}"
    if isinstance(score, (int, float)):
        prefix += f" score={float(score):.4f}"
    preview_limit = int(os.getenv("DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS", "180") or 180)
    content = str(evidence.get("content") or "").strip().replace("\n", " ")
    if content:
        prefix += "\n  " + _truncate_text(content, limit=max(80, preview_limit))
    provenance = evidence.get("provenance") if isinstance(evidence.get("provenance"), dict) else None
    if provenance:
        meta = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
        path = (
            provenance.get("path")
            or provenance.get("file_path")
            or provenance.get("source_path")
            or meta.get("path")
            or meta.get("file_path")
            or meta.get("source_path")
        )
        url = provenance.get("url") or provenance.get("source_url") or meta.get("url") or meta.get("source_url")
        page = provenance.get("page") or meta.get("page") or meta.get("page_number")
        if path:
            prefix += "\n  source_path=" + _truncate_text(str(path), limit=160)
        elif url:
            prefix += "\n  source_url=" + _truncate_text(str(url), limit=160)
        if page is not None and str(page).strip():
            prefix += f" page={page}"
        patterns = provenance.get("patterns")
        if isinstance(patterns, list) and patterns:
            prefix += "\n  patterns=" + json.dumps(patterns[:6], ensure_ascii=False, separators=(",", ":"), default=str)
    return prefix


def _render_tool_response(payload: Dict[str, Any]) -> str:
    tool_name = str(payload.get("tool_name") or payload.get("tool") or "").strip() or "unknown"
    call_id = str(payload.get("call_id") or "").strip()
    ok = payload.get("ok")
    route = payload.get("route")
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if ok is None:
        ok = False if result.get("error") else True

    summary = str(result.get("summary") or "").strip()
    if summary:
        summary = " ".join(summary.splitlines()).strip()
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    evidences = result.get("evidences") if isinstance(result.get("evidences"), list) else []

    lines: list[str] = []
    header = f"tool={tool_name}"
    if call_id:
        header += f" call_id={call_id}"
    if ok is not None:
        header += f" ok={bool(ok)}"
    if route:
        header += f" route={route}"
    lines.append(header)
    if summary:
        lines.append("summary=" + _truncate_text(summary, limit=420))

    latency_ms = diagnostics.get("latency_ms")
    if isinstance(latency_ms, (int, float)):
        lines.append(f"latency_ms={int(latency_ms)}")
    lines.append(f"evidence_count={len(evidences)}")

    if isinstance(diagnostics, dict):
        if diagnostics.get("unique_match_count") is not None:
            lines.append(f"unique_match_count={diagnostics.get('unique_match_count')}")
        if isinstance(diagnostics.get("match_chunk_ids"), list) and diagnostics.get("match_chunk_ids"):
            lines.append(
                "match_chunk_ids="
                + json.dumps(diagnostics.get("match_chunk_ids")[:12], ensure_ascii=False, separators=(",", ":"), default=str)
            )

    traversal = payload.get("traversal") if isinstance(payload.get("traversal"), dict) else None
    if traversal:
        lines.append(
            "traversal="
            + json.dumps(
                {
                    "hop_count": traversal.get("hop_count"),
                    "visited_nodes": len(traversal.get("visited_nodes") or []) if isinstance(traversal.get("visited_nodes"), list) else None,
                    "visited_edges": len(traversal.get("visited_edges") or []) if isinstance(traversal.get("visited_edges"), list) else None,
                    "seed_entities": traversal.get("seed_entities")[:8] if isinstance(traversal.get("seed_entities"), list) else None,
                    "retrieved_chunks": traversal.get("retrieved_chunks")[:12] if isinstance(traversal.get("retrieved_chunks"), list) else None,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )

    sample_n = int(os.getenv("DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT", "3") or 3)
    if sample_n < 0:
        sample_n = 0
    if evidences and sample_n:
        lines.append("evidence_samples:")
        for raw in evidences[:sample_n]:
            if isinstance(raw, dict):
                lines.append(_evidence_preview(raw))
    else:
        lines.append("evidence_samples: (none)")

    triple_samples: list[dict] = []
    for raw in evidences[: max(1, sample_n)]:
        if not isinstance(raw, dict):
            continue
        prov = raw.get("provenance")
        if not isinstance(prov, dict):
            continue
        triples = prov.get("triples")
        if isinstance(triples, list):
            for item in triples:
                if isinstance(item, dict) and item.get("head") and item.get("relation") and item.get("tail"):
                    triple_samples.append(item)
        if len(triple_samples) >= 6:
            break
    if triple_samples:
        lines.append("triple_samples:")
        for item in triple_samples[:6]:
            lines.append(f"- {item.get('head')} -[{item.get('relation')}]-> {item.get('tail')}")

    return "\n".join([ln for ln in lines if ln is not None and str(ln).strip()])


def _render_trace_payload(
    *,
    trace_tag: str,
    content: str,
    run_id: str,
) -> tuple[str, str]:
    normalized = (trace_tag or "").strip().lower() or "think"
    text = str(content or "")
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict) and "event" in parsed and "data" in parsed:
        event_type = str(parsed.get("event") or "event")
        payload = parsed.get("data")
        rendered = _format_weaver_progress(event_type, payload, run_id=run_id)
        if event_type.strip().lower() in {"error", "done"}:
            return "terminate", rendered
        return "progress", rendered

    if normalized == "tool_call" and isinstance(parsed, dict):
        return normalized, _render_tool_call(parsed)
    if normalized == "tool_response" and isinstance(parsed, dict):
        return normalized, _render_tool_response(parsed)
    return normalized, text


def _weaver_block(tag: str, content: str) -> str:
    normalized = (tag or "").strip().lower()
    if normalized not in _WEAVER_ALLOWED_TAGS:
        normalized = "progress"
    body = str(content or "")
    if body.endswith("\n"):
        body = body.rstrip("\n")
    if body:
        return f"<{normalized}>\n{body}\n</{normalized}>"
    return f"<{normalized}>\n</{normalized}>"

def _format_weaver_progress(event_type: str, payload: Any, *, run_id: str) -> str:
    """Render non-trace task events as human-readable progress blocks."""

    normalized = (event_type or "").strip().lower() or "event"
    if normalized == "heartbeat":
        return f"heartbeat run_id={run_id}"

    if not isinstance(payload, dict):
        return "\n".join(
            [
                f"event={normalized}",
                f"run_id={run_id}",
                f"payload={payload}",
            ]
        )

    stage = str(payload.get("stage") or "").strip()
    progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
    stage_record = payload.get("stage_record") if isinstance(payload.get("stage_record"), dict) else {}
    errors = payload.get("errors") if isinstance(payload.get("errors"), dict) else {}
    stage_history = payload.get("stage_history") if isinstance(payload.get("stage_history"), list) else []

    lines: list[str] = [f"event={normalized}", f"run_id={run_id}"]
    if stage:
        lines.append(f"stage={stage}")
    if progress:
        pct = progress.get("percent")
        step_idx = progress.get("step_index")
        step_total = progress.get("step_total")
        prog_stage = progress.get("stage")
        lines.append(f"progress.stage={prog_stage} progress.percent={pct} progress.step={step_idx}/{step_total}")
    if stage_record:
        rec_stage = stage_record.get("stage")
        rec_ts = stage_record.get("timestamp")
        lines.append(f"stage_record.stage={rec_stage} stage_record.timestamp={rec_ts}")
        rec_meta = stage_record.get("metadata")
        if isinstance(rec_meta, dict) and rec_meta:
            lines.append("stage_record.metadata=" + json.dumps(rec_meta, ensure_ascii=False, separators=(",", ":"), default=str))
    if errors:
        lines.append("errors=" + json.dumps(errors, ensure_ascii=False, separators=(",", ":"), default=str))
    if stage_history:
        lines.append("stage_history:")
        for item in stage_history:
            if not isinstance(item, dict):
                continue
            hist_stage = item.get("stage")
            hist_ts = item.get("timestamp")
            hist_meta = item.get("metadata")
            meta_text = ""
            if isinstance(hist_meta, dict) and hist_meta:
                meta_text = " " + json.dumps(hist_meta, ensure_ascii=False, separators=(",", ":"), default=str)
            lines.append(f"- {hist_ts} {hist_stage}{meta_text}")
    return "\n".join(lines)


async def _stream_events_weaver(run_id: str, *, last_event_id: int = -1):
    info = await TASKS.get(run_id)
    if not info:
        yield sse_text(_weaver_block("think", f"run_id not found: {run_id}"))
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
    saw_terminate = False
    saw_all_tools = False

    while True:
        await asyncio.sleep(0)
        pending_events: list[Dict[str, Any]] = []
        async with info.cond:
            while cursor + 1 >= len(info.events) and not info.done:
                try:
                    await asyncio.wait_for(info.cond.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield sse_text(_weaver_block("think", f"heartbeat run_id={run_id}"))
            next_index = cursor + 1
            while next_index < len(info.events):
                pending_events.append(info.events[next_index])
                next_index += 1

        for event in pending_events:
            cursor = int(event.get("id", cursor + 1))
            event_type = str(event.get("type") or "message")
            payload = event.get("payload") or {}
            if event_type == "trace" and isinstance(payload, dict):
                raw_tag = str(payload.get("trace_tag") or "think")
                content = str(payload.get("content") or "")
                tag, rendered = _render_trace_payload(trace_tag=raw_tag, content=content, run_id=run_id)
                if tag.strip().lower() == "terminate":
                    saw_terminate = True
                if tag.strip().lower() == "all_tools":
                    if saw_all_tools:
                        await asyncio.sleep(0)
                        continue
                    saw_all_tools = True
                yield sse_text(_weaver_block(tag, rendered))
                await asyncio.sleep(0)
                continue

            normalized_event_type = event_type.strip().lower()
            if normalized_event_type in {"progress", "result", "heartbeat"}:
                yield sse_text(_weaver_block("progress", _format_weaver_progress(event_type, payload, run_id=run_id)))
                await asyncio.sleep(0)
                continue
            if normalized_event_type in {"error"}:
                yield sse_text(_weaver_block("terminate", _format_weaver_progress(event_type, payload, run_id=run_id)))
                await asyncio.sleep(0)
                continue
            yield sse_text(_weaver_block("progress", _format_weaver_progress(event_type, payload, run_id=run_id)))
            await asyncio.sleep(0)

        if info.done and cursor + 1 >= len(info.events):
            if not saw_terminate:
                done_payload = {
                    "run_id": run_id,
                    "done": True,
                    "error": info.error,
                }
                yield sse_text(_weaver_block("terminate", _format_weaver_progress("done", done_payload, run_id=run_id)))
            yield sse_done()
            return


async def _stream_events_openai(
    run_id: str,
    *,
    last_event_id: int = -1,
    model_name: str,
):
    """Stream DeepSearch progress as Qwen(OpenAI-compatible) chat.completion.chunk SSE events."""

    chunk_id = new_chatcmpl_id()
    created = now_epoch_seconds()

    yield sse_json(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(role="assistant", content=""),
        )
    )

    info = await TASKS.get(run_id)
    if not info:
        missing_payload = with_trace_protocol(
            {"flow": "deepsearch", "event": "error", "data": {"run_id": run_id, "message": "run_id not found"}},
            run_id=run_id,
        )
        tool_calls = [
            {
                "index": 0,
                "id": f"call_progress_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": "rag_arc_progress",
                    "arguments": json.dumps(
                        missing_payload,
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    ),
                },
            }
        ]
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(tool_calls=tool_calls),
            )
        )
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(),
                finish_reason="stop",
            )
        )
        yield sse_done()
        return

    cursor = max(-1, last_event_id)
    sent_result = False

    while True:
        await asyncio.sleep(0)
        async with info.cond:
            while cursor + 1 >= len(info.events) and not info.done:
                try:
                    await asyncio.wait_for(info.cond.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    heartbeat = with_trace_protocol({"flow": "deepsearch", "event": "heartbeat", "data": {"run_id": run_id}}, run_id=run_id)
                    tool_calls = [
                        {
                            "index": 0,
                            "id": f"call_progress_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": "rag_arc_progress",
                                "arguments": json.dumps(
                                    heartbeat,
                                    ensure_ascii=False,
                                    default=str,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ]
                    yield sse_json(
                        openai_chat_completion_chunk(
                            chunk_id=chunk_id,
                            model=model_name,
                            created=created,
                            delta=delta_envelope(tool_calls=tool_calls),
                        )
                    )
            pending = info.events[cursor + 1 :]

        for event in pending:
            cursor = int(event.get("id", cursor + 1))
            payload = event.get("payload") or {}
            envelope = with_trace_protocol(
                {
                "flow": "deepsearch",
                "event": event.get("type") or "message",
                "id": cursor,
                "timestamp_ms": event.get("timestamp_ms") or int(time.time() * 1000),
                "data": {"run_id": run_id, **payload},
                },
                run_id=run_id,
            )
            tool_calls = [
                {
                    "index": 0,
                    "id": f"call_progress_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": "rag_arc_progress",
                        "arguments": json.dumps(
                            envelope,
                            ensure_ascii=False,
                            default=str,
                            separators=(",", ":"),
                        ),
                    },
                }
            ]
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(tool_calls=tool_calls),
                )
            )
            await asyncio.sleep(0)

            if not sent_result and (event.get("type") == "result" or info.done) and info.result:
                sent_result = True
                payload_envelope = with_trace_protocol({"flow": "deepsearch", "run_id": run_id, "result": info.result}, run_id=run_id)
                tool_calls = [
                    {
                        "index": 0,
                        "id": f"call_payload_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": "rag_arc_payload",
                            "arguments": json.dumps(
                                payload_envelope,
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ]
                yield sse_json(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(tool_calls=tool_calls),
                    )
                )
                await asyncio.sleep(0)

        if info.done and cursor + 1 >= len(info.events):
            if not sent_result and info.result:
                sent_result = True
                payload_envelope = with_trace_protocol({"flow": "deepsearch", "run_id": run_id, "result": info.result}, run_id=run_id)
                tool_calls = [
                    {
                        "index": 0,
                        "id": f"call_payload_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": "rag_arc_payload",
                            "arguments": json.dumps(
                                payload_envelope,
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ]
                yield sse_json(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(tool_calls=tool_calls),
                    )
                )
                await asyncio.sleep(0)

            done_payload = with_trace_protocol(
                {
                    "flow": "deepsearch",
                    "event": "done",
                    "data": {
                        "run_id": run_id,
                        "done": True,
                        "error": info.error,
                        "progress": info.last_progress.get("progress") if isinstance(info.last_progress, dict) else None,
                    },
                },
                run_id=run_id,
            )
            tool_calls = [
                {
                    "index": 0,
                    "id": f"call_progress_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": "rag_arc_progress",
                        "arguments": json.dumps(
                            done_payload,
                            ensure_ascii=False,
                            default=str,
                            separators=(",", ":"),
                        ),
                    },
                }
            ]
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(tool_calls=tool_calls),
                )
            )
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(),
                    finish_reason="stop",
                )
            )
            yield sse_done()
            return


async def _stream_events_redis(run_id: str, *, last_event_id: int = -1):
    task_queue = _get_task_queue()
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        yield format_sse(event="error", data={"run_id": run_id, "message": "run_id not found"})
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
    while True:
        task_run = task_queue.get_task_run(run_id) or {}
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        block_ms = 0 if done else 15000
        events = await asyncio.to_thread(
            task_queue.read_progress_events,
            run_id,
            last_seq=cursor,
            count=200,
            block_ms=block_ms,
        )
        for event in events:
            try:
                seq = int(event.get("seq", cursor + 1))
            except Exception:
                seq = cursor + 1
            cursor = seq
            payload = event.get("payload") or {}
            yield format_sse(
                event=str(event.get("status") or "message"),
                event_id=cursor,
                data={"run_id": run_id, **(payload if isinstance(payload, dict) else {"payload": payload})},
            )

        if done and not events:
            done_payload = {
                "run_id": run_id,
                "done": True,
                "error": task_run.get("error_message"),
                "progress": {"percent": task_run.get("progress_percent")} if task_run.get("progress_percent") is not None else None,
            }
            yield format_sse(event="done", data=done_payload, event_id=cursor + 1)
            yield sse_done()
            return
        if not events and not done:
            yield format_sse(event="heartbeat", data={"run_id": run_id})


async def _stream_events_weaver_redis(run_id: str, *, last_event_id: int = -1):
    task_queue = _get_task_queue()
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        yield sse_text(_weaver_block("think", f"run_id not found: {run_id}"))
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
    saw_terminate = False
    saw_all_tools = False

    while True:
        task_run = task_queue.get_task_run(run_id) or {}
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        block_ms = 0 if done else 15000
        events = await asyncio.to_thread(
            task_queue.read_progress_events,
            run_id,
            last_seq=cursor,
            count=200,
            block_ms=block_ms,
        )
        for event in events:
            try:
                seq = int(event.get("seq", cursor + 1))
            except Exception:
                seq = cursor + 1
            cursor = seq
            status = str(event.get("status") or "message")
            payload = event.get("payload") or {}
            if status == "trace" and isinstance(payload, dict):
                raw_tag = str(payload.get("trace_tag") or "think")
                content = str(payload.get("content") or "")
                tag, rendered = _render_trace_payload(trace_tag=raw_tag, content=content, run_id=run_id)
                if tag.strip().lower() == "terminate":
                    saw_terminate = True
                if tag.strip().lower() == "all_tools":
                    if saw_all_tools:
                        continue
                    saw_all_tools = True
                yield sse_text(_weaver_block(tag, rendered))
                continue

            normalized_status = status.strip().lower()
            if normalized_status in {"progress", "result", "heartbeat"}:
                yield sse_text(_weaver_block("progress", _format_weaver_progress(status, payload, run_id=run_id)))
                continue
            if normalized_status in {"error"}:
                yield sse_text(_weaver_block("terminate", _format_weaver_progress(status, payload, run_id=run_id)))
                continue
            yield sse_text(_weaver_block("progress", _format_weaver_progress(status, payload, run_id=run_id)))

        if done and not events:
            if not saw_terminate:
                done_payload = {
                    "run_id": run_id,
                    "done": True,
                    "error": task_run.get("error_message"),
                }
                yield sse_text(_weaver_block("terminate", _format_weaver_progress("done", done_payload, run_id=run_id)))
            yield sse_done()
            return
        if not events and not done:
            yield sse_text(_weaver_block("progress", _format_weaver_progress("heartbeat", {}, run_id=run_id)))


async def _stream_events_openai_redis(
    run_id: str,
    *,
    last_event_id: int = -1,
    model_name: str,
):
    chunk_id = new_chatcmpl_id()
    created = now_epoch_seconds()

    yield sse_json(
        openai_chat_completion_chunk(
            chunk_id=chunk_id,
            model=model_name,
            created=created,
            delta=delta_envelope(role="assistant", content=""),
        )
    )

    task_queue = _get_task_queue()
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        heartbeat = with_trace_protocol(
            {"flow": "deepsearch", "event": "error", "data": {"run_id": run_id, "message": "run_id not found"}},
            run_id=run_id,
        )
        tool_calls = [
            {
                "index": 0,
                "id": f"call_progress_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": "rag_arc_progress", "arguments": json.dumps(heartbeat, ensure_ascii=False, separators=(",", ":"))},
            }
        ]
        yield sse_json(
            openai_chat_completion_chunk(
                chunk_id=chunk_id,
                model=model_name,
                created=created,
                delta=delta_envelope(tool_calls=tool_calls),
            )
        )
        yield sse_json(openai_chat_completion_chunk(chunk_id=chunk_id, model=model_name, created=created, delta=delta_envelope(), finish_reason="stop"))
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
    sent_result = False

    while True:
        task_run = task_queue.get_task_run(run_id) or {}
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        block_ms = 0 if done else 15000
        events = await asyncio.to_thread(
            task_queue.read_progress_events,
            run_id,
            last_seq=cursor,
            count=200,
            block_ms=block_ms,
        )
        for event in events:
            try:
                seq = int(event.get("seq", cursor + 1))
            except Exception:
                seq = cursor + 1
            cursor = seq
            payload = event.get("payload") or {}
            envelope = with_trace_protocol(
                {
                    "flow": "deepsearch",
                    "event": str(event.get("status") or "message"),
                    "id": cursor,
                    "timestamp_ms": event.get("ts_ms") or int(time.time() * 1000),
                    "data": {"run_id": run_id, **(payload if isinstance(payload, dict) else {"payload": payload})},
                },
                run_id=run_id,
            )
            tool_calls = [
                {
                    "index": 0,
                    "id": f"call_progress_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": "rag_arc_progress", "arguments": json.dumps(envelope, ensure_ascii=False, default=str, separators=(",", ":"))},
                }
            ]
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(tool_calls=tool_calls),
                )
            )

        if done and not sent_result:
            result = task_queue.get_task_result(run_id)
            if result is not None:
                sent_result = True
                envelope = with_trace_protocol({"flow": "deepsearch", "run_id": run_id, "result": result}, run_id=run_id)
                tool_calls = [
                    {
                        "index": 0,
                        "id": f"call_progress_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": "rag_arc_progress", "arguments": json.dumps(envelope, ensure_ascii=False, default=str, separators=(",", ":"))},
                    }
                ]
                yield sse_json(
                    openai_chat_completion_chunk(
                        chunk_id=chunk_id,
                        model=model_name,
                        created=created,
                        delta=delta_envelope(tool_calls=tool_calls),
                    )
                )
        if done and not events:
            yield sse_json(openai_chat_completion_chunk(chunk_id=chunk_id, model=model_name, created=created, delta=delta_envelope(), finish_reason="stop"))
            yield sse_done()
            return
        if not events and not done:
            heartbeat = with_trace_protocol({"flow": "deepsearch", "event": "heartbeat", "data": {"run_id": run_id}}, run_id=run_id)
            tool_calls = [
                {
                    "index": 0,
                    "id": f"call_progress_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": "rag_arc_progress", "arguments": json.dumps(heartbeat, ensure_ascii=False, default=str, separators=(",", ":"))},
                }
            ]
            yield sse_json(
                openai_chat_completion_chunk(
                    chunk_id=chunk_id,
                    model=model_name,
                    created=created,
                    delta=delta_envelope(tool_calls=tool_calls),
                )
            )

@router.post("/run", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def run_deepsearch(
    request: DeepSearchRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Graph-first DeepSearch entry point."""

    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    effective_owner = request.owner_id or current_user.id
    if request.owner_id and not is_admin_owner(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators may override owner scope",
        )

    service = _get_deepsearch_service()
    try:
        result = await service.run(
            request.question,
            owner_id=str(effective_owner),
            metadata=request.metadata,
        )
    except Exception as exc:  # pragma: no cover - rely on logging upstream
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DeepSearch execution failed: {exc}",
        ) from exc

    graph_store = _get_graph_store()
    trimmed = trim_deepsearch_payload(
        result,
        include_evidence=request.include_evidence,
        graph_store=graph_store,
    )
    return JSONResponse(content=trimmed)


class DeepSearchAsyncRequest(BaseModel):
    question: str = Field(..., description="User question, must be a non-empty string")
    owner_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional owner override, limited to admin users",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata merged into DeepSearch state",
    )
    include_evidence: bool = Field(
        default=False,
        description="When true, attach chunk/graph summaries to the response.",
    )


async def _schedule_deepsearch(
    request: DeepSearchAsyncRequest,
    *,
    effective_owner: uuid.UUID,
) -> JSONResponse:
    run_id = new_run_id()
    task_queue = _get_task_queue()
    task_queue.create_task_run(
        task_run_id=run_id,
        task_type="deepsearch",
        owner_id=effective_owner,
        resource_id=run_id,
        metadata={"include_evidence": request.include_evidence, "metadata": request.metadata or {}, "executor": "api"},
    )

    if _use_celery():
        from application.deepsearch.celery_tasks import run_deepsearch as run_deepsearch_task

        queue = os.getenv("CELERY_QUEUE_DEEPSEARCH", "deepsearch")
        run_deepsearch_task.apply_async(
            kwargs={
                "question": request.question,
                "owner_id": str(effective_owner),
                "metadata": request.metadata,
                "include_evidence": request.include_evidence,
            },
            task_id=run_id,
            queue=queue,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "run_id": run_id,
                "status": "scheduled",
                "progress_url": f"/deepsearch/progress/{run_id}",
                "stream_url": f"/deepsearch/stream/{run_id}",
                "result_url": f"/deepsearch/result/{run_id}",
            },
        )

    service = _get_deepsearch_service()
    await TASKS.create(run_id)

    def _listener(record: Dict[str, Any], state) -> None:  # noqa: ANN001
        progress = _stage_progress(getattr(state, "stage", "unknown"))
        payload = with_trace_protocol(
            {
                "run_id": run_id,
                "stage": getattr(state, "stage", "unknown"),
                "stage_record": dict(record),
                "stage_history": list(getattr(state, "stage_history", []) or []),
                "errors": list(getattr(state, "errors", []) or []),
                "progress": progress,
            },
            run_id=run_id,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(TASKS.publish(run_id, event_type="progress", payload=payload))
        except RuntimeError:
            return

    async def _runner() -> None:
        task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
        emitter = make_inprocess_trace_emitter(
            run_id=run_id,
            publish=lambda event_type, payload: TASKS.publish(run_id, event_type=event_type, payload=payload),
        )
        token = set_trace_emitter(emitter)
        try:
            await emit_trace(
                "think",
                f"Received question. Starting graph-first DeepSearch run.\nrun_id={run_id}",
                meta={"run_id": run_id, "external_allowed": False},
            )
            try:
                planner = getattr(service, "planner", None)
                await emit_trace(
                    "all_tools",
                    render_all_tools_block(include_llm_tools=bool(getattr(planner, "include_llm_tools_in_catalog", False))),
                    meta={"run_id": run_id},
                )
            except Exception:
                pass
            result = await service.run(
                request.question,
                owner_id=str(effective_owner),
                metadata=request.metadata,
                run_id=run_id,
                stage_listener=_listener,
            )
        except Exception as exc:  # noqa: BLE001
            await TASKS.publish(
                run_id,
                event_type="progress",
                payload=with_trace_protocol(
                    {
                        "run_id": run_id,
                        "stage": "failed",
                        "errors": [{"message": str(exc)}],
                        "progress": _stage_progress("failed"),
                    },
                    run_id=run_id,
                ),
            )
            await TASKS.mark_done(run_id, error=str(exc))
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=str(exc),
                finished=True,
            )
            await emit_trace(
                "terminate",
                f"DeepSearch failed.\nrun_id={run_id}\nerror={exc}",
                meta={"run_id": run_id, "ok": False},
            )
            reset_trace_emitter(token)
            return

        graph_store = _get_graph_store()
        trimmed = trim_deepsearch_payload(
            result,
            include_evidence=request.include_evidence,
            graph_store=graph_store,
        )
        try:
            report_block = trimmed.get("report") if isinstance(trimmed, dict) else None
            report_text = report_block.get("answer") if isinstance(report_block, dict) else None
            if isinstance(report_text, str) and report_text.strip():
                await emit_trace(
                    "write",
                    report_text,
                    meta={"run_id": run_id, "question": trimmed.get("question")},
                )
        except Exception:
            pass
        await TASKS.publish(
            run_id,
            event_type="result",
            payload=with_trace_protocol(
                {
                    "run_id": run_id,
                    "stage": trimmed.get("state", {}).get("stage"),
                    "progress": _stage_progress(trimmed.get("state", {}).get("stage", "")),
                },
                run_id=run_id,
            ),
        )
        await TASKS.mark_done(run_id, result=trimmed)
        task_queue.update_task_run(run_id, state=TaskState.SUCCESS, progress_percent=100, finished=True)
        await emit_trace(
            "terminate",
            f"DeepSearch completed.\nrun_id={run_id}",
            meta={"run_id": run_id, "ok": True},
        )
        reset_trace_emitter(token)

    task = asyncio.create_task(_runner())
    info = await TASKS.get(run_id)
    if info:
        info.task = task

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "run_id": run_id,
            "status": "scheduled",
            "progress_url": f"/deepsearch/progress/{run_id}",
            "stream_url": f"/deepsearch/stream/{run_id}",
            "result_url": f"/deepsearch/result/{run_id}",
        },
    )


@router.post("/run_async", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED)
async def run_deepsearch_async(
    request: DeepSearchAsyncRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Schedule DeepSearch in background and return run_id + SSE URLs."""

    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    effective_owner = request.owner_id or current_user.id
    if request.owner_id and not is_admin_owner(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators may override owner scope",
        )

    return await _schedule_deepsearch(request, effective_owner=effective_owner)


@router.get("/progress/{run_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_progress(
    run_id: str,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if _use_celery():
        task_queue = _get_task_queue()
        task_run = task_queue.get_task_run(run_id)
        if not task_run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
        _assert_task_owner(task_run, user_id=current_user.id)
        latest = task_queue.get_latest_progress_event_filtered(run_id, exclude_statuses={"trace"}) or task_queue.get_latest_progress_event(run_id) or {}
        payload = dict((latest.get("payload") or {}) if isinstance(latest.get("payload"), dict) else {})
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        payload.setdefault("run_id", run_id)
        payload.setdefault("done", done)
        if task_run.get("error_message"):
            payload.setdefault("error", task_run.get("error_message"))
        return payload
    info = await TASKS.get(run_id)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    payload = dict(info.last_progress or {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("done", info.done)
    if info.error:
        payload.setdefault("error", info.error)
    return payload


@router.get("/result/{run_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_result(
    run_id: str,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if _use_celery():
        task_queue = _get_task_queue()
        task_run = task_queue.get_task_run(run_id)
        if not task_run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
        _assert_task_owner(task_run, user_id=current_user.id)
        state = str(task_run.get("state") or "")
        if state not in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="run not finished")
        if state != TaskState.SUCCESS.value:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(task_run.get("error_message") or "task failed"))
        result = task_queue.get_task_result(run_id)
        return result or {"run_id": run_id, "done": True, "result": None}
    info = await TASKS.get(run_id)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    if not info.done:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="run not finished")
    if info.error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=info.error)
    return info.result or {"run_id": run_id, "done": True, "result": None}


@router.get("/stream/{run_id}")
async def stream_progress(
    run_id: str,
    current_user: Annotated[User | None, Depends(get_current_user)],
    last_event_id: int = -1,
    format: Literal["legacy", "openai", "weaver"] = "legacy",
):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    if format == "weaver":
        if _use_celery():
            task_queue = _get_task_queue()
            task_run = task_queue.get_task_run(run_id)
            if not task_run:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
            _assert_task_owner(task_run, user_id=current_user.id)
            return StreamingResponse(
                _stream_events_weaver_redis(run_id, last_event_id=last_event_id),
                media_type="text/event-stream",
                headers=headers,
            )
        return StreamingResponse(
            _stream_events_weaver(run_id, last_event_id=last_event_id),
            media_type="text/event-stream",
            headers=headers,
        )
    if format == "openai":
        model_name = os.getenv("CHAT_MODEL_NAME") or os.getenv("OPENAI_CHAT_MODEL") or "rag-arc-deepsearch"
        if _use_celery():
            task_queue = _get_task_queue()
            task_run = task_queue.get_task_run(run_id)
            if not task_run:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
            _assert_task_owner(task_run, user_id=current_user.id)
            return StreamingResponse(
                _stream_events_openai_redis(run_id, last_event_id=last_event_id, model_name=model_name),
                media_type="text/event-stream",
                headers=headers,
            )
        return StreamingResponse(
            _stream_events_openai(run_id, last_event_id=last_event_id, model_name=model_name),
            media_type="text/event-stream",
            headers=headers,
        )
    if _use_celery():
        task_queue = _get_task_queue()
        task_run = task_queue.get_task_run(run_id)
        if not task_run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
        _assert_task_owner(task_run, user_id=current_user.id)
        return StreamingResponse(
            _stream_events_redis(run_id, last_event_id=last_event_id),
            media_type="text/event-stream",
            headers=headers,
        )
    return StreamingResponse(
        _stream_events(run_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )
