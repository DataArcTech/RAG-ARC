import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict

from api.deepsearch.tasks import TASKS, format_sse
from api.sse import (
    delta_envelope,
    new_chatcmpl_id,
    now_epoch_seconds,
    openai_chat_completion_chunk,
    sse_done,
    sse_json,
    sse_text,
)
from core.deepsearch.trace import with_trace_protocol
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState

from api.routers.deepsearch_weaver_render import (
    format_weaver_progress,
    render_trace_payload,
    weaver_block,
)


async def stream_events_inprocess(run_id: str, *, last_event_id: int = -1) -> AsyncIterator[str]:
    info = await TASKS.get(run_id)
    if not info:
        yield format_sse(event="error", data=with_trace_protocol({"run_id": run_id, "message": "run_id not found"}, run_id=run_id))
        yield sse_done()
        return

    cursor = max(-1, int(last_event_id))
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
            yield format_sse(event=event.get("type") or "message", event_id=cursor, data={"run_id": run_id, **payload})

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


async def stream_events_weaver_inprocess(run_id: str, *, last_event_id: int = -1) -> AsyncIterator[str]:
    info = await TASKS.get(run_id)
    if not info:
        yield sse_text(weaver_block("think", f"run_id not found: {run_id}"))
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
                    yield sse_text(weaver_block("think", f"heartbeat run_id={run_id}"))
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
                tag, rendered = render_trace_payload(trace_tag=raw_tag, content=content, run_id=run_id)
                if tag.strip().lower() == "terminate":
                    saw_terminate = True
                if tag.strip().lower() == "all_tools":
                    if saw_all_tools:
                        await asyncio.sleep(0)
                        continue
                    saw_all_tools = True
                yield sse_text(weaver_block(tag, rendered))
                await asyncio.sleep(0)
                continue

            normalized_event_type = event_type.strip().lower()
            if normalized_event_type in {"progress", "result", "heartbeat"}:
                yield sse_text(weaver_block("progress", format_weaver_progress(event_type, payload, run_id=run_id)))
                await asyncio.sleep(0)
                continue
            if normalized_event_type in {"error"}:
                yield sse_text(weaver_block("terminate", format_weaver_progress(event_type, payload, run_id=run_id)))
                await asyncio.sleep(0)
                continue
            yield sse_text(weaver_block("progress", format_weaver_progress(event_type, payload, run_id=run_id)))
            await asyncio.sleep(0)

        if info.done and cursor + 1 >= len(info.events):
            if not saw_terminate:
                done_payload = {"run_id": run_id, "done": True, "error": info.error}
                yield sse_text(weaver_block("terminate", format_weaver_progress("done", done_payload, run_id=run_id)))
            yield sse_done()
            return


async def stream_events_openai_inprocess(
    run_id: str,
    *,
    last_event_id: int = -1,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream DeepSearch progress as OpenAI-compatible chat.completion.chunk SSE events."""

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
                    "arguments": json.dumps(missing_payload, ensure_ascii=False, default=str, separators=(",", ":")),
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

    cursor = max(-1, int(last_event_id))
    sent_result = False

    while True:
        await asyncio.sleep(0)
        async with info.cond:
            while cursor + 1 >= len(info.events) and not info.done:
                try:
                    await asyncio.wait_for(info.cond.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    heartbeat = with_trace_protocol(
                        {"flow": "deepsearch", "event": "heartbeat", "data": {"run_id": run_id}},
                        run_id=run_id,
                    )
                    tool_calls = [
                        {
                            "index": 0,
                            "id": f"call_progress_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": "rag_arc_progress",
                                "arguments": json.dumps(heartbeat, ensure_ascii=False, default=str, separators=(",", ":")),
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
                        "arguments": json.dumps(envelope, ensure_ascii=False, default=str, separators=(",", ":")),
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
                            "arguments": json.dumps(payload_envelope, ensure_ascii=False, default=str, separators=(",", ":")),
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
                            "arguments": json.dumps(payload_envelope, ensure_ascii=False, default=str, separators=(",", ":")),
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
                        "arguments": json.dumps(done_payload, ensure_ascii=False, default=str, separators=(",", ":")),
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


async def stream_events_redis(
    task_queue: RedisTaskQueue,
    run_id: str,
    *,
    last_event_id: int = -1,
) -> AsyncIterator[str]:
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
        events = await asyncio.to_thread(task_queue.read_progress_events, run_id, last_seq=cursor, count=200, block_ms=block_ms)
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


async def stream_events_weaver_redis(
    task_queue: RedisTaskQueue,
    run_id: str,
    *,
    last_event_id: int = -1,
) -> AsyncIterator[str]:
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        yield sse_text(weaver_block("think", f"run_id not found: {run_id}"))
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
        events = await asyncio.to_thread(task_queue.read_progress_events, run_id, last_seq=cursor, count=200, block_ms=block_ms)
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
                tag, rendered = render_trace_payload(trace_tag=raw_tag, content=content, run_id=run_id)
                if tag.strip().lower() == "terminate":
                    saw_terminate = True
                if tag.strip().lower() == "all_tools":
                    if saw_all_tools:
                        continue
                    saw_all_tools = True
                yield sse_text(weaver_block(tag, rendered))
                continue

            normalized_status = status.strip().lower()
            if normalized_status in {"progress", "result", "heartbeat"}:
                yield sse_text(weaver_block("progress", format_weaver_progress(status, payload, run_id=run_id)))
                continue
            if normalized_status in {"error"}:
                yield sse_text(weaver_block("terminate", format_weaver_progress(status, payload, run_id=run_id)))
                continue
            yield sse_text(weaver_block("progress", format_weaver_progress(status, payload, run_id=run_id)))

        if done and not events:
            if not saw_terminate:
                done_payload = {"run_id": run_id, "done": True, "error": task_run.get("error_message")}
                yield sse_text(weaver_block("terminate", format_weaver_progress("done", done_payload, run_id=run_id)))
            yield sse_done()
            return
        if not events and not done:
            yield sse_text(weaver_block("progress", format_weaver_progress("heartbeat", {}, run_id=run_id)))


async def stream_events_openai_redis(
    task_queue: RedisTaskQueue,
    run_id: str,
    *,
    last_event_id: int = -1,
    model_name: str,
) -> AsyncIterator[str]:
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

    cursor = max(-1, int(last_event_id))
    sent_result = False

    while True:
        task_run = task_queue.get_task_run(run_id) or {}
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        block_ms = 0 if done else 15000
        events = await asyncio.to_thread(task_queue.read_progress_events, run_id, last_seq=cursor, count=200, block_ms=block_ms)
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

