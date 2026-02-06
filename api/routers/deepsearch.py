import asyncio
import logging
import os
import uuid
from typing import Annotated, Any, Dict, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.deepsearch.tasks import TASKS, new_run_id
from api.routers.auth import get_current_user
from api.routers.deepsearch_models import DeepSearchAsyncRequest, DeepSearchRequest
from api.routers.deepsearch_streaming import (
    stream_events_inprocess,
    stream_events_openai_inprocess,
    stream_events_openai_redis,
    stream_events_redis,
    stream_events_weaver_inprocess,
    stream_events_weaver_redis,
)
from application.deepsearch.trace_emitter import make_inprocess_trace_emitter
from application.rag_inference.module import RAGInference
from core.deepsearch.tooling.all_tools import render_all_tools_block
from core.deepsearch.trace import emit_trace, reset_trace_emitter, set_trace_emitter, with_trace_protocol
from core.deepsearch.utils.progress import compute_deepsearch_progress
from core.presentation.deepsearch_payload import trim_deepsearch_payload
from core.utils.owner_guard import is_admin_owner
from encapsulation.data_model.orm_models import User
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from framework.register import Register

router = APIRouter(prefix="/deepsearch", tags=["deepsearch"])
registrator = Register()
logger = logging.getLogger(__name__)

# Optional override for tests that want a stable queue instance.
TASK_QUEUE: RedisTaskQueue | None = None


def _get_task_queue() -> RedisTaskQueue:
    override = TASK_QUEUE
    if isinstance(override, RedisTaskQueue):
        current_ns = os.getenv("MQ_NAMESPACE", "rag-arc:mq")
        if override.settings.namespace == current_ns:
            return override
    return RedisTaskQueue.from_env()


def _use_celery() -> bool:
    return os.getenv("TASK_QUEUE_MODE", "inprocess").strip().lower() == "celery"


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


def _assert_inprocess_task_owner(info: Any, *, user_id: uuid.UUID) -> None:
    """Enforce per-owner isolation for in-process DeepSearch runs."""
    if is_admin_owner(user_id):
        return
    raw_owner = getattr(info, "owner_id", None)
    if not raw_owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")
    try:
        owner_uuid = uuid.UUID(str(raw_owner))
    except Exception:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id") from None
    if owner_uuid != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")


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
    return compute_deepsearch_progress(stage)


@router.post("/run", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def run_deepsearch(
    request: DeepSearchRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Graph-first DeepSearch entry point."""

    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    effective_owner = request.owner_id or current_user.id
    if request.owner_id and not is_admin_owner(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators may override owner scope")

    service = _get_deepsearch_service()
    try:
        result = await service.run(
            request.question,
            owner_id=str(effective_owner),
            metadata=request.metadata,
        )
    except Exception as exc:  # pragma: no cover - rely on logging upstream
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DeepSearch execution failed: {exc}") from exc

    graph_store = _get_graph_store()
    trimmed = trim_deepsearch_payload(result, include_evidence=request.include_evidence, graph_store=graph_store)
    return JSONResponse(content=trimmed)


async def _schedule_deepsearch(
    request: DeepSearchAsyncRequest,
    *,
    effective_owner: uuid.UUID,
) -> JSONResponse:
    run_id = new_run_id()
    if _use_celery():
        task_queue = _get_task_queue()
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="deepsearch",
            owner_id=effective_owner,
            resource_id=run_id,
            metadata={"include_evidence": request.include_evidence, "metadata": request.metadata or {}, "executor": "api"},
        )
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
    await TASKS.create(run_id, owner_id=str(effective_owner))

    def _listener(record: Dict[str, Any], state) -> None:  # noqa: ANN001
        progress = compute_deepsearch_progress(
            getattr(state, "stage", "unknown"),
            stage_history=getattr(state, "stage_history", None),
            stage_record=record,
        )
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
        # In in-process mode, never block the event loop on Redis IO.
        # Redis mirroring (best-effort) is handled by DeepSearchTaskRegistry.publish().
        emitter = make_inprocess_trace_emitter(
            run_id=run_id,
            publish=lambda event_type, payload: TASKS.publish(run_id, event_type=event_type, payload=payload),
        )
        token = set_trace_emitter(emitter)
        try:
            await emit_trace(
                "think",
                f"Received question. Starting graph-first DeepSearch run.\nrun_id={run_id}",
                meta={"run_id": run_id},
            )
            try:
                include_llm_tools = True
                think_cfg = getattr(getattr(service, "graph_loop", None), "_think_config", None)
                if isinstance(think_cfg, dict) and "include_llm_tools" in think_cfg:
                    include_llm_tools = bool(think_cfg.get("include_llm_tools"))
                registry = None
                try:
                    registry = getattr(getattr(getattr(service, "tool_manager", None), "local_registry", None), "tool_hint_registry", None)
                except Exception:
                    registry = None
                await emit_trace(
                    "all_tools",
                    render_all_tools_block(include_llm_tools=include_llm_tools, registry=registry),
                    meta={"run_id": run_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to emit DeepSearch all_tools trace: %s", exc, exc_info=True)
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
                    {"run_id": run_id, "stage": "failed", "errors": [{"message": str(exc)}], "progress": _stage_progress("failed")},
                    run_id=run_id,
                ),
            )
            await TASKS.mark_done(run_id, error=str(exc))
            await emit_trace("terminate", f"DeepSearch failed.\nrun_id={run_id}\nerror={exc}", meta={"run_id": run_id, "ok": False})
            reset_trace_emitter(token)
            return

        graph_store = _get_graph_store()
        trimmed = trim_deepsearch_payload(result, include_evidence=request.include_evidence, graph_store=graph_store)
        try:
            report_block = trimmed.get("report") if isinstance(trimmed, dict) else None
            report_text = report_block.get("answer") if isinstance(report_block, dict) else None
            if isinstance(report_text, str) and report_text.strip():
                await emit_trace("write", report_text, meta={"run_id": run_id, "question": trimmed.get("question")})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to emit DeepSearch write trace: %s", exc, exc_info=True)
        await TASKS.publish(
            run_id,
            event_type="result",
            payload=with_trace_protocol(
                {"run_id": run_id, "stage": trimmed.get("state", {}).get("stage"), "progress": _stage_progress(trimmed.get("state", {}).get("stage", ""))},
                run_id=run_id,
            ),
        )
        await TASKS.mark_done(run_id, result=trimmed)
        await emit_trace("terminate", f"DeepSearch completed.\nrun_id={run_id}", meta={"run_id": run_id, "ok": True})
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    effective_owner = request.owner_id or current_user.id
    if request.owner_id and not is_admin_owner(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators may override owner scope")

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
        latest = (
            task_queue.get_latest_progress_event_filtered(run_id, exclude_statuses={"trace"})
            or task_queue.get_latest_progress_event(run_id)
            or {}
        )
        payload = dict((latest.get("payload") or {}) if isinstance(latest.get("payload"), dict) else {})
        state = str(task_run.get("state") or "")
        done = state in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}
        payload.setdefault("run_id", run_id)
        payload.setdefault("done", done)
        if task_run.get("error_message"):
            payload.setdefault("error", task_run.get("error_message"))
        if payload.get("done") is True and not payload.get("error"):
            payload["stage"] = "done"
            payload["progress"] = _stage_progress("done")
        return payload

    info = await TASKS.get(run_id)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    _assert_inprocess_task_owner(info, user_id=current_user.id)
    payload = dict(info.last_progress or {})
    payload.setdefault("run_id", run_id)
    payload.setdefault("done", info.done)
    if info.error:
        payload.setdefault("error", info.error)
    if payload.get("done") is True and not payload.get("error"):
        payload["stage"] = "done"
        payload["progress"] = _stage_progress("done")
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
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(task_run.get("error_message") or "task failed"),
            )
        result = task_queue.get_task_result(run_id)
        return result or {"run_id": run_id, "done": True, "result": None}

    info = await TASKS.get(run_id)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    _assert_inprocess_task_owner(info, user_id=current_user.id)
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

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}

    if not _use_celery():
        info = await TASKS.get(run_id)
        if not info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
        _assert_inprocess_task_owner(info, user_id=current_user.id)

    if format == "weaver":
        if _use_celery():
            task_queue = _get_task_queue()
            task_run = task_queue.get_task_run(run_id)
            if not task_run:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
            _assert_task_owner(task_run, user_id=current_user.id)
            return StreamingResponse(
                stream_events_weaver_redis(task_queue, run_id, last_event_id=last_event_id),
                media_type="text/event-stream",
                headers=headers,
            )
        return StreamingResponse(
            stream_events_weaver_inprocess(run_id, last_event_id=last_event_id),
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
                stream_events_openai_redis(task_queue, run_id, last_event_id=last_event_id, model_name=model_name),
                media_type="text/event-stream",
                headers=headers,
            )
        return StreamingResponse(
            stream_events_openai_inprocess(run_id, last_event_id=last_event_id, model_name=model_name),
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
            stream_events_redis(task_queue, run_id, last_event_id=last_event_id),
            media_type="text/event-stream",
            headers=headers,
        )
    return StreamingResponse(
        stream_events_inprocess(run_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )
