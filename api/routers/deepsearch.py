import uuid
import asyncio
from typing import Annotated, Any, Dict, Optional

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
from api.deepsearch.tasks import TASKS, format_sse, new_run_id

router = APIRouter(prefix="/deepsearch", tags=["deepsearch"])
registrator = Register()


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
        yield format_sse(event="error", data={"run_id": run_id, "message": "run_id not found"})
        return

    cursor = max(-1, last_event_id)
    while True:
        await asyncio.sleep(0)
        async with info.cond:
            while cursor + 1 >= len(info.events) and not info.done:
                try:
                    await asyncio.wait_for(info.cond.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield format_sse(event="heartbeat", data={"run_id": run_id})
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
            yield format_sse(event="done", data=done_payload, event_id=cursor + 1)
            return


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
    service = _get_deepsearch_service()
    run_id = new_run_id()
    await TASKS.create(run_id)

    def _listener(record: Dict[str, Any], state) -> None:  # noqa: ANN001
        progress = _stage_progress(getattr(state, "stage", "unknown"))
        payload = {
            "stage": getattr(state, "stage", "unknown"),
            "stage_record": dict(record),
            "stage_history": list(getattr(state, "stage_history", []) or []),
            "errors": list(getattr(state, "errors", []) or []),
            "progress": progress,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(TASKS.publish(run_id, event_type="progress", payload=payload))
        except RuntimeError:
            return

    async def _runner() -> None:
        try:
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
                payload={
                    "stage": "failed",
                    "errors": [{"message": str(exc)}],
                    "progress": _stage_progress("failed"),
                },
            )
            await TASKS.mark_done(run_id, error=str(exc))
            return

        graph_store = _get_graph_store()
        trimmed = trim_deepsearch_payload(
            result,
            include_evidence=request.include_evidence,
            graph_store=graph_store,
        )
        await TASKS.publish(
            run_id,
            event_type="result",
            payload={
                "stage": trimmed.get("state", {}).get("stage"),
                "progress": _stage_progress(trimmed.get("state", {}).get("stage", "")),
            },
        )
        await TASKS.mark_done(run_id, result=trimmed)

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
):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream_events(run_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )
