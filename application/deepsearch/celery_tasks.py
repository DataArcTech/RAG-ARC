import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

from celery.exceptions import Retry

from encapsulation.message_queue.celery_app import app as celery_app
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from core.presentation.deepsearch_payload import trim_deepsearch_payload

from application.celery_bootstrap import ensure_initialized

logger = logging.getLogger(__name__)


def _parse_uuid(value: str) -> uuid.UUID:
    value = (value or "").strip()
    if len(value) == 32:
        return uuid.UUID(hex=value)
    return uuid.UUID(value)


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


def _get_deepsearch_service():
    import app_registration

    return app_registration.registrator.get_object("deepsearch_service")


def _get_graph_store() -> Any | None:
    import app_registration

    try:
        rag = app_registration.registrator.get_object("rag_inference")
    except KeyError:
        return None
    try:
        from application.rag_inference.module import RAGInference

        if isinstance(rag, RAGInference):
            return rag.get_graph_store()
    except Exception:
        return None
    return None


@celery_app.task(bind=True, name="rag_arc.deepsearch.run")
def run_deepsearch(
    self,
    *,
    question: str,
    owner_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    include_evidence: bool = False,
) -> Dict[str, Any]:
    ensure_initialized()

    run_id = str(getattr(self.request, "id", "") or uuid.uuid4().hex)
    task_queue = RedisTaskQueue.from_env()
    owner_uuid = _parse_uuid(owner_id)

    if not task_queue.get_task_run(run_id):
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="deepsearch",
            owner_id=owner_uuid,
            resource_id=run_id,
            metadata={"include_evidence": include_evidence, "metadata": metadata or {}, "executor": "celery"},
        )

    service = _get_deepsearch_service()

    def _listener(record: Dict[str, Any], state) -> None:  # noqa: ANN001
        stage = getattr(state, "stage", "unknown")
        progress = _stage_progress(stage)
        payload = {
            "stage": stage,
            "stage_record": dict(record),
            "stage_history": list(getattr(state, "stage_history", []) or []),
            "errors": list(getattr(state, "errors", []) or []),
            "progress": progress,
        }
        try:
            task_queue.append_progress_event(
                flow="deepsearch",
                task_run_id=run_id,
                stage=str(stage),
                status="progress",
                percent=int(progress.get("percent") or 0),
                resource_id=run_id,
                payload=payload,
            )
            task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=int(progress.get("percent") or 1))
        except Exception:
            return

    task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)

    try:
        result = asyncio.run(
            service.run(
                question,
                owner_id=str(owner_uuid),
                metadata=metadata,
                run_id=run_id,
                stage_listener=_listener,
            )
        )
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        max_retries = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        countdown = int(os.getenv("CELERY_TASK_RETRY_COUNTDOWN_SECONDS", "5"))
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            try:
                task_queue.append_progress_event(
                    flow="deepsearch",
                    task_run_id=run_id,
                    stage=str(getattr(getattr(self, "request", None), "retries", 0) or 0),
                    status="retry",
                    percent=0,
                    resource_id=run_id,
                    payload={"stage": "retry", "error": err, "retry_in_seconds": countdown},
                )
                task_queue.update_task_run(
                    run_id,
                    state=TaskState.PENDING,
                    progress_percent=0,
                    error_message=f"retrying: {err}",
                    metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
                )
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        logger.exception("DeepSearch failed (run_id=%s): %s", run_id, err)
        task_queue.append_progress_event(
            flow="deepsearch",
            task_run_id=run_id,
            stage="failed",
            status="error",
            percent=100,
            resource_id=run_id,
            payload={"stage": "failed", "errors": [{"message": err}], "progress": _stage_progress("failed")},
        )
        task_queue.update_task_run(
            run_id,
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        return {"run_id": run_id, "done": True, "error": err}

    graph_store = _get_graph_store()
    trimmed = trim_deepsearch_payload(
        result,
        include_evidence=include_evidence,
        graph_store=graph_store,
    )

    task_queue.set_task_result(run_id, trimmed)
    task_queue.append_progress_event(
        flow="deepsearch",
        task_run_id=run_id,
        stage=str(trimmed.get("state", {}).get("stage") or "reported"),
        status="result",
        percent=100,
        resource_id=run_id,
        payload={"stage": trimmed.get("state", {}).get("stage"), "progress": _stage_progress(trimmed.get("state", {}).get("stage", ""))},
    )
    task_queue.update_task_run(
        run_id,
        state=TaskState.SUCCESS,
        progress_percent=100,
        finished=True,
        result_ref=task_queue.settings.key_task_result(run_id),
    )
    return {"run_id": run_id, "done": True}
