import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from celery.exceptions import Retry

from config.application.deepsearch_task_defaults import (
    DEFAULT_CELERY_TASK_MAX_RETRIES,
    DEFAULT_CELERY_TASK_RETRY_COUNTDOWN_SECONDS,
)
from encapsulation.message_queue.celery_app import app as celery_app
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from core.presentation.deepsearch_payload import trim_deepsearch_payload

from application.celery_bootstrap import ensure_initialized
from application.deepsearch.trace_emitter import make_redis_trace_emitter
from core.deepsearch.trace import emit_trace, reset_trace_emitter, set_trace_emitter, with_trace_protocol
from core.deepsearch.tooling.all_tools import render_all_tools_block
from core.deepsearch.utils.progress import compute_deepsearch_progress

logger = logging.getLogger(__name__)


def _parse_uuid(value: str) -> uuid.UUID:
    value = (value or "").strip()
    if len(value) == 32:
        return uuid.UUID(hex=value)
    return uuid.UUID(value)


def _stage_progress(stage: str) -> Dict[str, Any]:
    return compute_deepsearch_progress(stage)


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

    existing = task_queue.get_task_result(run_id)
    if existing is not None:
        task_queue.append_progress_event(
            flow="deepsearch",
            task_run_id=run_id,
            stage="done",
            status="dedup",
            percent=100,
            resource_id=run_id,
            payload=with_trace_protocol(
                {"run_id": run_id, "stage": "done", "dedup": True, "progress": _stage_progress("done")},
                run_id=run_id,
            ),
        )
        task_queue.update_task_run(
            run_id,
            state=TaskState.SUCCESS,
            progress_percent=100,
            finished=True,
            result_ref=task_queue.settings.key_task_result(run_id),
            metadata_patch={"dedup": True},
        )
        return {"run_id": run_id, "done": True, "dedup": True}

    service = _get_deepsearch_service()

    def _listener(record: Dict[str, Any], state) -> None:  # noqa: ANN001
        stage = getattr(state, "stage", "unknown")
        progress = compute_deepsearch_progress(
            stage,
            stage_history=getattr(state, "stage_history", None),
            stage_record=record,
        )
        payload = with_trace_protocol(
            {
                "run_id": run_id,
                "stage": stage,
                "stage_record": dict(record),
                "stage_history": list(getattr(state, "stage_history", []) or []),
                "errors": list(getattr(state, "errors", []) or []),
                "progress": progress,
            },
            run_id=run_id,
        )
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
        async def _run_with_trace():  # noqa: ANN202
            emitter = make_redis_trace_emitter(run_id=run_id, task_queue=task_queue, resource_id=run_id)
            token = set_trace_emitter(emitter)
            try:
                await emit_trace(
                    "think",
                    f"Received question. Starting graph-first DeepSearch run.\nrun_id={run_id}",
                    meta={"run_id": run_id, "external_allowed": False},
                )
                try:
                    await emit_trace(
                        "all_tools",
                        render_all_tools_block(
                            include_llm_tools=bool(getattr(service.planner, "include_llm_tools_in_catalog")),
                            registry=getattr(service.planner, "_tool_hint_registry", None),
                        ),
                        meta={"run_id": run_id},
                    )
                except Exception:
                    pass
                raw_result = await service.run(
                    question,
                    owner_id=str(owner_uuid),
                    metadata=metadata,
                    run_id=run_id,
                    stage_listener=_listener,
                )
                graph_store = _get_graph_store()
                trimmed = trim_deepsearch_payload(
                    raw_result,
                    include_evidence=include_evidence,
                    graph_store=graph_store,
                )
                report_block = trimmed.get("report") if isinstance(trimmed, dict) else None
                report_text = report_block.get("answer") if isinstance(report_block, dict) else None
                if isinstance(report_text, str) and report_text.strip():
                    await emit_trace(
                        "write",
                        report_text,
                        meta={"run_id": run_id, "question": trimmed.get("question")},
                    )
                await emit_trace(
                    "terminate",
                    f"DeepSearch completed.\nrun_id={run_id}",
                    meta={"run_id": run_id, "ok": True},
                )
                return trimmed
            except Exception as exc:  # noqa: BLE001
                await emit_trace(
                    "terminate",
                    f"DeepSearch failed.\nrun_id={run_id}\nerror={exc}",
                    meta={"run_id": run_id, "ok": False},
                )
                raise
            finally:
                reset_trace_emitter(token)

        result = asyncio.run(_run_with_trace())
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        max_retries = int(DEFAULT_CELERY_TASK_MAX_RETRIES)
        countdown = int(DEFAULT_CELERY_TASK_RETRY_COUNTDOWN_SECONDS)
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            task_queue.append_progress_event(
                flow="deepsearch",
                task_run_id=run_id,
                stage=str(getattr(getattr(self, "request", None), "retries", 0) or 0),
                status="retry",
                percent=0,
                resource_id=run_id,
                payload=with_trace_protocol(
                    {
                        "run_id": run_id,
                        "stage": "retry",
                        "error": err,
                        "retry_in_seconds": countdown,
                    },
                    run_id=run_id,
                ),
            )
            task_queue.update_task_run(
                run_id,
                state=TaskState.PENDING,
                progress_percent=0,
                error_message=f"retrying: {err}",
                metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
            )
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        logger.exception("DeepSearch failed (run_id=%s): %s", run_id, err)
        task_queue.append_progress_event(
            flow="deepsearch",
            task_run_id=run_id,
            stage="failed",
            status="error",
            percent=100,
            resource_id=run_id,
            payload=with_trace_protocol(
                {
                    "run_id": run_id,
                    "stage": "failed",
                    "errors": [{"message": err}],
                    "progress": _stage_progress("failed"),
                },
                run_id=run_id,
            ),
        )
        task_queue.update_task_run(
            run_id,
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        return {"run_id": run_id, "done": True, "error": err}

    task_queue.set_task_result_and_finalize_run(
        run_id,
        result=result if isinstance(result, dict) else {"result": result},
        state=TaskState.SUCCESS,
        progress_percent=100,
        finished=True,
    )
    task_queue.append_progress_event(
        flow="deepsearch",
        task_run_id=run_id,
        stage=str(result.get("state", {}).get("stage") or "reported") if isinstance(result, dict) else "reported",
        status="result",
        percent=100,
        resource_id=run_id,
        payload=with_trace_protocol(
            {
                "run_id": run_id,
                "stage": (result.get("state", {}).get("stage") if isinstance(result, dict) else None),
                "progress": _stage_progress((result.get("state", {}).get("stage", "") if isinstance(result, dict) else "")),
            },
            run_id=run_id,
        ),
    )
    return {"run_id": run_id, "done": True}
