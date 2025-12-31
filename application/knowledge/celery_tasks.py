import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, List

from celery.exceptions import Retry
from fastapi import HTTPException

from encapsulation.database.cache_db.redis_db import RedisDB
from encapsulation.message_queue.celery_app import app as celery_app
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.data_model.orm_models import FileStatus

from application.celery_bootstrap import ensure_initialized
from application.knowledge.graph_export import export_full_graph_payload
from application.knowledge.mindmap_export import export_file_mindmap_payload
from config.output_limits import KNOWLEDGE_GRAPH_EXPORT_MAX_NODES, KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES

logger = logging.getLogger(__name__)


def _run_coroutine(coro):  # noqa: ANN001
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def _acquire_lock(client, key: str, token: str, ttl_seconds: int) -> bool:
    try:
        return bool(client.set(key, token, nx=True, ex=int(ttl_seconds)))
    except Exception:
        return False


_RELEASE_LOCK_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
"""


def _release_lock(client, key: str, token: str) -> None:
    try:
        client.eval(_RELEASE_LOCK_LUA, 1, key, token)
    except Exception:
        return


def _file_lock_key(*, namespace: str, file_id: str) -> str:
    file_id = (file_id or "").strip()
    return f"{namespace}:lock:file:{file_id}"


def _parse_uuid(value: str) -> uuid.UUID:
    value = (value or "").strip()
    if len(value) == 32:
        return uuid.UUID(hex=value)
    return uuid.UUID(value)


def _get_knowledge():
    import app_registration

    return app_registration.registrator.get_object("knowledge")


def _get_rag_inference():
    import app_registration

    return app_registration.registrator.get_object("rag_inference")


@celery_app.task(bind=True, name="rag_arc.knowledge.index_file")
def index_file(self, *, file_id: str, owner_id: str) -> Dict[str, Any]:
    ensure_initialized()

    run_id = str(getattr(self.request, "id", "") or uuid.uuid4().hex)
    task_queue = RedisTaskQueue.from_env()
    owner_uuid = _parse_uuid(owner_id)

    if not task_queue.get_task_run(run_id):
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="index_file",
            owner_id=owner_uuid,
            resource_id=file_id,
            metadata={"executor": "celery"},
        )

    lock_ttl = int(os.getenv("FILE_OP_LOCK_TTL_SECONDS", str(6 * 3600)))
    lock_key = _file_lock_key(namespace=task_queue.settings.namespace, file_id=file_id)
    redis_client = RedisDB(RedisConfig()).client
    if not _acquire_lock(redis_client, lock_key, run_id, lock_ttl):
        max_retries = int(os.getenv("CELERY_TASK_LOCK_MAX_RETRIES", "30"))
        countdown = int(os.getenv("CELERY_TASK_LOCK_RETRY_COUNTDOWN_SECONDS", "2"))
        try:
            task_queue.update_task_run(
                run_id,
                state=TaskState.PENDING,
                progress_percent=0,
                error_message="waiting for file lock",
                metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
            )
        except Exception:
            pass
        try:
            raise self.retry(exc=RuntimeError("file op lock busy"), countdown=countdown, max_retries=max_retries)
        except Exception as exc:  # noqa: BLE001
            result_payload = {"success": False, "file_id": file_id, "error_message": "failed to acquire file lock"}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=f"failed to acquire file lock: {exc}",
                finished=True,
            )
            return result_payload

    try:
        knowledge = _get_knowledge()
        metadata = knowledge.file_storage.get_file_metadata(file_id)
        if not metadata:
            result_payload = {"success": False, "file_id": file_id, "error_message": "file not found"}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="file not found",
                finished=True,
            )
            return result_payload

        if getattr(metadata, "status", None) == FileStatus.DELETED:
            result_payload = {"success": False, "file_id": file_id, "error_message": "file deleted", "canceled": True}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.CANCELED,
                progress_percent=100,
                error_message="file deleted",
                finished=True,
            )
            return result_payload

        if getattr(metadata, "owner_id", None) != owner_uuid:
            result_payload = {"success": False, "file_id": file_id, "error_message": "owner mismatch"}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="owner mismatch",
                finished=True,
            )
            return result_payload

        # Dependency preflight (fail fast with actionable diagnosis).
        try:
            from core.utils.dependency_health import check_dependencies, format_dependency_failures

            health = check_dependencies(
                mode_env="RAGARC_INDEXING_DEPENDENCY_CHECK_MODE",
                default_mode="strict",
            )
            failure = format_dependency_failures(health)
            if failure:
                raise RuntimeError(failure)
        except Exception as exc:  # noqa: BLE001
            err = f"dependency health check failed: {exc}"
            logger.error(err)
            result_payload = {"success": False, "file_id": file_id, "error_message": err}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=err,
                finished=True,
            )
            task_queue.append_progress_event(
                flow="indexing",
                task_run_id=run_id,
                stage="dependency_check",
                status="error",
                percent=100,
                resource_id=file_id,
                payload={"file_id": file_id, "success": False, "error_message": err},
            )
            return result_payload

        # Ensure idempotency under at-least-once semantics by cleaning previous derived artifacts/index entries.
        cleanup = knowledge.file_index.delete_file_data(file_id)
        if not cleanup.get("success", False):
            err = str(cleanup.get("error_message") or "pre-index cleanup failed")
            result_payload = {"success": False, "file_id": file_id, "error_message": err}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=err,
                finished=True,
            )
            task_queue.append_progress_event(
                flow="indexing",
                task_run_id=run_id,
                stage="cleanup",
                status="error",
                percent=100,
                resource_id=file_id,
                payload={"file_id": file_id, "success": False, "error_message": err},
            )
            return result_payload

        task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
        task_queue.append_progress_event(
            flow="indexing",
            task_run_id=run_id,
            stage="index",
            status="start",
            percent=1,
            resource_id=file_id,
            payload={"file_id": file_id},
        )

        def _progress(stage: str, percent: int | None, payload: dict[str, Any] | None = None) -> None:
            try:
                merged = {"file_id": file_id}
                if payload and isinstance(payload, dict):
                    merged.update(payload)
                task_queue.append_progress_event(
                    flow="indexing",
                    task_run_id=run_id,
                    stage=str(stage),
                    status="progress",
                    percent=percent,
                    resource_id=file_id,
                    payload=merged,
                )
                if percent is not None:
                    task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=int(percent))
            except Exception:
                return

        result = _run_coroutine(knowledge.file_index.index_file(file_id, progress=_progress))
        if result.get("success"):
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result if isinstance(result, dict) else {"result": result},
                state=TaskState.SUCCESS,
                progress_percent=100,
                finished=True,
            )
            task_queue.append_progress_event(
                flow="indexing",
                task_run_id=run_id,
                stage="index",
                status="end",
                percent=100,
                resource_id=file_id,
                payload={"file_id": file_id, "success": True},
            )
            return result

        err = str(result.get("error_message") or "indexing failed")
        result_payload = {"success": False, "file_id": file_id, "error_message": err}
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result_payload,
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="indexing",
            task_run_id=run_id,
            stage="index",
            status="error",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "success": False, "error_message": err},
        )
        return result_payload
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        max_retries = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        countdown = int(os.getenv("CELERY_TASK_RETRY_COUNTDOWN_SECONDS", "5"))
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            try:
                task_queue.update_task_run(
                    run_id,
                    state=TaskState.PENDING,
                    progress_percent=0,
                    error_message=f"retrying: {err}",
                    metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
                )
                task_queue.append_progress_event(
                    flow="indexing",
                    task_run_id=run_id,
                    stage="retry",
                    status="retry",
                    percent=0,
                    resource_id=file_id,
                    payload={"file_id": file_id, "error_message": err, "retry_in_seconds": countdown},
                )
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        logger.exception("Indexing failed (file_id=%s run_id=%s): %s", file_id, run_id, err)
        result_payload = {"success": False, "file_id": file_id, "error_message": err}
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result_payload,
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="indexing",
            task_run_id=run_id,
            stage="index",
            status="error",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "success": False, "error_message": err},
        )
        return result_payload
    finally:
        _release_lock(redis_client, lock_key, run_id)


@celery_app.task(bind=True, name="rag_arc.knowledge.delete_file")
def delete_file(self, *, file_id: str, owner_id: str, delete_file_metadata: bool = True) -> Dict[str, Any]:
    ensure_initialized()

    run_id = str(getattr(self.request, "id", "") or uuid.uuid4().hex)
    task_queue = RedisTaskQueue.from_env()
    owner_uuid = _parse_uuid(owner_id)

    if not task_queue.get_task_run(run_id):
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="delete_file",
            owner_id=owner_uuid,
            resource_id=file_id,
            metadata={"executor": "celery"},
        )

    lock_ttl = int(os.getenv("FILE_OP_LOCK_TTL_SECONDS", str(6 * 3600)))
    lock_key = _file_lock_key(namespace=task_queue.settings.namespace, file_id=file_id)
    redis_client = RedisDB(RedisConfig()).client
    if not _acquire_lock(redis_client, lock_key, run_id, lock_ttl):
        max_retries = int(os.getenv("CELERY_TASK_LOCK_MAX_RETRIES", "30"))
        countdown = int(os.getenv("CELERY_TASK_LOCK_RETRY_COUNTDOWN_SECONDS", "2"))
        try:
            task_queue.update_task_run(
                run_id,
                state=TaskState.PENDING,
                progress_percent=0,
                error_message="waiting for file lock",
                metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
            )
        except Exception:
            pass
        try:
            raise self.retry(exc=RuntimeError("file op lock busy"), countdown=countdown, max_retries=max_retries)
        except Exception as exc:  # noqa: BLE001
            result_payload = {"success": False, "file_id": file_id, "error_message": "failed to acquire file lock"}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=f"failed to acquire file lock: {exc}",
                finished=True,
            )
            return result_payload

    try:
        knowledge = _get_knowledge()
        metadata = knowledge.file_storage.get_file_metadata(file_id)
        if not metadata:
            result_payload = {"success": True, "file_id": file_id, "deleted": False}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.SUCCESS,
                progress_percent=100,
                finished=True,
            )
            return result_payload

        if getattr(metadata, "owner_id", None) != owner_uuid:
            result_payload = {"success": False, "file_id": file_id, "error_message": "owner mismatch"}
            task_queue.set_task_result_and_finalize_run(
                run_id,
                result=result_payload,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="owner mismatch",
                finished=True,
            )
            return result_payload

        task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
        task_queue.append_progress_event(
            flow="deletion",
            task_run_id=run_id,
            stage="delete",
            status="start",
            percent=1,
            resource_id=file_id,
            payload={"file_id": file_id},
        )

        deletion_result = knowledge.file_index.delete_file_data(file_id, delete_file_metadata=delete_file_metadata)
        if not deletion_result.get("success", False):
            error_msg = str(deletion_result.get("error_message") or "delete_file_data failed")
            if error_msg and "file_id must be a non-empty string" not in error_msg:
                raise RuntimeError(error_msg)

        storage_deleted = knowledge.file_storage.delete_file(file_id)
        if not storage_deleted:
            raise RuntimeError("file storage deletion returned False")

        result_payload = {"success": True, "file_id": file_id, "deleted": True}
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result_payload,
            state=TaskState.SUCCESS,
            progress_percent=100,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="deletion",
            task_run_id=run_id,
            stage="delete",
            status="end",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "success": True},
        )
        return result_payload
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        max_retries = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        countdown = int(os.getenv("CELERY_TASK_RETRY_COUNTDOWN_SECONDS", "5"))
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            try:
                task_queue.update_task_run(
                    run_id,
                    state=TaskState.PENDING,
                    progress_percent=0,
                    error_message=f"retrying: {err}",
                    metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
                )
                task_queue.append_progress_event(
                    flow="deletion",
                    task_run_id=run_id,
                    stage="retry",
                    status="retry",
                    percent=0,
                    resource_id=file_id,
                    payload={"file_id": file_id, "error_message": err, "retry_in_seconds": countdown},
                )
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        logger.exception("Deletion failed (file_id=%s run_id=%s): %s", file_id, run_id, err)
        result_payload = {"success": False, "file_id": file_id, "error_message": err}
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result_payload,
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="deletion",
            task_run_id=run_id,
            stage="delete",
            status="error",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "success": False, "error_message": err},
        )
        return result_payload
    finally:
        _release_lock(redis_client, lock_key, run_id)


@celery_app.task(bind=True, name="rag_arc.knowledge.export_graph")
def export_graph(
    self,
    *,
    owner_id: str,
    max_nodes: int = 500,
    max_edges: int = 2000,
    include_node_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    ensure_initialized()

    run_id = str(getattr(self.request, "id", "") or uuid.uuid4().hex)
    task_queue = RedisTaskQueue.from_env()
    owner_uuid = _parse_uuid(owner_id)
    max_nodes = max(1, min(int(max_nodes), int(KNOWLEDGE_GRAPH_EXPORT_MAX_NODES)))
    max_edges = max(0, min(int(max_edges), int(KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES)))

    if not task_queue.get_task_run(run_id):
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="graph_export",
            owner_id=owner_uuid,
            resource_id=str(owner_uuid),
            metadata={
                "executor": "celery",
                "max_nodes": int(max_nodes),
                "max_edges": int(max_edges),
                "include_node_types": include_node_types or [],
            },
        )

    existing = task_queue.get_task_result(run_id)
    if existing is not None:
        task_queue.append_progress_event(
            flow="export",
            task_run_id=run_id,
            stage="graph_export",
            status="dedup",
            percent=100,
            resource_id=str(owner_uuid),
            payload={"owner_id": str(owner_uuid), "dedup": True},
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

    try:
        task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
        task_queue.append_progress_event(
            flow="export",
            task_run_id=run_id,
            stage="graph_export",
            status="start",
            percent=1,
            resource_id=str(owner_uuid),
            payload={"owner_id": str(owner_uuid), "max_nodes": max_nodes, "max_edges": max_edges},
        )

        rag_inference = _get_rag_inference()
        result = export_full_graph_payload(
            rag_inference=rag_inference,
            owner_scope=str(owner_uuid),
            max_nodes=int(max_nodes),
            max_edges=int(max_edges),
            include_node_types=include_node_types,
        )

        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result if isinstance(result, dict) else {"result": result},
            state=TaskState.SUCCESS,
            progress_percent=100,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="export",
            task_run_id=run_id,
            stage="graph_export",
            status="result",
            percent=100,
            resource_id=str(owner_uuid),
            payload={"owner_id": str(owner_uuid)},
        )
        return {"run_id": run_id, "done": True}
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(getattr(exc, "detail", None) or exc)
        max_retries = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        countdown = int(os.getenv("CELERY_TASK_RETRY_COUNTDOWN_SECONDS", "5"))
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            try:
                task_queue.update_task_run(
                    run_id,
                    state=TaskState.PENDING,
                    progress_percent=0,
                    error_message=f"retrying: {err}",
                    metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
                )
                task_queue.append_progress_event(
                    flow="export",
                    task_run_id=run_id,
                    stage="graph_export",
                    status="retry",
                    percent=0,
                    resource_id=str(owner_uuid),
                    payload={"owner_id": str(owner_uuid), "error_message": err, "retry_in_seconds": countdown},
                )
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        if isinstance(exc, HTTPException):
            err = str(exc.detail)
        logger.exception("Graph export failed (owner_id=%s run_id=%s): %s", owner_id, run_id, err)
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result={"success": False, "error_message": err},
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        task_queue.append_progress_event(
            flow="export",
            task_run_id=run_id,
            stage="graph_export",
            status="error",
            percent=100,
            resource_id=str(owner_uuid),
            payload={"owner_id": str(owner_uuid), "error_message": err},
        )
        return {"run_id": run_id, "done": True, "error": err}


@celery_app.task(bind=True, name="rag_arc.knowledge.export_mindmap")
def export_mindmap(self, *, file_id: str, owner_id: str) -> Dict[str, Any]:
    ensure_initialized()

    run_id = str(getattr(self.request, "id", "") or uuid.uuid4().hex)
    task_queue = RedisTaskQueue.from_env()
    owner_uuid = _parse_uuid(owner_id)
    file_id = (file_id or "").strip()

    if not task_queue.get_task_run(run_id):
        task_queue.create_task_run(
            task_run_id=run_id,
            task_type="mindmap_export",
            owner_id=owner_uuid,
            resource_id=file_id,
            metadata={"executor": "celery"},
        )

    existing = task_queue.get_task_result(run_id)
    if existing is not None:
        task_queue.append_progress_event(
            flow="export",
            task_run_id=run_id,
            stage="mindmap_export",
            status="dedup",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "dedup": True},
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

    def _progress(stage: str, status: str, percent: int | None, payload: dict[str, Any] | None) -> None:
        try:
            task_queue.append_progress_event(
                flow="export",
                task_run_id=run_id,
                stage=str(stage),
                status=str(status),
                percent=percent,
                resource_id=file_id,
                payload=payload or {"file_id": file_id},
            )
            if percent is not None:
                task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=int(percent))
        except Exception:
            return

    task_queue.update_task_run(run_id, state=TaskState.RUNNING, progress_percent=1)
    _progress("mindmap_export", "start", 1, {"file_id": file_id})

    try:
        knowledge = _get_knowledge()
        rag_inference = _get_rag_inference()
        result = _run_coroutine(
            export_file_mindmap_payload(
                knowledge=knowledge,
                rag_inference=rag_inference,
                file_id=file_id,
                owner_id=owner_uuid,
                progress=_progress,
            )
        )
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result=result if isinstance(result, dict) else {"result": result},
            state=TaskState.SUCCESS,
            progress_percent=100,
            finished=True,
        )
        _progress("mindmap_export", "result", 100, {"file_id": file_id})
        return {"run_id": run_id, "done": True}
    except Retry:
        raise
    except Exception as exc:  # noqa: BLE001
        err = str(getattr(exc, "detail", None) or exc)
        max_retries = int(os.getenv("CELERY_TASK_MAX_RETRIES", "3"))
        countdown = int(os.getenv("CELERY_TASK_RETRY_COUNTDOWN_SECONDS", "5"))
        if int(getattr(self.request, "retries", 0) or 0) < max_retries:
            try:
                task_queue.update_task_run(
                    run_id,
                    state=TaskState.PENDING,
                    progress_percent=0,
                    error_message=f"retrying: {err}",
                    metadata_patch={"retries": int(getattr(self.request, "retries", 0) or 0)},
                )
                _progress("mindmap_export", "retry", 0, {"file_id": file_id, "error_message": err, "retry_in_seconds": countdown})
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)

        if isinstance(exc, HTTPException):
            err = str(exc.detail)
        logger.exception("Mindmap export failed (file_id=%s run_id=%s): %s", file_id, run_id, err)
        task_queue.set_task_result_and_finalize_run(
            run_id,
            result={"success": False, "error_message": err},
            state=TaskState.FAILURE,
            progress_percent=100,
            error_message=err,
            finished=True,
        )
        _progress("mindmap_export", "error", 100, {"file_id": file_id, "error_message": err})
        return {"run_id": run_id, "done": True, "error": err}
