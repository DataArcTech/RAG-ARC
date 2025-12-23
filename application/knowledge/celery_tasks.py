import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

from celery.exceptions import Retry

from encapsulation.database.cache_db.redis_db import RedisDB
from encapsulation.message_queue.celery_app import app as celery_app
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.data_model.orm_models import FileStatus

from application.celery_bootstrap import ensure_initialized

logger = logging.getLogger(__name__)


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
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=f"failed to acquire file lock: {exc}",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "failed to acquire file lock"}

    try:
        knowledge = _get_knowledge()
        metadata = knowledge.file_storage.get_file_metadata(file_id)
        if not metadata:
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="file not found",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "file not found"}

        if getattr(metadata, "status", None) == FileStatus.DELETED:
            task_queue.update_task_run(
                run_id,
                state=TaskState.CANCELED,
                error_message="file deleted",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "file deleted"}

        if getattr(metadata, "owner_id", None) != owner_uuid:
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="owner mismatch",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "owner mismatch"}

        # Ensure idempotency under at-least-once semantics by cleaning previous derived artifacts/index entries.
        cleanup = knowledge.file_index.delete_file_data(file_id)
        if not cleanup.get("success", False):
            err = str(cleanup.get("error_message") or "pre-index cleanup failed")
            task_queue.update_task_run(
                run_id,
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
            return {"success": False, "file_id": file_id, "error_message": err}

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

        result = asyncio.run(knowledge.file_index.index_file(file_id))
        if result.get("success"):
            task_queue.update_task_run(run_id, state=TaskState.SUCCESS, progress_percent=100, finished=True)
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
        task_queue.update_task_run(
            run_id,
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
        return {"success": False, "file_id": file_id, "error_message": err}
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
        task_queue.update_task_run(
            run_id,
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
        return {"success": False, "file_id": file_id, "error_message": err}
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
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message=f"failed to acquire file lock: {exc}",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "failed to acquire file lock"}

    try:
        knowledge = _get_knowledge()
        metadata = knowledge.file_storage.get_file_metadata(file_id)
        if not metadata:
            task_queue.update_task_run(run_id, state=TaskState.SUCCESS, progress_percent=100, finished=True)
            return {"success": True, "file_id": file_id, "deleted": False}

        if getattr(metadata, "owner_id", None) != owner_uuid:
            task_queue.update_task_run(
                run_id,
                state=TaskState.FAILURE,
                progress_percent=100,
                error_message="owner mismatch",
                finished=True,
            )
            return {"success": False, "file_id": file_id, "error_message": "owner mismatch"}

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

        task_queue.update_task_run(run_id, state=TaskState.SUCCESS, progress_percent=100, finished=True)
        task_queue.append_progress_event(
            flow="deletion",
            task_run_id=run_id,
            stage="delete",
            status="end",
            percent=100,
            resource_id=file_id,
            payload={"file_id": file_id, "success": True},
        )
        return {"success": True, "file_id": file_id, "deleted": True}
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
        task_queue.update_task_run(
            run_id,
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
        return {"success": False, "file_id": file_id, "error_message": err}
    finally:
        _release_lock(redis_client, lock_key, run_id)
