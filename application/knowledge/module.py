from framework.module import AbstractModule
import logging
from typing import TYPE_CHECKING, List, Optional, Dict, Any

if TYPE_CHECKING:
    from config.application.knowledge_config import KnowledgeConfig

logger = logging.getLogger(__name__)

import uuid
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from fastapi.responses import Response
from fastapi import UploadFile, HTTPException
from encapsulation.data_model.orm_models import (
    FileMetadata, FileStatus,
    FilePermission, PermissionReceiverType, PermissionType
)
from core.utils.thread_pool import run_blocking, run_coroutine_in_thread
from core.utils.http_headers import build_attachment_content_disposition
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from application.knowledge.permission_mixin import KnowledgePermissionMixin
from config.output_limits import KNOWLEDGE_MINDMAP_EXPORT_MAX_CHUNKS
from application.knowledge.runtime_state_mixin import KnowledgeRuntimeStateMixin
from application.rabc.visibility import build_owner_visibility
from core.utils.owner_guard import get_share_owner_id, is_admin_owner, is_org_admin_owner, normalize_owner_id

class Knowledge(KnowledgePermissionMixin, KnowledgeRuntimeStateMixin, AbstractModule):
    def __init__(self, config: 'KnowledgeConfig'):
        super().__init__(config=config)
        self.logger = logger
        self.file_storage = config.file_storage_config.build()
        self.file_index = config.index_manager_config.build()
        self.task_queue = RedisTaskQueue.from_env()
        
        # Semaphore to control concurrent indexing operations
        self.indexing_semaphore = asyncio.Semaphore(config.max_concurrent_indexing)
        self._active_index_tasks: Dict[str, asyncio.Task] = {}
        self._files_marked_for_deletion: set[str] = set()
        self._files_marked_for_deletion_by_owner: Dict[uuid.UUID, set[str]] = {}
        self._file_owner_cache: Dict[str, uuid.UUID] = {}
        self._active_deletion_tasks: Dict[str, asyncio.Task] = {}
        self._deletion_failures: Dict[str, str] = {}

    def _use_celery(self) -> bool:
        return os.getenv("TASK_QUEUE_MODE", "inprocess").lower() == "celery"

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a separate thread to avoid blocking the event loop."""
        return await run_blocking(func, *args, **kwargs)

    async def _run_coroutine_in_thread(self, coro_func, *args, **kwargs):  # noqa: ANN001
        """Run an async callable in a dedicated thread to keep FastAPI event loop responsive."""
        return await run_coroutine_in_thread(coro_func, *args, **kwargs)
    
    def _track_deletion_task(self, doc_id: str, task: asyncio.Task) -> None:
        """Register a background deletion task so we don't schedule duplicates."""
        self._active_deletion_tasks[doc_id] = task

        def _cleanup(fut: asyncio.Task, file_id: str = doc_id) -> None:
            self._active_deletion_tasks.pop(file_id, None)
            if fut.cancelled():
                logger.info(f"Background deletion task cancelled for file_id: {file_id}")
            elif fut.exception():
                logger.error(
                    f"Background deletion task failed for file_id {file_id}: {fut.exception()}",
                    exc_info=True
                )

        task.add_done_callback(_cleanup)

    async def _cancel_deletion_task(self, doc_id: str) -> None:
        task = self._active_deletion_tasks.get(doc_id)
        if not task:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Cancelled deletion task awaited for file_id: {doc_id}")
    
    async def upload_file(self, file: UploadFile, user_id: uuid.UUID, *, relative_path: str | None = None) -> str:
        return await self.upload_file_scoped(
            file=file,
            actor_id=user_id,
            owner_id=user_id,
            allow_non_owner=True,
            relative_path=relative_path,
        )

    async def upload_file_scoped(
        self,
        *,
        file: UploadFile,
        actor_id: uuid.UUID,
        owner_id: uuid.UUID,
        allow_non_owner: bool = False,
        relative_path: str | None = None,
    ) -> str:
        """
        Upload a file into an explicit owner scope.

        This is the algorithm-layer entry point for supporting share owner uploads
        (e.g., org_admin → share_owner_id). API-layer RBAC should set `allow_non_owner=True`
        only when the caller is authorized.
        """
        if self._is_share_owner_scope(owner_id) and not self._is_share_library_admin(actor_id):
            raise HTTPException(status_code=403, detail="Only admins can manage shared library files")

        if actor_id != owner_id and not allow_non_owner:
            raise HTTPException(status_code=403, detail="You are not allowed to upload files into this scope")
        try:
            file_data = await self._run_blocking(file.file.read)
            doc_id = await self._run_blocking(
                self.file_storage.upload_file,
                filename=(relative_path or file.filename),
                file_data=file_data,
                owner_id=owner_id,
                content_type=file.content_type,
            )

            task_run_id = self.task_queue.create_task_run(
                task_type="index_file",
                owner_id=owner_id,
                resource_id=doc_id,
                metadata={"filename": relative_path or file.filename, "content_type": file.content_type, "actor_id": str(actor_id)},
            )
            if self._use_celery():
                from application.knowledge.celery_tasks import index_file as index_file_task

                queue = os.getenv("CELERY_QUEUE_INDEXING", "indexing")
                index_file_task.apply_async(
                    kwargs={"file_id": doc_id, "owner_id": str(owner_id)},
                    task_id=task_run_id,
                    queue=queue,
                )
                logger.info(
                    "File %s uploaded with ID %s, indexing enqueued (task_run_id=%s owner_id=%s)",
                    file.filename,
                    doc_id,
                    task_run_id,
                    owner_id,
                )
            else:
                task = asyncio.create_task(self._index_file_background(doc_id, task_run_id=task_run_id))
                self._track_background_task(doc_id, task)
                logger.info(
                    "File %s uploaded with ID %s, indexing started in background (owner_id=%s)",
                    file.filename,
                    doc_id,
                    owner_id,
                )
            return doc_id

        except Exception as e:
            logger.error(e)
            raise

    async def _index_file_background(self, doc_id: str, *, task_run_id: str | None = None) -> Dict[str, Any]:
        """Background task for indexing files with semaphore control
        
        Returns:
            Dict with indexing result containing 'success' (bool) and 'file_id' (str) keys
        """
        if self._is_file_marked_for_deletion(doc_id):
            logger.info(f"Skipping indexing for file_id {doc_id} because it is marked for deletion")
            if task_run_id:
                self.task_queue.update_task_run(
                    task_run_id,
                    state=TaskState.CANCELED,
                    error_message="file scheduled for deletion",
                    finished=True,
                )
            return {"success": False, "file_id": doc_id, "error_message": "file scheduled for deletion"}

        async with self.indexing_semaphore:
            try:
                if self._is_file_marked_for_deletion(doc_id):
                    logger.info(f"Aborting indexing for file_id {doc_id}; deletion scheduled")
                    if task_run_id:
                        self.task_queue.update_task_run(
                            task_run_id,
                            state=TaskState.CANCELED,
                            error_message="file scheduled for deletion",
                            finished=True,
                        )
                    return {"success": False, "file_id": doc_id, "error_message": "file scheduled for deletion"}

                logger.info(f"Starting background indexing for file_id: {doc_id} (semaphore acquired)")
                if task_run_id:
                    self.task_queue.update_task_run(task_run_id, state=TaskState.RUNNING, progress_percent=1)
                    self.task_queue.append_progress_event(
                        flow="indexing",
                        task_run_id=task_run_id,
                        stage="index",
                        status="start",
                        percent=1,
                        resource_id=doc_id,
                        payload={"file_id": doc_id},
                    )

                # Dependency preflight (fail fast with actionable diagnosis).
                try:
                    from core.utils.dependency_health import check_dependencies, format_dependency_failures

                    health = check_dependencies(
                        mode_env="RAGARC_INDEXING_DEPENDENCY_CHECK_MODE",
                        default_mode="strict",
                    )
                    failure = format_dependency_failures(health)
                    if failure and health.get("mode") == "strict":
                        raise RuntimeError(failure)
                except Exception as exc:
                    err = f"dependency health check failed: {exc}"
                    logger.error(err)
                    if task_run_id:
                        try:
                            self.task_queue.update_task_run(
                                task_run_id,
                                state=TaskState.FAILURE,
                                progress_percent=100,
                                error_message=err,
                                finished=True,
                            )
                            self.task_queue.append_progress_event(
                                flow="indexing",
                                task_run_id=task_run_id,
                                stage="dependency_check",
                                status="error",
                                percent=100,
                                resource_id=doc_id,
                                payload={"file_id": doc_id, "success": False, "error_message": err},
                            )
                        except Exception:
                            pass
                    return {"success": False, "file_id": doc_id, "error_message": err}

                # Idempotency under at-least-once semantics: remove old derived artifacts/index entries first.
                cleanup = await self._run_blocking(self.file_index.delete_file_data, doc_id)
                if not cleanup.get("success", False):
                    err = str(cleanup.get("error_message") or "pre-index cleanup failed")
                    logger.error("Pre-index cleanup failed for file_id=%s: %s", doc_id, err)
                    if task_run_id:
                        self.task_queue.update_task_run(
                            task_run_id,
                            state=TaskState.FAILURE,
                            progress_percent=100,
                            error_message=err,
                            finished=True,
                        )
                        self.task_queue.append_progress_event(
                            flow="indexing",
                            task_run_id=task_run_id,
                            stage="cleanup",
                            status="error",
                            percent=100,
                            resource_id=doc_id,
                            payload={"file_id": doc_id, "success": False, "error_message": err},
                        )
                    return {"success": False, "file_id": doc_id, "error_message": err}
                def _progress(stage: str, percent: int | None, payload: dict[str, Any] | None = None) -> None:
                    if not task_run_id:
                        return
                    try:
                        merged = {"file_id": doc_id}
                        if payload and isinstance(payload, dict):
                            merged.update(payload)
                        self.task_queue.append_progress_event(
                            flow="indexing",
                            task_run_id=task_run_id,
                            stage=str(stage),
                            status="progress",
                            percent=percent,
                            resource_id=doc_id,
                            payload=merged,
                        )
                        if percent is not None:
                            self.task_queue.update_task_run(task_run_id, state=TaskState.RUNNING, progress_percent=int(percent))
                    except Exception:
                        return

                # IndexManager is async but performs heavy blocking work; run it off the main event loop.
                result = await self._run_coroutine_in_thread(self.file_index.index_file, doc_id, progress=_progress)
                if result.get("success"):
                    logger.info(f"Background indexing completed successfully for file_id: {doc_id}")
                    if task_run_id:
                        self.task_queue.update_task_run(task_run_id, state=TaskState.SUCCESS, progress_percent=100, finished=True)
                        self.task_queue.append_progress_event(
                            flow="indexing",
                            task_run_id=task_run_id,
                            stage="index",
                            status="end",
                            percent=100,
                            resource_id=doc_id,
                            payload={"file_id": doc_id, "success": True},
                        )
                else:
                    logger.error(f"Background indexing failed for file_id: {doc_id}, error: {result.get('error_message')}")
                    if task_run_id:
                        err = str(result.get("error_message") or "indexing failed")
                        self.task_queue.update_task_run(
                            task_run_id,
                            state=TaskState.FAILURE,
                            progress_percent=100,
                            error_message=err,
                            finished=True,
                        )
                        self.task_queue.append_progress_event(
                            flow="indexing",
                            task_run_id=task_run_id,
                            stage="index",
                            status="error",
                            percent=100,
                            resource_id=doc_id,
                            payload={"file_id": doc_id, "success": False, "error_message": err},
                        )
                return result
            except asyncio.CancelledError:
                logger.info(f"Background indexing task cancelled for file_id: {doc_id}")
                if task_run_id:
                    self.task_queue.update_task_run(
                        task_run_id,
                        state=TaskState.CANCELED,
                        error_message="cancelled",
                        finished=True,
                    )
                raise
            except Exception as e:
                logger.error(f"Background indexing failed for file_id: {doc_id}, exception: {str(e)}")
                if task_run_id:
                    self.task_queue.update_task_run(
                        task_run_id,
                        state=TaskState.FAILURE,
                        progress_percent=100,
                        error_message=str(e),
                        finished=True,
                    )
                    self.task_queue.append_progress_event(
                        flow="indexing",
                        task_run_id=task_run_id,
                        stage="index",
                        status="error",
                        percent=100,
                        resource_id=doc_id,
                        payload={"file_id": doc_id, "success": False, "error_message": str(e)},
                    )
                return {"success": False, "file_id": doc_id, "error_message": str(e)}
            finally:
                logger.debug(f"Background indexing semaphore released for file_id: {doc_id}")

    async def get_file(self, doc_id: str, user_id: uuid.UUID | None = None) -> Response:
        # Run database queries asynchronously to avoid blocking the event loop
        metadata = await self._run_blocking(
            self.file_storage.get_file_metadata,
            doc_id
        )

        if metadata is None:
            logger.warning(f"File metadata not found for file_id: {doc_id}")
            raise HTTPException(status_code=404, detail="File not found")
        if metadata.status == FileStatus.DELETED or self._is_file_marked_for_deletion(doc_id):
            logger.warning(f"File is deleted or marked for deletion: file_id={doc_id}, status={metadata.status}")
            raise HTTPException(status_code=404, detail="File not found")

        content = await self._run_blocking(
            self.file_storage.get_file_content,
            doc_id
        )
        if content is None:
            raise HTTPException(status_code=404, detail="File content not found")

        headers = {"Content-Disposition": build_attachment_content_disposition(str(metadata.filename or ""))}
        return Response(content=content, media_type=metadata.content_type, headers=headers)

    async def mark_file_deleted_cli(self, doc_id: str, user_id: uuid.UUID) -> Dict[str, Any]:
        """Mark a file as deleted for CLI scenarios without triggering heavy cleanup."""
        metadata = await self._run_blocking(self.file_storage.get_file_metadata, doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        if metadata.owner_id != user_id:
            raise HTTPException(status_code=403, detail="You are not allowed to delete this file")

        failure_reason = self._deletion_failures.get(doc_id)
        if metadata.status == FileStatus.DELETED and not failure_reason:
            logger.info(f"[CLI] File {doc_id} already marked as deleted")
            return {"status": "deleted", "file_id": doc_id}

        await self._cancel_indexing_task(doc_id)
        await self._cancel_deletion_task(doc_id)

        if not self._is_file_marked_for_deletion(doc_id):
            self._mark_file_for_deletion(doc_id, metadata.owner_id)

        await self._run_blocking(
            self.file_storage.metadata_store.update_file_status,
            doc_id,
            FileStatus.DELETED
        )

        self._deletion_failures.pop(doc_id, None)
        logger.info(f"[CLI] Marked file {doc_id} as deleted (metadata only)")
        response = {"status": "marked", "file_id": doc_id}
        if failure_reason:
            response["previous_failure"] = failure_reason
        return response

    async def delete_file(self, doc_id: str, user_id: uuid.UUID) -> Dict[str, Any]:
        # Check if the file exists before attempting deletion
        metadata = await self._run_blocking(self.file_storage.get_file_metadata, doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        # Only the file owner can delete the file
        if metadata.owner_id != user_id:
            raise HTTPException(status_code=403, detail="You are not allowed to delete this file")

        failure_reason = self._deletion_failures.get(doc_id)

        if metadata.status == FileStatus.DELETED and not failure_reason:
            logger.info(f"File {doc_id} already deleted")
            return {"status": "deleted", "file_id": doc_id}

        if self._is_file_marked_for_deletion(doc_id) and not failure_reason:
            logger.info(f"Deletion already scheduled for {doc_id}")
            return {"status": "deleting", "file_id": doc_id}

        # Ensure background indexing/deletion is not running for this file
        await self._cancel_indexing_task(doc_id)
        await self._cancel_deletion_task(doc_id)

        self._mark_file_for_deletion(doc_id, metadata.owner_id)

        # Mark file as DELETED to hide immediately (physical cleanup happens in background)
        await self._run_blocking(
            self.file_storage.metadata_store.update_file_status,
            doc_id,
            FileStatus.DELETED
        )

        if self._use_celery():
            from application.knowledge.celery_tasks import delete_file as delete_file_task

            task_run_id = self.task_queue.create_task_run(
                task_type="delete_file",
                owner_id=user_id,
                resource_id=doc_id,
                metadata={"trigger": "api_delete"},
            )
            queue = os.getenv("CELERY_QUEUE_INDEXING", "indexing")
            delete_file_task.apply_async(
                kwargs={"file_id": doc_id, "owner_id": str(user_id), "delete_file_metadata": True},
                task_id=task_run_id,
                queue=queue,
            )
            logger.info("Deletion enqueued for file_id=%s (task_run_id=%s)", doc_id, task_run_id)
        else:
            # Schedule deletion in background
            delete_task = asyncio.create_task(self._delete_file_background(doc_id))
            self._track_deletion_task(doc_id, delete_task)
            logger.info(f"Deletion scheduled for file_id: {doc_id}")

        response = {"status": "deleting", "file_id": doc_id}
        if failure_reason:
            response["previous_failure"] = failure_reason
        return response

    async def delete_file_scoped(
        self,
        doc_id: str,
        *,
        actor_id: uuid.UUID,
        owner_id: uuid.UUID,
        allow_non_owner: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a file in an explicit owner scope.

        - When `allow_non_owner=False`, only the owner can delete (same as delete_file()).
        - When `allow_non_owner=True`, callers may delete on behalf of `owner_id` (e.g., share owner cleanup),
          but we still enforce that the file belongs to the requested owner scope.
        """
        if self._is_share_owner_scope(owner_id) and not self._is_share_library_admin(actor_id):
            raise HTTPException(status_code=403, detail="Only admins can manage shared library files")

        metadata = await self._run_blocking(self.file_storage.get_file_metadata, doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        if metadata.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="File not found")

        if metadata.owner_id != actor_id and not allow_non_owner:
            raise HTTPException(status_code=403, detail="You are not allowed to delete this file")

        failure_reason = self._deletion_failures.get(doc_id)

        if metadata.status == FileStatus.DELETED and not failure_reason:
            logger.info("File %s already deleted (scope owner_id=%s)", doc_id, owner_id)
            return {"status": "deleted", "file_id": doc_id}

        if self._is_file_marked_for_deletion(doc_id) and not failure_reason:
            logger.info("Deletion already scheduled for %s (scope owner_id=%s)", doc_id, owner_id)
            return {"status": "deleting", "file_id": doc_id}

        await self._cancel_indexing_task(doc_id)
        await self._cancel_deletion_task(doc_id)

        self._mark_file_for_deletion(doc_id, metadata.owner_id)
        await self._run_blocking(self.file_storage.metadata_store.update_file_status, doc_id, FileStatus.DELETED)

        if self._use_celery():
            from application.knowledge.celery_tasks import delete_file as delete_file_task

            task_run_id = self.task_queue.create_task_run(
                task_type="delete_file",
                owner_id=metadata.owner_id,
                resource_id=doc_id,
                metadata={"trigger": "scoped_delete", "actor_id": str(actor_id)},
            )
            queue = os.getenv("CELERY_QUEUE_INDEXING", "indexing")
            delete_file_task.apply_async(
                kwargs={"file_id": doc_id, "owner_id": str(metadata.owner_id), "delete_file_metadata": True},
                task_id=task_run_id,
                queue=queue,
            )
            logger.info("Deletion enqueued for file_id=%s (task_run_id=%s owner_id=%s)", doc_id, task_run_id, owner_id)
        else:
            delete_task = asyncio.create_task(self._delete_file_background(doc_id))
            self._track_deletion_task(doc_id, delete_task)
            logger.info("Deletion scheduled for file_id: %s (owner_id=%s)", doc_id, owner_id)

        response = {"status": "deleting", "file_id": doc_id}
        if failure_reason:
            response["previous_failure"] = failure_reason
        return response

    @staticmethod
    def _is_share_owner_scope(owner_id: uuid.UUID) -> bool:
        share = get_share_owner_id()
        if not share:
            return False
        return normalize_owner_id(owner_id) == share

    @staticmethod
    def _is_share_library_admin(actor_id: uuid.UUID) -> bool:
        return bool(is_admin_owner(actor_id) or is_org_admin_owner(actor_id))

    async def _delete_file_background(self, doc_id: str) -> None:
        """Execute the deletion pipeline asynchronously."""
        logger.info(f"Background deletion started for file_id: {doc_id}")
        success = False
        try:
            deletion_result = await self._run_blocking(
                self.file_index.delete_file_data,
                doc_id,
                delete_file_metadata=True
            )

            if not deletion_result.get("success", False):
                error_msg = deletion_result.get("error_message", "")
                if error_msg and "file_id must be a non-empty string" not in error_msg:
                    logger.error(f"Deletion pipeline failed for {doc_id}: {error_msg}")
                    raise RuntimeError(error_msg)
                logger.info(f"No indexed content found for file {doc_id}, continuing deletion workflow")

            storage_deleted = await self._run_blocking(self.file_storage.delete_file, doc_id)
            if not storage_deleted:
                raise RuntimeError("File storage deletion returned False")

            logger.info(f"Background deletion completed for file_id: {doc_id}")
            success = True
        except Exception as e:
            logger.error(f"Background deletion failed for {doc_id}: {e}")
            self._deletion_failures[doc_id] = str(e)
            try:
                await self._run_blocking(
                    self.file_storage.metadata_store.update_file_status,
                    doc_id,
                    FileStatus.DELETED
                )
            except Exception as status_error:
                logger.error(f"Failed to persist DELETED status after deletion failure for {doc_id}: {status_error}")
        finally:
            if success:
                self._deletion_failures.pop(doc_id, None)
                self._unmark_file_for_deletion(doc_id)

    def list_user_files(
        self,
        user_id: uuid.UUID,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        search: Optional[str] = None
    ) -> List[FileMetadata]:
        """
        Get all files accessible to a specific user (files with permissions only).
        
        Args:
            user_id: UUID of the user
            status: Optional filter by file status
            limit: Maximum number of files to return
            offset: Number of files to skip (for pagination)
            search: Optional search keyword for filename (fuzzy match)
            
        Returns:
            List of FileMetadata objects accessible to the user
        """
        try:
            # When status is None, we need to filter out deleted files in application layer.
            # To ensure correct pagination, we must filter first, then apply limit/offset.
            if status is None:
                # Get all files without pagination first
                all_files_from_db = self.file_storage.list_accessible_files(
                    user_id=user_id,
                    status=status,
                    limit=None,
                    offset=None,
                    search=search,
                )
                logger.info(f"Retrieved {len(all_files_from_db)} files from database for user {user_id} (before filtering, status={status})")
                
                # Filter out deleted files and files marked for deletion
                files = [
                    file for file in all_files_from_db
                    if self._is_active_status(file.status) and not self._is_file_marked_for_deletion(file.file_id)
                ]
                removed_count = len(all_files_from_db) - len(files)
                logger.info(f"After filtering: {len(files)} active files (removed {removed_count} deleted/marked files)")
                
                # Log status distribution for debugging
                if removed_count > 0:
                    status_dist = {}
                    for file in all_files_from_db:
                        status_val = file.status.value if hasattr(file.status, 'value') else str(file.status)
                        status_dist[status_val] = status_dist.get(status_val, 0) + 1
                    logger.info(f"File status distribution before filtering: {status_dist}")
                
                # Ensure files are sorted by created_at DESC (same as database order)
                # This is critical for consistent pagination
                files.sort(key=lambda f: f.created_at, reverse=True)
                
                # Apply pagination after filtering and sorting
                # offset: number of items to skip (0-based)
                # limit: maximum number of items to return
                offset_val = int(offset or 0)
                if offset_val < 0:
                    offset_val = 0
                
                limit_val = int(limit) if limit is not None else None
                
                # Calculate slice indices
                start_idx = offset_val
                if limit_val is None:
                    result = files[start_idx:]
                else:
                    end_idx = start_idx + limit_val
                    result = files[start_idx:end_idx]
                
                logger.info(
                    f"Pagination result: retrieved {len(result)} files for user {user_id} "
                    f"(offset={offset_val}, limit={limit_val}, total_filtered={len(files)}, "
                    f"requested_range=[{start_idx}:{start_idx + (limit_val or len(files))}], "
                    f"actual_slice=[{start_idx}:{min(start_idx + (limit_val or len(files)), len(files))}])"
                )
                
                # Additional validation logging
                if limit_val and len(result) < limit_val and start_idx + limit_val <= len(files):
                    logger.warning(
                        f"Pagination warning: requested {limit_val} files starting from offset {offset_val}, "
                        f"but only got {len(result)} files. Total filtered files: {len(files)}"
                    )
                
                return result
            else:
                # When status is specified, database already filters correctly, so we can use limit/offset directly
                files = self.file_storage.list_accessible_files(
                    user_id=user_id,
                    status=status,
                    limit=limit,
                    offset=offset,
                    search=search,
                )
                logger.info(f"Retrieved {len(files)} accessible files for user {user_id}")
                return files
        except Exception as e:
            logger.error(f"Failed to list accessible files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve files: {str(e)}")

    def list_visible_files(
        self,
        user_id: uuid.UUID,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[FileMetadata]:
        """
        List files visible under an explicit owner visibility scope.

        Notes:
        - This bypasses the fine-grained `file_permission` model and instead unions file listings by owner_id.
        - Intended for algorithm-layer "me / me+share" rollout; API-layer RBAC should decide when to call it.
        - Pagination is applied after union+sort (created_at desc).
        """
        visibility = build_owner_visibility(
            primary_owner_id=user_id,
            include_share=bool(include_share),
            share_owner_id=share_owner_id,
            label=("me+share" if include_share else "me"),
        )

        owner_uuids: list[uuid.UUID] = []
        for token in visibility.owner_ids:
            try:
                owner_uuids.append(uuid.UUID(str(token)))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Invalid owner_id in visibility scope: {token} ({exc})") from exc

        files: list[FileMetadata] = []
        try:
            for oid in owner_uuids:
                files.extend(self.file_storage.list_files(owner_id=oid, status=status, limit=None, offset=None))
        except Exception as e:
            logger.error("Failed to list files for visibility scope %s: %s", visibility.label, e)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve files: {str(e)}") from e

        if status is None:
            files = [
                file for file in files
                if self._is_active_status(file.status) and not self._is_file_marked_for_deletion(file.file_id)
            ]

        try:
            files.sort(key=lambda f: f.created_at, reverse=True)
        except Exception:
            pass

        start = int(offset or 0)
        if start < 0:
            start = 0
        if limit is None:
            return files[start:]
        return files[start : start + max(int(limit), 0)]

    async def list_user_files_async(
        self,
        user_id: uuid.UUID,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        search: Optional[str] = None,
    ) -> List[FileMetadata]:
        """Async wrapper for list_user_files() for FastAPI handlers."""
        return await self._run_blocking(
            self.list_user_files,
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset,
            search=search,
        )

    async def list_visible_files_async(
        self,
        user_id: uuid.UUID,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[FileMetadata]:
        return await self._run_blocking(
            self.list_visible_files,
            user_id=user_id,
            include_share=include_share,
            share_owner_id=share_owner_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def count_user_files(
        self,
        user_id: uuid.UUID,
        status: FileStatus | None = None,
        search: Optional[str] = None
    ) -> int:
        """
        Count all files accessible to a specific user (files with permissions).
        
        Args:
            user_id: UUID of the user
            status: Optional filter by file status
            search: Optional search keyword for filename (fuzzy match)
            
        Returns:
            Total count of files accessible to the user
        """
        try:
            if status is None:
                total = self.file_storage.count_accessible_files(user_id=user_id, search=search)
                deleted = self.file_storage.count_accessible_files(user_id=user_id, status=FileStatus.DELETED, search=search)
                mark_only = len(self._files_marked_for_deletion_by_owner.get(user_id, set()))
                count = max(total - deleted - mark_only, 0)
            else:
                count = self.file_storage.count_accessible_files(user_id=user_id, status=status, search=search)
            logger.info(f"Counted {count} accessible files for user {user_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to count accessible files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to count files: {str(e)}")

    def count_visible_files(
        self,
        user_id: uuid.UUID,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        status: FileStatus | None = None,
    ) -> int:
        files = self.list_visible_files(
            user_id,
            include_share=include_share,
            share_owner_id=share_owner_id,
            status=status,
            limit=None,
            offset=None,
        )
        return len(files)

    async def count_visible_files_async(
        self,
        user_id: uuid.UUID,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        status: FileStatus | None = None,
    ) -> int:
        return await self._run_blocking(
            self.count_visible_files,
            user_id,
            include_share=include_share,
            share_owner_id=share_owner_id,
            status=status,
        )

    async def count_user_files_async(
        self,
        user_id: uuid.UUID,
        status: FileStatus | None = None,
        search: Optional[str] = None,
    ) -> int:
        """Async wrapper for count_user_files() for FastAPI handlers."""
        return await self._run_blocking(self.count_user_files, user_id, status=status, search=search)

    def _is_active_status(self, status: FileStatus) -> bool:
        return status != FileStatus.DELETED

    def is_file_active(self, file_id: str) -> bool:
        """
        Determine if a file is still visible for retrieval/listing purposes.
        """
        from config.knowledge_visibility import KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS

        try:
            metadata = self.file_storage.get_file_metadata(file_id)
        except Exception as e:
            logger.debug(f"Failed to fetch metadata for file {file_id}: {e}")
            return False

        if not metadata:
            return False

        if not self._is_active_status(metadata.status):
            return False

        if self._is_file_marked_for_deletion(file_id):
            return False

        if KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS:
            # Defensive: hide "dangling" metadata rows that no longer have a retrievable blob.
            # This prevents returning citations that later fail in `/static/files/*` download flows.
            blob_key = getattr(metadata, "blob_key", None)
            if not blob_key:
                return False
            try:
                if not self.file_storage.blob_store.exists(str(blob_key)):
                    return False
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to check blob existence for file %s (key=%s): %s", file_id, blob_key, e)
                return False

        return True

    async def trigger_indexing(
        self,
        file_ids: List[str],
        user_id: uuid.UUID,
        *,
        force: bool = False,
        wait: bool = False,
    ) -> str:
        """
        Trigger indexing for multiple files asynchronously.
        
        Args:
            file_ids: List of file IDs to index
            user_id: UUID of the user requesting the indexing
            
        Returns:
            String containing basic info about the triggered indexing or error message
        """
        # Validate files and collect those eligible for indexing
        # Only allow indexing of STORED or FAILED files
        # Skip files that are INDEXED, or in intermediate states (PARSED, CHUNKED) indicating processing is in progress
        valid_files = []
        invalid_files = []
        skipped_files = []

        for file_id in file_ids:
            try:
                metadata = self.file_storage.get_file_metadata(file_id)
                if not metadata:
                    invalid_files.append(f"File not found or invalid: {file_id}")
                    continue
                if metadata.owner_id != user_id:
                    invalid_files.append(f"You are not authorized to operate on this file: {file_id}")
                    continue

                if self._use_celery():
                    # Cross-process de-duplication: skip when an active TaskRun exists.
                    latest_run_id = self.task_queue.get_latest_task_run_id_for_resource(
                        task_type="index_file",
                        resource_id=file_id,
                    )
                    if latest_run_id:
                        latest_run = self.task_queue.get_task_run(latest_run_id) or {}
                        latest_state = str(latest_run.get("state") or "")
                        if latest_state in {TaskState.PENDING.value, TaskState.RUNNING.value}:
                            # When force=True, allow a new TaskRun to be scheduled even if the previous
                            # run looks active. This is used for manual "reindex everything" workflows
                            # and helps recover from stuck TaskRuns (e.g. workers restarted).
                            if not force:
                                skipped_files.append(file_id)
                                continue
                            logger.warning(
                                "Force reindex requested while an active TaskRun exists (file_id=%s run_id=%s state=%s)",
                                file_id,
                                latest_run_id,
                                latest_state,
                            )
                else:
                    active_task = self._active_index_tasks.get(file_id)
                    if active_task is not None and not active_task.done():
                        skipped_files.append(file_id)
                        continue

                if metadata.status == FileStatus.DELETED:
                    invalid_files.append(f"File is deleted: {file_id}")
                    continue

                # Only allow indexing for STORED or FAILED files, unless force is enabled.
                # Skip files that are in intermediate processing states (PARSED, CHUNKED) or already running.
                if metadata.status in {FileStatus.STORED, FileStatus.FAILED} or force:
                    valid_files.append(file_id)
                else:
                    skipped_files.append(file_id)
            except Exception as e:
                invalid_files.append(file_id)
                logger.exception(f"Error accessing file {file_id}")
                continue

        # If all files are invalid or already indexed/in progress, directly return
        if not valid_files:
            message_parts = []
            if invalid_files:
                message_parts.append(f"Invalid files: {'; '.join(invalid_files)}")
            if skipped_files:
                message_parts.append(f"Skipped files (already indexed or in progress): {'; '.join(skipped_files)}")
            message_parts.append("No files scheduled for indexing.")
            return "\n".join(message_parts)

        logger.info(
            f"Triggering indexing for files: {'; '.join(valid_files)}"
        )

        results: List[Dict[str, Any] | Exception] = []
        if self._use_celery():
            for file_id in valid_files:
                task_run_id = self.task_queue.create_task_run(
                    task_type="index_file",
                    owner_id=user_id,
                    resource_id=file_id,
                    metadata={"trigger": "manual"},
                )
                from application.knowledge.celery_tasks import index_file as index_file_task

                queue = os.getenv("CELERY_QUEUE_INDEXING", "indexing")
                index_file_task.apply_async(
                    kwargs={"file_id": file_id, "owner_id": str(user_id)},
                    task_id=task_run_id,
                    queue=queue,
                )
        else:
            task_specs: List[tuple[str, str]] = []
            for file_id in valid_files:
                task_run_id = self.task_queue.create_task_run(
                    task_type="index_file",
                    owner_id=user_id,
                    resource_id=file_id,
                    metadata={"trigger": "manual"},
                )
                task_specs.append((file_id, task_run_id))

            if wait:
                tasks = [self._index_file_background(file_id, task_run_id=task_run_id) for file_id, task_run_id in task_specs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for file_id, task_run_id in task_specs:
                    task = asyncio.create_task(self._index_file_background(file_id, task_run_id=task_run_id))
                    self._track_background_task(file_id, task)

        message_parts: List[str] = []
        if wait and not self._use_celery():
            succeeded: List[str] = []
            failed: List[str] = []
            for file_id, outcome in zip(valid_files, results):
                if isinstance(outcome, Exception):
                    failed.append(file_id)
                    continue
                if isinstance(outcome, dict) and outcome.get("success", False):
                    succeeded.append(file_id)
                else:
                    failed.append(file_id)
            message_parts.append(f"Indexing completed for files: {'; '.join(valid_files)}")
            message_parts.append(f"succeeded={len(succeeded)}, failed={len(failed)}")
            if failed:
                message_parts.append(f"failed_files: {'; '.join(failed)}")
        else:
            message_parts.append(f"Indexing started for files: {'; '.join(valid_files)}")
        if skipped_files:
            message_parts.append(f"Skipped files (already indexed or in progress): {'; '.join(skipped_files)}")
        if invalid_files:
            message_parts.append(f"Invalid files: {'; '.join(invalid_files)}")

        return "\n".join(message_parts)

    async def get_file_task_status(self, file_id: str, user_id: uuid.UUID) -> Dict[str, Any]:
        metadata = await self._run_blocking(self.file_storage.get_file_metadata, file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        if metadata.status == FileStatus.DELETED or self._is_file_marked_for_deletion(file_id):
            raise HTTPException(status_code=404, detail="File not found")

        permission_type = await self._run_blocking(self.check_file_access, file_id, user_id)
        if permission_type is None:
            raise HTTPException(status_code=403, detail="You are not allowed to access this file")

        task_run_id = await self._run_blocking(
            self.task_queue.get_latest_task_run_id_for_resource,
            task_type="index_file",
            resource_id=file_id,
        )
        task_run = None
        if task_run_id:
            task_run = await self._run_blocking(self.task_queue.get_task_run, task_run_id)

        return {
            "file_id": file_id,
            "file_status": metadata.status.value if metadata.status else None,
            "task_run_id": task_run_id,
            "task_state": (task_run or {}).get("state"),
            "progress_percent": (task_run or {}).get("progress_percent"),
            "error_message": (task_run or {}).get("error_message"),
            "updated_at_ms": (task_run or {}).get("updated_at_ms"),
        }

    async def _index_multiple_files_background(self, file_ids: List[str], user_id: uuid.UUID):
        """Background task for indexing multiple files with semaphore control
        
        Reuses _index_file_background for each file, ensuring consistent semaphore control
        and preventing GPU OOM when processing multiple files concurrently.
        Each file will acquire the semaphore individually, so they are processed with
        controlled concurrency based on max_concurrent_indexing configuration.
        """
        logger.info(f"Starting background indexing for {len(file_ids)} files for user {user_id}")
        
        # Reuse _index_file_background for each file
        # This ensures each file goes through semaphore control, preventing GPU OOM
        # The semaphore in _index_file_background will limit concurrent processing
        tasks = [self._index_file_background(file_id) for file_id in file_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successful = 0
        failed = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif isinstance(result, dict) and result.get("success", False):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Background indexing completed for user {user_id}: {successful} successful, {failed} failed out of {len(file_ids)} files")

    def get_indexing_status(self) -> Dict[str, Any]:
        """
        Get current indexing semaphore status for monitoring.

        Returns:
            Dictionary containing semaphore status information
        """
        max_concurrent = self.indexing_semaphore._value + len(self.indexing_semaphore._waiters)
        available_slots = self.indexing_semaphore._value
        waiting_tasks = len(self.indexing_semaphore._waiters)
        active_tasks = max_concurrent - available_slots

        return {
            "max_concurrent_indexing": max_concurrent,
            "available_slots": available_slots,
            "waiting_tasks": waiting_tasks,
            "active_tasks": active_tasks
        }

    async def get_file_chunk_mindmaps(self, file_id: str, user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Retrieve per-chunk payloads for mindmap export.

        Mindmaps are optional: if a chunk has no `metadata["mindmap"]`, we still include the chunk content so the
        downstream mindmap export can infer structure from text (on-demand export).
        """
        metadata = await self._run_blocking(self.file_storage.get_file_metadata, file_id)

        if metadata is None:
            raise HTTPException(status_code=404, detail="File not found")

        if metadata.owner_id != user_id:
            raise HTTPException(status_code=403, detail="You are not allowed to access this file")

        parsed_content_list = await self._run_blocking(
            self.file_index.parsed_content_storage.metadata_store.list_parsed_content_metadata,
            source_file_id=file_id
        )

        if not parsed_content_list:
            return {"file_id": file_id, "filename": metadata.filename, "chunks": []}

        chunk_id_order: List[str] = []
        chunk_index_map: Dict[str, int] = {}

        for parsed_content in parsed_content_list:
            if not parsed_content or not getattr(parsed_content, "parsed_content_id", None):
                continue

            chunk_metadata_list = await self._run_blocking(
                self.file_index.chunk_storage.metadata_store.list_chunk_metadata,
                source_parsed_content_id=parsed_content.parsed_content_id
            )

            for chunk_meta in chunk_metadata_list or []:
                chunk_id = getattr(chunk_meta, "chunk_id", None)
                if not chunk_id:
                    continue
                if chunk_id not in chunk_index_map:
                    chunk_id_order.append(chunk_id)
                chunk_index_map[chunk_id] = getattr(chunk_meta, "chunk_index", None)

        if not chunk_id_order:
            return {"file_id": file_id, "filename": metadata.filename, "chunks": []}

        total_chunks = len(chunk_id_order)
        max_chunks = int(KNOWLEDGE_MINDMAP_EXPORT_MAX_CHUNKS or 0)
        truncated = False
        if max_chunks > 0 and total_chunks > max_chunks:
            truncated = True
            if max_chunks == 1:
                chunk_id_order = [chunk_id_order[0]]
            else:
                n = total_chunks
                sampled: list[str] = []
                last_idx = -1
                for i in range(max_chunks):
                    idx = int(i * (n - 1) / (max_chunks - 1))
                    if idx <= last_idx:
                        idx = last_idx + 1
                    if idx >= n:
                        idx = n - 1
                    sampled.append(chunk_id_order[idx])
                    last_idx = idx
                chunk_id_order = sampled

        graph_store = None
        for indexer in getattr(self.file_index, "indexers", []):
            if hasattr(indexer, "graph_store"):
                graph_store = indexer.graph_store
                break

        if graph_store is None:
            raise HTTPException(status_code=500, detail="Graph store is not configured for mind map export")

        try:
            chunk_objects = await self._run_blocking(graph_store.get_by_ids, chunk_id_order)
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from graph store for file {file_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve chunk data from graph store")

        chunk_map = {chunk.id: chunk for chunk in chunk_objects if getattr(chunk, "id", None)}

        mindmap_chunks: List[Dict[str, Any]] = []
        for chunk_id in chunk_id_order:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue

            metadata_dict = chunk.metadata or {}
            mindmap = metadata_dict.get("mindmap") if isinstance(metadata_dict, dict) else None

            mindmap_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index_map.get(chunk_id),
                    "content": chunk.content,
                    "mindmap": mindmap if isinstance(mindmap, dict) else {},
                }
            )

        return {
            "file_id": file_id,
            "filename": metadata.filename,
            "chunks": mindmap_chunks,
            "chunks_total": total_chunks,
            "chunks_sampled": len(chunk_id_order),
            "chunks_truncated": truncated,
        }

    async def shutdown(self):
        """
        Shutdown the Knowledge module and flush all pending indexer data.
        Should be called when the application is shutting down.
        """
        logger.info("Shutting down Knowledge module...")

        # Shutdown all indexers to flush pending chunks
        if hasattr(self.file_index, 'indexers') and self.file_index.indexers:
            for indexer in self.file_index.indexers:
                if hasattr(indexer, 'shutdown'):
                    try:
                        logger.info(f"Shutting down indexer: {type(indexer).__name__}")
                        await indexer.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down indexer {type(indexer).__name__}: {e}")

        logger.info("Knowledge module shutdown complete")
