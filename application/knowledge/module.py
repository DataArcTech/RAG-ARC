from framework.module import AbstractModule
import logging
from typing import TYPE_CHECKING, List, Optional, Dict, Any

if TYPE_CHECKING:
    from config.application.knowledge_config import KnowledgeConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi.responses import Response
from fastapi import UploadFile, HTTPException
from encapsulation.data_model.orm_models import (
    FileMetadata, FileStatus,
    FilePermission, PermissionReceiverType, PermissionType
)
from core.utils.thread_pool import run_blocking, run_coroutine_in_thread

class Knowledge(AbstractModule):
    def __init__(self, config: 'KnowledgeConfig'):
        super().__init__(config=config)
        self.file_storage = config.file_storage_config.build()
        self.file_index = config.index_manager_config.build()
        
        # Semaphore to control concurrent indexing operations
        self.indexing_semaphore = asyncio.Semaphore(config.max_concurrent_indexing)
        self._active_index_tasks: Dict[str, asyncio.Task] = {}
        self._files_marked_for_deletion: set[str] = set()
        self._files_marked_for_deletion_by_owner: Dict[uuid.UUID, set[str]] = {}
        self._file_owner_cache: Dict[str, uuid.UUID] = {}
        self._active_deletion_tasks: Dict[str, asyncio.Task] = {}
        self._deletion_failures: Dict[str, str] = {}

    def _track_background_task(self, doc_id: str, task: asyncio.Task) -> None:
        """Register a background indexing task so it can be cancelled or awaited later."""
        self._active_index_tasks[doc_id] = task

        def _cleanup(fut: asyncio.Task, file_id: str = doc_id) -> None:
            self._active_index_tasks.pop(file_id, None)
            if fut.cancelled():
                logger.info(f"Background indexing task cancelled for file_id: {file_id}")
            elif fut.exception():
                logger.error(
                    f"Background indexing task failed for file_id {file_id}: {fut.exception()}",
                    exc_info=True
                )

        task.add_done_callback(_cleanup)

    async def _cancel_indexing_task(self, doc_id: str) -> None:
        """Cancel an active background indexing task for the specified file."""
        task = self._active_index_tasks.get(doc_id)
        if not task:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Cancelled indexing task awaited for file_id: {doc_id}")

    def _mark_file_for_deletion(self, doc_id: str, owner_id: Optional[uuid.UUID] = None) -> None:
        """Mark a file so background tasks can skip further processing."""
        self._files_marked_for_deletion.add(doc_id)
        resolved_owner = self._resolve_owner_for_file(doc_id, owner_id)
        if resolved_owner is not None:
            owner_set = self._files_marked_for_deletion_by_owner.setdefault(resolved_owner, set())
            owner_set.add(doc_id)
            self._file_owner_cache[doc_id] = resolved_owner

    def _unmark_file_for_deletion(self, doc_id: str) -> None:
        self._files_marked_for_deletion.discard(doc_id)
        owner_id = self._file_owner_cache.pop(doc_id, None)
        if owner_id is not None:
            owner_set = self._files_marked_for_deletion_by_owner.get(owner_id)
            if owner_set is not None:
                owner_set.discard(doc_id)
                if not owner_set:
                    self._files_marked_for_deletion_by_owner.pop(owner_id, None)

    def _is_file_marked_for_deletion(self, doc_id: str) -> bool:
        return doc_id in self._files_marked_for_deletion

    def _resolve_owner_for_file(
        self,
        doc_id: str,
        explicit_owner: Optional[uuid.UUID] = None
    ) -> Optional[uuid.UUID]:
        if explicit_owner is not None:
            return explicit_owner
        cached_owner = self._file_owner_cache.get(doc_id)
        if cached_owner is not None:
            return cached_owner
        try:
            metadata = self.file_storage.get_file_metadata(doc_id)
        except Exception:
            metadata = None
        if metadata and getattr(metadata, "owner_id", None) is not None:
            owner_id = metadata.owner_id
            self._file_owner_cache[doc_id] = owner_id
            return owner_id
        return None

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
        try:
            # Read file data asynchronously to avoid blocking the event loop
            file_data = await self._run_blocking(file.file.read)
            
            # Upload file asynchronously to avoid blocking the event loop
            doc_id = await self._run_blocking(
                self.file_storage.upload_file,
                filename=(relative_path or file.filename),
                file_data=file_data,
                owner_id=user_id,
                content_type=file.content_type
            )
            # Start indexing in background (fire-and-forget)
            # execute file indexing without waiting for it to complete
            task = asyncio.create_task(self._index_file_background(doc_id))
            self._track_background_task(doc_id, task)
            logger.info(f"File {file.filename} uploaded with ID {doc_id}, indexing started in background")
            return doc_id

        except Exception as e:
            logger.error(e)
            raise

    async def _index_file_background(self, doc_id: str) -> Dict[str, Any]:
        """Background task for indexing files with semaphore control
        
        Returns:
            Dict with indexing result containing 'success' (bool) and 'file_id' (str) keys
        """
        if self._is_file_marked_for_deletion(doc_id):
            logger.info(f"Skipping indexing for file_id {doc_id} because it is marked for deletion")
            return {"success": False, "file_id": doc_id, "error_message": "file scheduled for deletion"}

        async with self.indexing_semaphore:
            try:
                if self._is_file_marked_for_deletion(doc_id):
                    logger.info(f"Aborting indexing for file_id {doc_id}; deletion scheduled")
                    return {"success": False, "file_id": doc_id, "error_message": "file scheduled for deletion"}

                logger.info(f"Starting background indexing for file_id: {doc_id} (semaphore acquired)")
                # IndexManager is async but performs heavy blocking work; run it off the main event loop.
                result = await self._run_coroutine_in_thread(self.file_index.index_file, doc_id)
                if result.get("success"):
                    logger.info(f"Background indexing completed successfully for file_id: {doc_id}")
                else:
                    logger.error(f"Background indexing failed for file_id: {doc_id}, error: {result.get('error_message')}")
                return result
            except asyncio.CancelledError:
                logger.info(f"Background indexing task cancelled for file_id: {doc_id}")
                raise
            except Exception as e:
                logger.error(f"Background indexing failed for file_id: {doc_id}, exception: {str(e)}")
                return {"success": False, "file_id": doc_id, "error_message": str(e)}
            finally:
                logger.debug(f"Background indexing semaphore released for file_id: {doc_id}")

    async def get_file(self, doc_id: str, user_id: uuid.UUID) -> Response:
        # Run database queries asynchronously to avoid blocking the event loop
        metadata = await self._run_blocking(
            self.file_storage.get_file_metadata,
            doc_id
        )

        if metadata is None:
            raise HTTPException(status_code=404, detail="File not found")
        if metadata.status == FileStatus.DELETED or self._is_file_marked_for_deletion(doc_id):
            raise HTTPException(status_code=404, detail="File not found")

        # Check if user has access (owner or has VIEW/EDIT permission)
        permission_type = await self._run_blocking(
            self.check_file_access,
            doc_id,
            user_id
        )
        if permission_type is None:
            raise HTTPException(status_code=403, detail="You are not allowed to access this file")

        content = await self._run_blocking(
            self.file_storage.get_file_content,
            doc_id
        )
        if content is None:
            raise HTTPException(status_code=404, detail="File content not found")

        download_name = Path(str(metadata.filename or "")).name or "download"
        headers = {"Content-Disposition": f"attachment; filename=\"{download_name}\""}
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

        # Schedule deletion in background
        delete_task = asyncio.create_task(self._delete_file_background(doc_id))
        self._track_deletion_task(doc_id, delete_task)
        logger.info(f"Deletion scheduled for file_id: {doc_id}")

        response = {"status": "deleting", "file_id": doc_id}
        if failure_reason:
            response["previous_failure"] = failure_reason
        return response

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

    async def list_user_files(
        self,
        user_id: uuid.UUID,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[FileMetadata]:
        """
        Get all files accessible to a specific user (files with permissions only).
        
        Args:
            user_id: UUID of the user
            status: Optional filter by file status
            limit: Maximum number of files to return
            offset: Number of files to skip (for pagination)
            
        Returns:
            List of FileMetadata objects accessible to the user
        """
        try:
            # Run database query asynchronously to avoid blocking the event loop
            files = await self._run_blocking(
                self.file_storage.list_accessible_files,
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            if status is None:
                files = [
                    file for file in files
                    if self._is_active_status(file.status) and not self._is_file_marked_for_deletion(file.file_id)
                ]
            logger.info(f"Retrieved {len(files)} accessible files for user {user_id}")
            return files
        except Exception as e:
            logger.error(f"Failed to list accessible files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve files: {str(e)}")
    
    async def count_user_files(
        self,
        user_id: uuid.UUID,
        status: FileStatus | None = None
    ) -> int:
        """
        Count all files accessible to a specific user (files with permissions).
        
        Args:
            user_id: UUID of the user
            status: Optional filter by file status
            
        Returns:
            Total count of files accessible to the user
        """
        try:
            if status is None:
                # Run database queries asynchronously to avoid blocking the event loop
                total = await self._run_blocking(
                    self.file_storage.count_accessible_files,
                    user_id=user_id
                )
                deleted = await self._run_blocking(
                    self.file_storage.count_accessible_files,
                    user_id=user_id,
                    status=FileStatus.DELETED
                )
                mark_only = len(self._files_marked_for_deletion_by_owner.get(user_id, set()))
                count = max(total - deleted - mark_only, 0)
            else:
                count = await self._run_blocking(
                    self.file_storage.count_accessible_files,
                    user_id=user_id,
                    status=status
                )
            logger.info(f"Counted {count} accessible files for user {user_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to count accessible files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to count files: {str(e)}")

    def _is_active_status(self, status: FileStatus) -> bool:
        return status != FileStatus.DELETED

    def is_file_active(self, file_id: str) -> bool:
        """
        Determine if a file is still visible for retrieval/listing purposes.
        """
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

        return True

    async def trigger_indexing(self, file_ids: List[str], user_id: uuid.UUID) -> str:
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

                # Only allow indexing for STORED or FAILED files
                # Skip files that are already indexed or in intermediate processing states
                if metadata.status == FileStatus.STORED or metadata.status == FileStatus.FAILED:
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

        # Start background indexing task for files not indexed yet only
        await self._index_multiple_files_background(valid_files, user_id)

        # Return immediately with basic info
        message_parts = [
            f"Indexing started for files: {'; '.join(valid_files)}"
        ]
        if skipped_files:
            message_parts.append(f"Skipped files (already indexed or in progress): {'; '.join(skipped_files)}")
        if invalid_files:
            message_parts.append(f"Invalid files: {'; '.join(invalid_files)}")

        return "\n".join(message_parts)

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
        """Retrieve chunk mind map data for a file owned by the user."""
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

            if mindmap and isinstance(mindmap, dict) and mindmap.get("nodes"):
                mindmap_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index_map.get(chunk_id),
                    "content": chunk.content,
                    "mindmap": mindmap
                })

        return {
            "file_id": file_id,
            "filename": metadata.filename,
            "chunks": mindmap_chunks
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

    # ==================== FILE PERMISSION MANAGEMENT ====================
    def get_file_id_by_permission_id(
        self,
        permission_id: uuid.UUID
    ) -> Optional[str]:
        """
        Get the file ID by permission ID.

        Args:
            permission_id: Permission ID to look up

        Returns:
            File ID string if permission exists, None otherwise
        """
        permission = self.file_storage.metadata_store.get_file_permission(permission_id)
        if not permission or not permission.file_id:
            return None
        return permission.file_id

    def grant_file_permission(
        self,
        file_id: str,
        receiver_type: PermissionReceiverType,
        permission_type: PermissionType,
        granted_by: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        department_id: Optional[uuid.UUID] = None
    ) -> uuid.UUID:
        """
        Grant file permission to a user, department, or all users.

        Args:
            file_id: File ID to grant permission for
            receiver_type: Type of receiver (USER, DEPARTMENT, or ALL)
            permission_type: Type of permission (VIEW or EDIT)
            granted_by: User ID who is granting the permission
            user_id: User ID if receiver_type is USER
            department_id: Department ID if receiver_type is DEPARTMENT

        Returns:
            Permission ID (UUID) of the created permission

        Raises:
            HTTPException: If file not found or user doesn't have permission to grant
        """
        metadata = self.file_storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        # Check if user has EDIT permission to grant permissions
        if self.check_file_access(file_id, granted_by) != PermissionType.EDIT:
            raise HTTPException(status_code=403, detail="You are not allowed to grant permissions for this file")

        try:
            permission_id = self.file_storage.metadata_store.grant_file_permission(
                file_id=file_id,
                receiver_type=receiver_type,
                permission_type=permission_type,
                granted_by=granted_by,
                user_id=user_id,
                department_id=department_id,
            )
            logger.info(f"Granted {permission_type.value} permission for file {file_id} to {receiver_type.value}")
            return permission_id
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to grant file permission: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to grant permission: {str(e)}")

    def revoke_file_permission(
        self,
        permission_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> bool:
        """
        Revoke a file permission by permission ID.

        Args:
            permission_id: Permission ID to revoke
            user_id: User ID requesting the revocation (must have EDIT permission)

        Returns:
            True if permission was revoked, False if not found

        Raises:
            HTTPException: If user doesn't have permission to revoke
        """
        # Get permission to check user has EDIT permission to revoke permissions
        permission = self.file_storage.metadata_store.get_file_permission(permission_id)
        if not permission:
            raise HTTPException(status_code=404, detail="Permission not found")
        if self.check_file_access(permission.file_id, user_id) != PermissionType.EDIT:
            raise HTTPException(status_code=403, detail="You are not allowed to revoke permissions for this file")
        
        try:
            result = self.file_storage.metadata_store.revoke_file_permission(
                permission_id=permission_id
            )
            if result:
                logger.info(f"Revoked permission {permission_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to revoke file permission: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to revoke permission: {str(e)}")


    def list_file_permissions(
        self,
        file_id: str,
        user_id: uuid.UUID
    ) -> List[FilePermission]:
        """
        List all permissions for a specific file.

        Args:
            file_id: File ID to list permissions for
            user_id: User ID requesting the list (must have VIEW or EDIT permission)

        Returns:
            List of FilePermission objects

        Raises:
            HTTPException: If file not found or user doesn't have permission
        """
        # Check if file exists
        metadata = self.file_storage.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        # Check if user has VIEW or EDIT permission to list permissions
        permission_type = self.check_file_access(file_id, user_id)
        if permission_type is None:
            raise HTTPException(status_code=403, detail="You are not allowed to list permissions for this file")

        try:
            permissions = self.file_storage.metadata_store.list_file_permissions(file_id)
            logger.info(f"Retrieved {len(permissions)} permissions for file {file_id}")
            return permissions
        except Exception as e:
            logger.error(f"Failed to list file permissions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list permissions: {str(e)}")

    def list_user_permissions(
        self,
        user_id: uuid.UUID
    ) -> List[FilePermission]:
        """
        List all permissions granted to a specific user (direct grants and department grants).

        Args:
            user_id: User ID to list permissions for

        Returns:
            List of FilePermission objects

        Raises:
            HTTPException: If user not found
        """
        try:
            permissions = self.file_storage.metadata_store.list_user_permissions(user_id)
            logger.info(f"Retrieved {len(permissions)} permissions for user {user_id}")
            return permissions
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to list user permissions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list permissions: {str(e)}")

    def check_file_access(
        self,
        file_id: str,
        user_id: uuid.UUID
    ) -> Optional[PermissionType]:
        """
        Check if a user has access to a file and return the permission type.

        Args:
            file_id: File ID to check
            user_id: User ID to check access for

        Returns:
            PermissionType (VIEW or EDIT) if user has access, None otherwise
        """
        try:
            permission_type = self.file_storage.metadata_store.check_file_access(
                file_id=file_id,
                user_id=user_id
            )
            return permission_type
        except Exception as e:
            logger.error(f"Failed to check file access: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to check access: {str(e)}")

    def update_file_permission(
        self,
        permission_id: uuid.UUID,
        permission_type: PermissionType,
        user_id: uuid.UUID,
    ) -> bool:
        """
        Update an existing file permission.

        Args:
            permission_id: Permission ID to update
            permission_type: New permission type (VIEW or EDIT)
            user_id: User ID requesting the update (must have EDIT permission)

        Returns:
            True if permission was updated, False if not found

        Raises:
            HTTPException: If permission not found or user doesn't have permission
        """
        # Get permission to check user has EDIT permission to update permissions
        permission = self.file_storage.metadata_store.get_file_permission(permission_id)
        if not permission:
            raise HTTPException(status_code=404, detail="Permission not found")
        
        file_id = permission.file_id

        # Check if user has EDIT permission
        if self.check_file_access(file_id, user_id) != PermissionType.EDIT:
            raise HTTPException(
                status_code=403,
                detail="Only users with EDIT permission can update permissions"
            )

        try:
            result = self.file_storage.metadata_store.update_file_permission(
                permission_id=permission_id,
                permission_type=permission_type,
            )
            
            if result:
                logger.info(f"Updated permission {permission_id}")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update file permission: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update permission: {str(e)}")
