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
from fastapi.responses import Response
from fastapi import File, UploadFile, HTTPException
from encapsulation.data_model.orm_models import FileMetadata, FileStatus

class Knowledge(AbstractModule):
    def __init__(self, config: 'KnowledgeConfig'):
        super().__init__(config=config)
        self.file_storage = config.file_storage_config.build()
        self.file_index = config.index_manager_config.build()
        
        # Semaphore to control concurrent indexing operations
        max_concurrent_indexing = config.max_concurrent_indexing
        self.indexing_semaphore = asyncio.Semaphore(max_concurrent_indexing)
    
    def upload_file(self, file: UploadFile, user_id: uuid.UUID) -> str:
        try:
            doc_id = self.file_storage.upload_file(
                filename=file.filename,
                file_data=file.file.read(),
                owner_id=user_id,
                content_type=file.content_type
            )
            # Start indexing in background (fire-and-forget)
            self._start_background_indexing(doc_id)
            logger.info(f"File {file.filename} uploaded with ID {doc_id}, indexing started in background")
            return doc_id

        except Exception as e:
            logger.error(e)
            raise

    def _start_background_indexing(self, doc_id: str):
        """Start background indexing task safely"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create the task
            loop.create_task(self._index_file_background(doc_id))
        except RuntimeError:
            # No event loop running, start a new one in a thread
            import threading
            def run_async():
                asyncio.run(self._index_file_background(doc_id))
            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()

    async def _index_file_background(self, doc_id: str):
        """Background task for indexing files with semaphore control"""
        async with self.indexing_semaphore:
            try:
                logger.info(f"Starting background indexing for file_id: {doc_id} (semaphore acquired)")
                result = await self.file_index.index_file(doc_id)
                if result.get("success"):
                    logger.info(f"Background indexing completed successfully for file_id: {doc_id}")
                else:
                    logger.error(f"Background indexing failed for file_id: {doc_id}, error: {result.get('error_message')}")
            except Exception as e:
                logger.error(f"Background indexing failed for file_id: {doc_id}, exception: {str(e)}")
            finally:
                logger.debug(f"Background indexing semaphore released for file_id: {doc_id}")

    def get_file(self, doc_id: str, user_id: uuid.UUID) -> Response:
        metadata = self.file_storage.get_file_metadata(doc_id)

        if metadata is None:
            raise HTTPException(status_code=404, detail="File not found")

        if metadata.owner_id != user_id:
            raise HTTPException(status_code=403, detail="You are not allowed to access this file")

        content = self.file_storage.get_file_content(doc_id)
        if content is None:
            raise HTTPException(status_code=404, detail="File content not found")

        headers = {"Content-Disposition": f"attachment; filename=\"{metadata.filename}\""}
        return Response(content=content, media_type=metadata.content_type, headers=headers)

    def delete_file(self, doc_id: str, user_id: uuid.UUID):
        # Check if the file exists before attempting deletion
        metadata = self.file_storage.get_file_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        # Only the file owner can delete the file
        if metadata.owner_id != user_id:
            raise HTTPException(status_code=403, detail="You are not allowed to delete this file")
        
        # Delete all file data including derived artifacts and file metadata
        # This handles the complete deletion in the correct order to avoid foreign key constraint violations
        try:
            deletion_result = self.file_index.delete_file_data(doc_id, delete_file_metadata=True)
            # IndexManager.delete_file_data returns a dict with a "success" flag
            if not deletion_result.get("success", False):
                error_msg = deletion_result.get("error_message", "")
                if error_msg and "file_id must be a non-empty string" not in error_msg:
                    logger.error(f"File deletion failed for {doc_id}: {error_msg}")
                    raise HTTPException(status_code=500, detail=f"Failed to delete file: {error_msg}")
                else:
                    logger.info(f"No indexed content found for file {doc_id}, but deletion completed")

            if not self.file_storage.delete_file(doc_id):
                raise HTTPException(status_code=500, detail="Failed to delete file")
        except HTTPException:
            # Propagate 4xx errors up to the router
            raise
        except Exception as e:
            logger.error(f"Error during file deletion for {doc_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete file")

    def list_user_files(
        self,
        user_id: uuid.UUID,
        status: Optional[FileStatus] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[FileMetadata]:
        """
        Get all files for a specific user.
        
        Args:
            user_id: UUID of the file owner
            status: Optional filter by file status
            limit: Maximum number of files to return
            offset: Number of files to skip (for pagination)
            
        Returns:
            List of FileMetadata objects
        """
        try:
            files = self.file_storage.list_files_by_owner(
                owner_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            logger.info(f"Retrieved {len(files)} files for user {user_id}")
            return files
        except Exception as e:
            logger.error(f"Failed to list files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve files: {str(e)}")
    
    def count_user_files(
        self,
        user_id: uuid.UUID,
        status: FileStatus | None = None
    ) -> int:
        """
        Count all files for a specific user.
        
        Args:
            user_id: UUID of the file owner
            status: Optional filter by file status
            
        Returns:
            Total count of files for the user
        """
        try:
            count = self.file_storage.count_files(
                owner_id=user_id,
                status=status
            )
            logger.info(f"Counted {count} files for user {user_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to count files for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to count files: {str(e)}")

    def trigger_indexing(self, file_ids: List[str], user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Trigger indexing for multiple files asynchronously.
        
        Args:
            file_ids: List of file IDs to index
            user_id: UUID of the user requesting the indexing
            
        Returns:
            Dictionary containing basic info about the triggered indexing
        """
        try:
            # Validate files and collect valid ones
            valid_file_ids = []
            invalid_files = []
            
            for file_id in file_ids:
                try:
                    metadata = self.file_storage.get_file_metadata(file_id)
                    if metadata is None:
                        invalid_files.append(f"File not found: {file_id}")
                        continue
                    if metadata.owner_id != user_id:
                        invalid_files.append(f"You are not allowed to access file: {file_id}")
                        continue
                    valid_file_ids.append(file_id)
                except Exception as e:
                    invalid_files.append(f"Error accessing file {file_id}: {str(e)}")
                    continue
            
            # If no valid files, return error
            if not valid_file_ids:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No valid files to index. Issues: {'; '.join(invalid_files)}"
                )
            
            logger.info(f"Triggering async indexing for {len(valid_file_ids)} valid files (out of {len(file_ids)} requested) for user {user_id}")
            
            # Start background indexing task for valid files only
            self._start_background_indexing_multiple_files(valid_file_ids, user_id)
            
            # Return immediately with basic info
            message = f"Indexing started for {len(valid_file_ids)} files in background"
            if invalid_files:
                message += f". Skipped {len(invalid_files)} invalid files: {'; '.join(invalid_files)}"
            
            return {
                "total_files": len(file_ids),  # Return original count for consistency with test expectations
                "status": "indexing_started",
                "message": message
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions (400, 403)
            raise
        except Exception as e:
            logger.error(f"Failed to trigger indexing for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to trigger indexing: {str(e)}")

    def _start_background_indexing_multiple_files(self, file_ids: List[str], user_id: uuid.UUID):
        """Start background indexing task for multiple files safely"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, create the task
            loop.create_task(self._index_multiple_files_background(file_ids, user_id))
        except RuntimeError:
            # No event loop running, start a new one in a thread
            import threading
            def run_async():
                asyncio.run(self._index_multiple_files_background(file_ids, user_id))
            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()

    async def _index_multiple_files_background(self, file_ids: List[str], user_id: uuid.UUID):
        """Background task for indexing multiple files with semaphore control"""
        async with self.indexing_semaphore:
            try:
                logger.info(f"Starting background indexing for {len(file_ids)} files for user {user_id} (semaphore acquired)")
                
                # Use the IndexManager's process_multiple_files method
                result = self.file_index.process_multiple_files(file_ids)
                
                logger.info(f"Background indexing completed for user {user_id}: {result.get('successful_files', 0)} successful, {result.get('failed_files', 0)} failed out of {result.get('total_files', 0)} files")
                
            except Exception as e:
                logger.error(f"Background indexing failed for user {user_id}: {str(e)}")
            finally:
                logger.debug(f"Background indexing semaphore released for user {user_id} ({len(file_ids)} files)")

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
        
