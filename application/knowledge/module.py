from framework.module import AbstractModule
import logging
from typing import TYPE_CHECKING, List, Optional

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
        """Background task for indexing files"""
        try:
            logger.info(f"Starting background indexing for file_id: {doc_id}")
            result = await self.file_index.index_file(doc_id)
            if result.get("success"):
                logger.info(f"Background indexing completed successfully for file_id: {doc_id}")
            else:
                logger.error(f"Background indexing failed for file_id: {doc_id}, error: {result.get('error_message')}")
        except Exception as e:
            logger.error(f"Background indexing failed for file_id: {doc_id}, exception: {str(e)}")

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
        
