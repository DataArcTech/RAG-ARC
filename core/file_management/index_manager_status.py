import logging
from datetime import datetime
from typing import Any, Dict, List

from encapsulation.data_model.orm_models import ChunkIndexStatus

logger = logging.getLogger(__name__)


class _IndexManagerStatusMixin:
    def _update_indexed_chunks_status(self, chunk_ids: List[str], indexing_results: Dict[str, Any], **kwargs: Any) -> bool:
        """
        Update chunk metadata status to INDEXED for successfully indexed chunks.

        Args:
            chunk_ids: List of chunk IDs that were indexed
            indexing_results: Results from indexers
            **kwargs: Additional arguments

        Returns:
            bool: True if any chunks were successfully updated, False otherwise
        """
        # Check if any indexer succeeded
        any_success = any(result.get("success", False) for result in indexing_results.values())

        if not any_success:
            logger.warning("No indexer succeeded, skipping chunk status update")
            return False

        # Collect successfully indexed chunk IDs from all indexers
        successfully_indexed_ids = set()
        for indexer_name, result in indexing_results.items():
            if result.get("success", False):
                indexed_ids = result.get("indexed_ids", [])
                if indexed_ids:
                    successfully_indexed_ids.update(indexed_ids)

        if not successfully_indexed_ids:
            logger.warning("No chunks were successfully indexed")
            return False

        # Update status for each successfully indexed chunk
        now = datetime.now(tz=datetime.now().astimezone().tzinfo)
        updated_count = 0
        failed_count = 0

        for chunk_id in successfully_indexed_ids:
            try:
                success = self.chunk_storage.metadata_store.update_chunk_metadata(
                    chunk_id,
                    {"index_status": ChunkIndexStatus.INDEXED, "indexed_at": now},
                    **kwargs,
                )
                if success:
                    updated_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to update status for chunk {chunk_id}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error updating status for chunk {chunk_id}: {e}")

        logger.info(f"Updated chunk status: {updated_count} succeeded, {failed_count} failed")
        return updated_count > 0

    def _update_file_status_to_indexed(self, file_id: str, **kwargs: Any) -> None:
        """
        Update file metadata status to INDEXED after successful indexing.

        Args:
            file_id: File ID to update
            **kwargs: Additional arguments
        """
        from encapsulation.data_model.orm_models import FileStatus

        try:
            # Deletion can race with indexing (file hidden first, physical cleanup later).
            # Never resurrect a DELETED file back to INDEXED.
            try:
                metadata = self.file_storage.get_file_metadata(file_id)
            except Exception:
                metadata = None
            if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                logger.info("Skip updating file %s to INDEXED because status is DELETED", file_id)
                return

            success = self.file_storage.metadata_store.update_file_metadata(
                file_id,
                {"status": FileStatus.INDEXED},
                **kwargs,
            )
            if success:
                logger.info(f"Updated file {file_id} status to INDEXED")
            else:
                logger.warning(f"Failed to update file {file_id} status to INDEXED")
        except Exception as e:
            logger.error(f"Error updating file {file_id} status: {e}")

    def _update_file_status_to_partially_indexed(self, file_id: str, **kwargs: Any) -> None:
        """
        Update file metadata status to PARTIAL_INDEXED when some indexers succeed but not all.
        """
        from encapsulation.data_model.orm_models import FileStatus

        try:
            try:
                metadata = self.file_storage.get_file_metadata(file_id)
            except Exception:
                metadata = None
            if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                logger.info("Skip updating file %s to PARTIAL_INDEXED because status is DELETED", file_id)
                return

            success = self.file_storage.metadata_store.update_file_metadata(
                file_id,
                {"status": FileStatus.PARTIAL_INDEXED},
                **kwargs,
            )
            if success:
                logger.info("Updated file %s status to PARTIAL_INDEXED", file_id)
            else:
                logger.warning("Failed to update file %s status to PARTIAL_INDEXED", file_id)
        except Exception as e:
            logger.error("Error updating file %s status: %s", file_id, e)

    def _update_file_status_to_parsed(self, file_id: str, **kwargs: Any) -> None:
        from encapsulation.data_model.orm_models import FileStatus

        try:
            try:
                metadata = self.file_storage.get_file_metadata(file_id)
            except Exception:
                metadata = None
            if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                logger.info("Skip updating file %s to PARSED because status is DELETED", file_id)
                return
            success = self.file_storage.metadata_store.update_file_metadata(
                file_id,
                {"status": FileStatus.PARSED},
                **kwargs,
            )
            if success:
                logger.info("Updated file %s status to PARSED", file_id)
            else:
                logger.warning("Failed to update file %s status to PARSED", file_id)
        except Exception as e:
            logger.error("Error updating file %s status to PARSED: %s", file_id, e)

    def _update_file_status_to_chunked(self, file_id: str, **kwargs: Any) -> None:
        from encapsulation.data_model.orm_models import FileStatus

        try:
            try:
                metadata = self.file_storage.get_file_metadata(file_id)
            except Exception:
                metadata = None
            if metadata is not None and getattr(metadata, "status", None) == FileStatus.DELETED:
                logger.info("Skip updating file %s to CHUNKED because status is DELETED", file_id)
                return
            success = self.file_storage.metadata_store.update_file_metadata(
                file_id,
                {"status": FileStatus.CHUNKED},
                **kwargs,
            )
            if success:
                logger.info("Updated file %s status to CHUNKED", file_id)
            else:
                logger.warning("Failed to update file %s status to CHUNKED", file_id)
        except Exception as e:
            logger.error("Error updating file %s status to CHUNKED: %s", file_id, e)

    def _update_parsed_content_status_to_chunked(self, parsed_content_id: str, **kwargs: Any) -> None:
        """
        Update parsed content metadata status to CHUNKED after successful chunking.

        Args:
            parsed_content_id: Parsed content ID to update
            **kwargs: Additional arguments
        """
        from encapsulation.data_model.orm_models import ParsedContentStatus

        try:
            success = self.parsed_content_storage.metadata_store.update_parsed_content_metadata(
                parsed_content_id,
                {"status": ParsedContentStatus.CHUNKED},
                **kwargs,
            )
            if success:
                logger.info(f"Updated parsed content {parsed_content_id} status to CHUNKED")
            else:
                logger.warning(f"Failed to update parsed content {parsed_content_id} status to CHUNKED")
        except Exception as e:
            logger.error(f"Error updating parsed content {parsed_content_id} status: {e}")

