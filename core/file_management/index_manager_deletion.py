import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class _IndexManagerDeletionMixin:
    def _delete_chunks_from_indexers(self, chunk_ids: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        Delete chunks from all configured indexers using ThreadPoolExecutor.

        Args:
            chunk_ids: List of chunk IDs to delete
            **kwargs: Optional deletion context propagated to indexers (e.g., `owner_id`).

        Returns:
            Dictionary with deletion results for each indexer
        """
        deletion_results = {}

        # Use ThreadPoolExecutor for concurrent deletion
        with ThreadPoolExecutor(max_workers=len(self.indexers)) as executor:
            future_to_indexer = {}

            for i, indexer in enumerate(self.indexers):
                indexer_name = f"{type(indexer).__name__}_{i}"
                future = executor.submit(self._delete_with_single_indexer_sync, indexer, indexer_name, chunk_ids, **kwargs)
                future_to_indexer[future] = (indexer, indexer_name)

            # Collect results
            for future in as_completed(future_to_indexer):
                indexer, indexer_name = future_to_indexer[future]

                try:
                    result = future.result()
                    deletion_results[indexer_name] = result
                    if result.get("success"):
                        logger.info(f"Successfully deleted {len(chunk_ids)} chunks from {indexer_name}")
                except Exception as e:
                    logger.error(f"Deletion failed with {indexer_name}: {str(e)}")
                    deletion_results[indexer_name] = {
                        "success": False,
                        "error_message": str(e),
                        "deleted_count": 0,
                        "total_chunks": len(chunk_ids),
                    }

        return deletion_results

    def _delete_with_single_indexer_sync(self, indexer, indexer_name: str, chunk_ids: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        Delete chunks with a single indexer (synchronous).

        This is a helper method that allows concurrent execution of multiple indexers.

        Args:
            indexer: The indexer instance
            indexer_name: Name of the indexer for logging
            chunk_ids: List of chunk IDs to delete

        Returns:
            Dictionary with deletion result for this indexer
        """
        try:
            logger.info(f"Deleting {len(chunk_ids)} chunks from {indexer_name}")
            success = indexer.delete_chunks(chunk_ids, **kwargs)

            if success:
                logger.info(f"Successfully deleted {len(chunk_ids)} chunks from {indexer_name}")
            else:
                logger.warning(f"Deletion returned False for {indexer_name}")

            return {"success": bool(success), "deleted_count": len(chunk_ids) if success else 0, "total_chunks": len(chunk_ids)}
        except Exception as e:
            logger.error(f"Deletion failed with {indexer_name}: {str(e)}", exc_info=True)
            return {"success": False, "error_message": str(e), "deleted_count": 0, "total_chunks": len(chunk_ids)}

    def delete_file(self, file_id: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Synchronous method for deleting a file and all its associated data.
        This is the main entry point for external usage.

        Args:
            file_id: The ID of the file to delete
            **kwargs: Additional arguments

        Returns:
            Dict containing deletion results
        """
        # Validate file_id
        if file_id is None or not isinstance(file_id, str) or not file_id.strip():
            error_msg = "file_id must be a non-empty string"
            logger.error(error_msg)
            return {"success": False, "file_id": file_id, "error_message": error_msg}

        return self.delete_file_data(file_id, **kwargs)

    def delete_file_data(self, file_id: str, *, preserve_parsed_content: bool = False, **kwargs: Any) -> Dict[str, Any]:
        """
        Delete a file and all its associated data (parsed content, chunks and index entries).

        This method:
        1. Finds all parsed content associated with the file
        2. Finds all chunks associated with each parsed content
        3. Deletes chunks from all configured indexers (FIRST - not covered by CASCADE)
        4. Deletes chunk blobs (SECOND - not covered by CASCADE)
        5. Deletes parsed content blobs (THIRD - not covered by CASCADE)

        Note:
        - Chunk metadata and parsed content metadata are automatically deleted via CASCADE
          when parsed_content_metadata is deleted (which happens when file_metadata is deleted)
        - However, indexer data and blob files must be manually deleted as they are not
          managed by the database

        Args:
            file_id: ID of the file to delete
            preserve_parsed_content: When true, delete only derived artifacts (chunks + index entries) and keep parsed
                content blobs/metadata intact so subsequent indexing can reuse them and skip parsing.
            **kwargs: Additional arguments

        Returns:
            Dictionary containing deletion results:
            - success: bool - Whether the entire deletion succeeded
            - file_id: str - Input file ID
            - deleted_parsed_content_ids: List[str] - IDs of deleted parsed content
            - deleted_chunk_ids: List[str] - IDs of deleted chunks
            - indexer_deletion_results: Dict - Results from each indexer
            - error_message: str - Error message if failed
        """
        result = {
            "success": False,
            "file_id": file_id,
            "deleted_parsed_content_ids": [],
            "deleted_chunk_ids": [],
            "indexer_deletion_results": {},
            "error_message": None,
        }

        try:
            logger.info(f"Starting deletion pipeline for file_id: {file_id}")

            # Step 1: Find all parsed content for this file
            logger.info(f"Step 1: Finding parsed content for file {file_id}")
            parsed_content_list = self.parsed_content_storage.metadata_store.list_parsed_content_metadata(
                source_file_id=file_id,
                **kwargs,
            )

            if not parsed_content_list:
                logger.warning(f"No parsed content found for file_id: {file_id}")
                result["success"] = True
                return result

            logger.info(f"Found {len(parsed_content_list)} parsed content entries")

            # Resolve owner_id for owner-scoped indices (FAISS/BM25). Prefer explicit kwargs, then metadata.
            owner_id = kwargs.get("owner_id")
            if not owner_id:
                try:
                    # ParsedContentMetadata usually carries owner_id.
                    owner_id = getattr(parsed_content_list[0], "owner_id", None)
                except Exception:
                    owner_id = None
            if not owner_id:
                try:
                    file_meta = self.file_storage.get_file_metadata(file_id, **kwargs)  # type: ignore[attr-defined]
                    owner_id = getattr(file_meta, "owner_id", None) if file_meta is not None else None
                except Exception:
                    owner_id = None

            # Step 2: For each parsed content, find all chunks
            all_chunk_ids = []
            for parsed_content in parsed_content_list:
                if not parsed_content or not hasattr(parsed_content, "parsed_content_id"):
                    logger.warning("Invalid parsed_content, skipping")
                    continue

                parsed_content_id = parsed_content.parsed_content_id
                if not parsed_content_id:
                    logger.warning("parsed_content has empty parsed_content_id, skipping")
                    continue

                # Find all chunks for this parsed content
                logger.info(f"Step 2: Finding chunks for parsed_content_id: {parsed_content_id}")
                chunk_list = self.chunk_storage.metadata_store.list_chunk_metadata(
                    source_parsed_content_id=parsed_content_id,
                    **kwargs,
                )

                if chunk_list:
                    chunk_ids = [
                        chunk.chunk_id
                        for chunk in chunk_list
                        if chunk and hasattr(chunk, "chunk_id") and chunk.chunk_id
                    ]
                    all_chunk_ids.extend(chunk_ids)
                    logger.info(f"Found {len(chunk_ids)} chunks for parsed_content {parsed_content_id}")

            # Step 3: Delete chunks from all indexers FIRST
            # REQUIRED: Indexer data (FAISS, BM25) is NOT managed by database CASCADE
            if all_chunk_ids and self.indexers:
                logger.info(f"Step 3: Deleting {len(all_chunk_ids)} chunks from {len(self.indexers)} indexers")
                delete_ctx = dict(kwargs)
                if owner_id and not delete_ctx.get("owner_id"):
                    delete_ctx["owner_id"] = owner_id
                deletion_results = self._delete_chunks_from_indexers(all_chunk_ids, **delete_ctx)
                result["indexer_deletion_results"] = deletion_results

                # Check if indexer deletion was successful
                all_indexers_successful = all(res.get("success", False) for res in deletion_results.values())
                if not all_indexers_successful:
                    error_msg = "Failed to delete chunks from some indexers"
                    logger.warning(error_msg)
                    # Log which indexers failed
                    failed_indexers = [name for name, res in deletion_results.items() if not res.get("success", False)]
                    logger.warning(f"Failed indexers: {failed_indexers}")
                    # Abort further deletion to keep metadata for retry
                    result["error_message"] = error_msg
                    return result
                else:
                    logger.info("Successfully deleted chunks from all indexers")
            else:
                logger.info("Step 3: No chunks to delete from indexers")

            # Step 4: Delete chunk blobs (SECOND, after indexer deletion succeeds)
            # REQUIRED: Blob files are NOT managed by database CASCADE
            # NOTE: Chunk metadata will be automatically deleted via CASCADE when parsed_content_metadata is deleted
            if all_chunk_ids:
                logger.info(f"Step 4: Deleting {len(all_chunk_ids)} chunk blobs")
                for chunk_id in all_chunk_ids:
                    if not chunk_id:
                        continue

                    try:
                        # Delete chunk blob (REQUIRED - not covered by CASCADE)
                        chunk_metadata = self.chunk_storage.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
                        if chunk_metadata and hasattr(chunk_metadata, "blob_key") and chunk_metadata.blob_key:
                            self.chunk_storage.blob_store.delete(chunk_metadata.blob_key, **kwargs)
                            logger.debug(f"Deleted chunk blob: {chunk_metadata.blob_key}")

                        # Delete chunk metadata (will also be deleted by CASCADE, but doing it explicitly for clarity)
                        self.chunk_storage.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                        logger.debug(f"Deleted chunk metadata: {chunk_id}")
                        result["deleted_chunk_ids"].append(chunk_id)
                    except Exception as e:
                        logger.error(f"Failed to delete chunk {chunk_id}: {e}")

                logger.info(f"Successfully deleted {len(result['deleted_chunk_ids'])}/{len(all_chunk_ids)} chunk blobs")

            # Step 5: Delete parsed content blobs (THIRD, after chunks are deleted)
            # REQUIRED: Blob files are NOT managed by database CASCADE
            # NOTE: Parsed content metadata will be automatically deleted via CASCADE when file_metadata is deleted
            if preserve_parsed_content:
                logger.info("Step 5: Skipping parsed content deletion (preserve_parsed_content=1)")
            else:
                logger.info(f"Step 5: Deleting {len(parsed_content_list)} parsed content blobs")
            for parsed_content in parsed_content_list:
                if not parsed_content or not hasattr(parsed_content, "parsed_content_id"):
                    continue

                parsed_content_id = parsed_content.parsed_content_id
                if not parsed_content_id:
                    continue

                if preserve_parsed_content:
                    continue

                try:
                    # Delete parsed content blob (REQUIRED - not covered by CASCADE)
                    if hasattr(parsed_content, "blob_key") and parsed_content.blob_key:
                        self.parsed_content_storage.blob_store.delete(parsed_content.blob_key, **kwargs)
                        logger.debug(f"Deleted parsed content blob: {parsed_content.blob_key}")

                    # Delete parsed content metadata (will also be deleted by CASCADE, but doing it explicitly for clarity)
                    self.parsed_content_storage.metadata_store.delete_parsed_content_metadata(parsed_content_id, **kwargs)
                    logger.debug(f"Deleted parsed content metadata: {parsed_content_id}")
                    result["deleted_parsed_content_ids"].append(parsed_content_id)
                except Exception as e:
                    logger.error(f"Failed to delete parsed content {parsed_content_id}: {e}")

            if not preserve_parsed_content:
                logger.info(
                    f"Successfully deleted {len(result['deleted_parsed_content_ids'])}/{len(parsed_content_list)} parsed content blobs"
                )

            if result["error_message"]:
                logger.warning(f"Deletion pipeline for file_id {file_id} completed with errors: {result['error_message']}")
            else:
                result["success"] = True
                logger.info(f"Successfully completed deletion pipeline for file_id: {file_id}")

        except Exception as e:
            error_msg = f"Deletion pipeline failed for file_id {file_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg

        return result
