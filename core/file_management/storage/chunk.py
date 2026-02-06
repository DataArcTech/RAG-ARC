from typing import (
    Any,
    Optional,
    TYPE_CHECKING,
)
from datetime import datetime
from zoneinfo import ZoneInfo

import json
import uuid
import concurrent.futures
from encapsulation.data_model.orm_models import ChunkMetadata
from encapsulation.data_model.orm_models import ChunkIndexStatus

from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.storage.chunk_storage import ChunkStorageConfig

import logging

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails"""
    pass


class StorageOperationError(Exception):
    """Raised when storage operation fails"""
    pass


class ChunkStorage(AbstractModule):
    """
    Core chunk storage interface for RAG system.

    Provides high-level chunk storage operations with coordination
    between blob storage and metadata storage.

    Key features:
    - Chunk storage linked to parsed content
    - Automatic cleanup on validation failures
    - Comprehensive error handling and reporting

    Architecture:
        Application Layer -> ChunkStorage (Core) -> Blob Storage + Metadata Storage

    Dependencies:
        blob_store: FileDB implementation (e.g., LocalDB, MinIODB)
        metadata_store: RelationalDB implementation (e.g., PostgreSQLDB)
    """

    def __init__(self, config):
        """Initialize ChunkStorage with eager blob and metadata store creation"""
        super().__init__(config)
        # Build stores directly (no intermediate data_store layer)
        self.blob_store = config.file_db_config.build()
        self.metadata_store = config.relational_db_config.build()

    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        return str(uuid.uuid4())

    def _generate_chunk_blob_key(self, chunk_id: str, source_parsed_content_id: str, chunker_type: str) -> str:
        """Generate blob storage key for chunk"""
        # Create hierarchical key: chunks/{first-2-chars-of-source-id}/{source-parsed-content-id}/{chunk-id}.json
        prefix = source_parsed_content_id[:2]
        return f"chunks/{prefix}/{source_parsed_content_id}/{chunk_id}.json"

    def _validate_stored_chunk(self, chunk_id: str, **kwargs: Any) -> bool:
        """Validate that chunk was stored successfully"""
        try:
            # Get chunk metadata
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for validation: {chunk_id}")
                return False

            # Check if blob exists
            blob_exists = self.blob_store.exists(metadata.blob_key, **kwargs)
            if not blob_exists:
                logger.error(f"Chunk blob validation failed - blob not found: {metadata.blob_key}")
                return False

            logger.info(f"Chunk upload validation successful: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate chunk upload {chunk_id}: {e}")
            return False

    def _cleanup_failed_chunk_upload(self, chunk_id: str, **kwargs: Any) -> bool:
        """Clean up artifacts from failed chunk upload"""
        try:
            cleanup_success = True

            # Get metadata to find blob key
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if metadata:
                # Try to delete blob
                try:
                    if self.blob_store.exists(metadata.blob_key, **kwargs):
                        blob_deleted = self.blob_store.delete(metadata.blob_key, **kwargs)
                        if not blob_deleted:
                            logger.warning(f"Failed to delete chunk blob during cleanup: {metadata.blob_key}")
                            cleanup_success = False
                        else:
                            logger.info(f"Deleted chunk blob during cleanup: {metadata.blob_key}")
                except Exception as blob_error:
                    logger.error(f"Error deleting chunk blob during cleanup: {blob_error}")
                    cleanup_success = False

            # Delete metadata
            try:
                metadata_deleted = self.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                if not metadata_deleted:
                    logger.warning(f"Failed to delete chunk metadata during cleanup: {chunk_id}")
                    cleanup_success = False
                else:
                    logger.info(f"Deleted chunk metadata during cleanup: {chunk_id}")
            except Exception as metadata_error:
                logger.error(f"Error deleting chunk metadata during cleanup: {metadata_error}")
                cleanup_success = False

            return cleanup_success

        except Exception as e:
            logger.error(f"Failed to cleanup failed chunk upload {chunk_id}: {e}")
            return False

    def store_chunk(
        self,
        source_parsed_content_id: str,
        chunker_type: str,
        chunk_data: bytes,
        chunk_index: int,
        owner_id: uuid.UUID = None,
        validate_after_store: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Store a single chunk linked to parsed content with coordination.

        Args:
            source_parsed_content_id: ID of the parsed content that was chunked
            chunker_type: Type of chunker used (e.g., "semantic_chunker", "token_chunker")
            chunk_data: Binary chunk data (JSON format)
            chunk_index: Index of the chunk within the parsed content (0-based)
            owner_id: UUID of the user who owns this chunk (for retrieval filtering)
            validate_after_store: Whether to validate after storing
            **kwargs: Additional arguments

        Returns:
            str: Chunk ID of the stored chunk

        Raises:
            FileValidationError: If validation fails
            StorageOperationError: If storage operation fails
        """
        # Validate parameters
        if not chunk_data:
            raise FileValidationError("chunk_data must be provided")

        try:
            # Verify source parsed content exists
            source_metadata = self.metadata_store.get_parsed_content_metadata(source_parsed_content_id, **kwargs)
            if not source_metadata:
                raise ValueError(f"Source parsed content {source_parsed_content_id} not found")

            # Get owner_id from source file if not provided
            if owner_id is None:
                # Get file metadata through parsed content -> file relationship
                file_metadata = self.metadata_store.get_file_metadata(source_metadata.source_file_id, **kwargs)
                if not file_metadata:
                    raise ValueError(f"Source file {source_metadata.source_file_id} not found")
                owner_id = file_metadata.owner_id

            # Generate IDs and keys
            chunk_id = self._generate_chunk_id()
            blob_key = self._generate_chunk_blob_key(chunk_id, source_parsed_content_id, chunker_type)

            # Create chunk metadata object
            now = datetime.now(tz=datetime.now().astimezone().tzinfo)
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_parsed_content_id=source_parsed_content_id,
                owner_id=owner_id,
                blob_key=blob_key,
                chunker_type=chunker_type,
                chunk_index=chunk_index,
                index_status=ChunkIndexStatus.STORED,
                created_at=now
            )

            # Store chunk metadata first
            logger.info(f"Storing chunk metadata: {chunk_id} (source: {source_parsed_content_id})")
            stored_metadata_id = self.metadata_store.store_chunk_metadata(chunk_metadata, **kwargs)
            assert stored_metadata_id == chunk_id

            try:
                # Store chunk blob
                logger.info(f"Storing chunk blob: {blob_key}")
                stored_blob_key, was_overwritten = self.blob_store.store(
                    blob_key,
                    chunk_data,
                    content_type="application/json",
                    **kwargs
                )

                # Update metadata with final blob key
                self.metadata_store.update_chunk_metadata(
                    chunk_id,
                    {
                        'blob_key': stored_blob_key
                    },
                    **kwargs
                )

                if was_overwritten:
                    logger.warning(f"Chunk blob was overwritten: {stored_blob_key}")

                logger.info(f"Successfully stored chunk: {chunk_id} (blob_key: {stored_blob_key})")

            except Exception as blob_error:
                # Blob storage failed, delete metadata
                logger.error(f"Chunk blob storage failed: {blob_error}")
                self.metadata_store.delete_chunk_metadata(chunk_id, **kwargs)
                raise StorageOperationError(f"Chunk blob storage failed: {str(blob_error)}")

            # Validate after storage if requested
            if validate_after_store:
                validation_passed = self._validate_stored_chunk(chunk_id, **kwargs)

                if not validation_passed:
                    # Validation failed, cleanup
                    logger.error(f"Chunk validation failed: {chunk_id}")
                    cleanup_success = self._cleanup_failed_chunk_upload(chunk_id, **kwargs)
                    if cleanup_success:
                        logger.info(f"Cleaned up failed chunk: {chunk_id}")
                    else:
                        logger.warning(f"Failed to cleanup after validation failure: {chunk_id}")

                    raise StorageOperationError("Chunk validation failed after storage")

                logger.info(f"Chunk validation passed: {chunk_id}")

            return chunk_id

        except FileValidationError:
            # Re-raise validation errors as-is
            raise

        except Exception as e:
            logger.error(f"Chunk storage error: {e}")
            raise StorageOperationError(f"Storage error: {str(e)}")

    def store_chunks_batch(
        self,
        source_parsed_content_id: str,
        chunker_type: str,
        chunks: list[dict],
        *,
        owner_id: uuid.UUID | None = None,
        validate_after_store: bool = False,
        blob_write_max_workers: int | None = None,
        blob_content_type: str = "application/json",
        **kwargs: Any,
    ) -> list[tuple[int, str]]:
        """
        Store multiple chunks in a batch.

        Key optimizations vs. `store_chunk` in a loop:
        - Resolve parsed-content/file owner once (no per-chunk DB lookups).
        - Batch insert chunk metadata in a single transaction when supported by the metadata store.
        - Avoid per-chunk threadpool transitions when called via a single run_blocking().

        Returns:
            List of (chunk_index, chunk_id) for successfully stored chunks.
        """
        if not chunks:
            return []

        # Verify source parsed content exists once.
        source_metadata = self.metadata_store.get_parsed_content_metadata(source_parsed_content_id, **kwargs)
        if not source_metadata:
            raise ValueError(f"Source parsed content {source_parsed_content_id} not found")

        # Resolve owner_id once.
        if owner_id is None:
            file_metadata = self.metadata_store.get_file_metadata(source_metadata.source_file_id, **kwargs)
            if not file_metadata:
                raise ValueError(f"Source file {source_metadata.source_file_id} not found")
            owner_id = file_metadata.owner_id

        now = datetime.now(tz=datetime.now().astimezone().tzinfo)

        batch_items: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if chunk is None:
                continue
            try:
                # Use errors='replace' to handle surrogate pairs and invalid Unicode characters.
                chunk_data = json.dumps(chunk, ensure_ascii=False).encode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to serialize chunk %s for %s: %s", i, source_parsed_content_id, exc)
                continue

            chunk_id = self._generate_chunk_id()
            blob_key = self._generate_chunk_blob_key(chunk_id, source_parsed_content_id, chunker_type)
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_parsed_content_id=source_parsed_content_id,
                owner_id=owner_id,
                blob_key=blob_key,
                chunker_type=chunker_type,
                chunk_index=i,
                index_status=ChunkIndexStatus.STORED,
                created_at=now,
            )
            batch_items.append(
                {
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "blob_key": blob_key,
                    "metadata": chunk_metadata,
                    "data": chunk_data,
                }
            )

        if not batch_items:
            return []

        # Store metadata in bulk when supported (single transaction).
        metas = [it["metadata"] for it in batch_items]
        try:
            store_batch = getattr(self.metadata_store, "store_chunk_metadata_batch", None)
            if callable(store_batch):
                store_batch(metas, **kwargs)
            else:
                for m in metas:
                    self.metadata_store.store_chunk_metadata(m, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise StorageOperationError(f"Chunk metadata batch store failed: {exc}") from exc

        stored: list[tuple[int, str]] = []
        failures: list[tuple[int, str, str]] = []

        # Parallelize blob writes (remote object stores benefit greatly). Keep DB updates/cleanup sequential.
        try:
            from config.core.file_management.chunk_ingest_defaults import (
                CHUNK_BATCH_BLOB_WRITE_MAX_WORKERS_DEFAULT,
            )

            max_workers = int(
                CHUNK_BATCH_BLOB_WRITE_MAX_WORKERS_DEFAULT
                if blob_write_max_workers is None
                else blob_write_max_workers
            )
        except Exception:
            max_workers = 8
        max_workers = max(1, max_workers)

        def _blob_store_one(item: dict[str, Any]) -> tuple[str, str, str, bool, str | None]:
            cid = str(item["chunk_id"])
            blob_key = str(item["blob_key"])
            try:
                stored_blob_key, was_overwritten = self.blob_store.store(
                    blob_key,
                    item["data"],
                    content_type=blob_content_type,
                    **kwargs,
                )
                return cid, blob_key, str(stored_blob_key), bool(was_overwritten), None
            except Exception as exc:  # noqa: BLE001
                return cid, blob_key, "", False, str(exc)

        blob_results: dict[str, tuple[str, str, bool, str | None]] = {}
        if max_workers <= 1 or len(batch_items) <= 1:
            for it in batch_items:
                cid, blob_key, stored_blob_key, was_overwritten, err = _blob_store_one(it)
                blob_results[cid] = (blob_key, stored_blob_key, was_overwritten, err)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_blob_store_one, it) for it in batch_items]
                for fut in concurrent.futures.as_completed(futs):
                    cid, blob_key, stored_blob_key, was_overwritten, err = fut.result()
                    blob_results[cid] = (blob_key, stored_blob_key, was_overwritten, err)

        for it in batch_items:
            cid = str(it["chunk_id"])
            cidx = int(it["chunk_index"])
            blob_key, stored_blob_key, was_overwritten, err = blob_results.get(cid, ("", "", False, "missing_blob_result"))
            if err:
                failures.append((cidx, cid, str(err)))
                try:
                    self.metadata_store.delete_chunk_metadata(cid, **kwargs)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    if blob_key and self.blob_store.exists(blob_key, **kwargs):
                        self.blob_store.delete(blob_key, **kwargs)
                except Exception:  # noqa: BLE001
                    pass
                continue

            if was_overwritten:
                logger.warning("Chunk blob was overwritten: %s", stored_blob_key)

            # Update metadata only when the backing store rewrites the key.
            if stored_blob_key and stored_blob_key != blob_key:
                try:
                    self.metadata_store.update_chunk_metadata(cid, {"blob_key": stored_blob_key}, **kwargs)
                except Exception as update_exc:  # noqa: BLE001
                    logger.warning("Failed to update chunk %s blob_key to %s: %s", cid, stored_blob_key, update_exc)

            if validate_after_store:
                try:
                    if not self.blob_store.exists(stored_blob_key or blob_key, **kwargs):
                        raise StorageOperationError(f"Chunk blob missing after store: {stored_blob_key or blob_key}")
                except Exception as validate_exc:  # noqa: BLE001
                    failures.append((cidx, cid, str(validate_exc)))
                    try:
                        self.metadata_store.delete_chunk_metadata(cid, **kwargs)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        key = stored_blob_key or blob_key
                        if key and self.blob_store.exists(key, **kwargs):
                            self.blob_store.delete(key, **kwargs)
                    except Exception:  # noqa: BLE001
                        pass
                    continue

            stored.append((cidx, cid))

        if failures:
            logger.warning(
                "Chunk batch store partial failures: %d failed / %d total (source_parsed_content_id=%s)",
                len(failures),
                len(batch_items),
                source_parsed_content_id,
            )
            for cidx, cid, msg in failures[:10]:
                logger.warning("Chunk batch failure chunk_index=%s chunk_id=%s err=%s", cidx, cid, msg)

        if not stored:
            raise StorageOperationError("Failed to store any chunks in batch")
        return stored

    def get_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> Optional['ChunkMetadata']:
        """Retrieve chunk metadata by ID"""
        try:
            return self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get chunk metadata for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk metadata: {e}")

    def get_chunk_content(self, chunk_id: str, **kwargs: Any) -> Optional[bytes]:
        """Retrieve chunk content by ID"""
        try:
            # Get metadata to find blob key
            metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
            if not metadata:
                logger.warning(f"Chunk metadata not found for id: {chunk_id}")
                return None

            # Retrieve blob content
            try:
                content = self.blob_store.retrieve(metadata.blob_key, **kwargs)
                logger.debug(f"Retrieved chunk content for id: {chunk_id}")
                return content
            except KeyError:
                logger.error(f"Chunk blob not found for id: {chunk_id}, blob_key: {metadata.blob_key}")
                return None

        except Exception as e:
            logger.error(f"Failed to get chunk content for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to retrieve chunk content: {e}")

    def overwrite_chunk_json(self, chunk_id: str, chunk: dict, **kwargs: Any) -> bool:
        """
        Overwrite the stored chunk JSON blob for an existing chunk_id.

        Indexers see the in-memory chunk dict, but later "fetch chunk JSON by id" flows
        rely on blob storage. This method is used to persist post-store metadata fixes
        (e.g., backfilled `anchor_chunk_id`) back into the stored chunk blob.
        """
        if not chunk_id or not str(chunk_id).strip():
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(chunk, dict):
            raise ValueError("chunk must be a dict")

        metadata = self.metadata_store.get_chunk_metadata(chunk_id, **kwargs)
        if not metadata:
            logger.warning("Chunk metadata not found for overwrite: %s", chunk_id)
            return False

        try:
            payload = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to serialize chunk %s: %s", chunk_id, exc)
            return False

        try:
            self.blob_store.store(
                metadata.blob_key,
                payload,
                content_type="application/json",
                **kwargs,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to overwrite chunk blob for %s: %s", chunk_id, exc)
            return False

    def update_chunk_metadata(self, chunk_id: str, **kwargs: Any) -> bool:
        """Update chunk metadata by ID"""
        try:
            result = self.metadata_store.update_chunk_metadata(chunk_id, kwargs, **kwargs)
            if result:
                logger.info(f"Updated chunk metadata: {chunk_id}")
            else:
                logger.warning(f"Failed to update chunk metadata: {chunk_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to update chunk metadata for {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to update chunk metadata: {e}")

    def delete_chunk(self, chunk_id: str, **kwargs: Any) -> bool:
        """Delete chunk and cleanup associated data"""
        try:
            return self._cleanup_failed_chunk_upload(chunk_id, **kwargs)
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise StorageOperationError(f"Failed to delete chunk: {e}")
