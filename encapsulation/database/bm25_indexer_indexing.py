import gc
import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from encapsulation.data_model.schema import Chunk
from encapsulation.database.bm25_indexer_constants import EXCLUDED_METADATA_FIELDS
from encapsulation.database.bm25_indexer_tantivy import TantivyDocument

logger = logging.getLogger(__name__)


class _BM25IndexBuilderIndexingMixin:
    def _writer_worker(self, writer) -> None:
        """Consumer thread: index writing worker

        Args:
            writer: Tantivy index writer
        """
        batch_docs = []
        while not self.stop_event.is_set() or not self.processing_queue.empty():
            try:
                doc = self.processing_queue.get(timeout=1)
                if doc is None:
                    break
                batch_docs.append(doc)
                if len(batch_docs) >= self.config.batch_size:
                    self._batch_write_documents(batch_docs, writer)
                    batch_docs.clear()
                    # Trigger garbage collection if enabled
                    if self.config.enable_gc:
                        gc.collect()
            except queue.Empty:
                continue

        if batch_docs:
            self._batch_write_documents(batch_docs, writer)

    def _batch_write_documents(self, docs: List[TantivyDocument], writer) -> None:
        """Write a batch of tantivy documents to the index

        Args:
            docs: List of Tantivy documents to write
            writer: Tantivy index writer
        """
        try:
            writer.add_documents(docs)
        except AttributeError:
            for d in docs:
                writer.add_document(d)
        except Exception as e:
            logger.error(f"Error writing batch of tantivy documents: {e}")
            raise

    def _delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        """Delete chunks by their IDs with retry mechanism for lock conflicts

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks actually deleted
        """
        if not chunk_ids:
            return 0

        # Defensive: Tantivy-backed indices require `meta.json` on disk. If it's missing (e.g. half-created directory
        # with only `.tantivy-*.lock` files), treat the index as empty and let callers rebuild it.
        meta_path = os.path.join(self.config.index_path, "meta.json")
        if self._index is not None and not os.path.exists(meta_path):
            logger.warning("BM25 index missing meta.json; resetting in-memory index and skipping delete: %s", meta_path)
            self._index = None
            self._tokenizers_registered = False
            return 0

        # Try to load index if not loaded
        if self._index is None:
            try:
                logger.info("Index not loaded, attempting to load from disk for deletion")
                self.load_local()
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"Cannot load index for deletion: {e}")
                # Index doesn't exist, nothing to delete
                return 0

        # Retry mechanism for lock conflicts
        max_retries = 3
        retry_delay = 0.1  # seconds

        for attempt in range(max_retries):
            try:
                # First, check which chunks actually exist
                searcher = self._index.searcher()
                existing_ids = []

                for chunk_id in chunk_ids:
                    query = self._index.parse_query(f'id:"{chunk_id}"', ["id"])
                    results = searcher.search(query, 1)
                    logger.info(f"Checking chunk {chunk_id}: found {len(results.hits)} hits")
                    if results.hits:
                        existing_ids.append(chunk_id)

                if not existing_ids:
                    logger.info("No chunks found to delete")
                    return 0

                # Delete only existing chunks
                writer = self._index.writer(heap_size=self.writer_heap_size)
                deleted_count = 0

                for chunk_id in existing_ids:
                    # Use delete_documents_by_term for exact matching (id field has tokenizer="raw")
                    writer.delete_documents_by_term("id", chunk_id)
                    logger.info(f"Deleted chunk {chunk_id}")
                    deleted_count += 1

                logger.info(f"Committing deletion of {deleted_count} chunks")
                writer.commit()
                logger.info("Reloading index after deletion")
                self._index.reload()
                logger.info(f"Successfully deleted {deleted_count} chunks from index (requested: {len(chunk_ids)})")
                return deleted_count

            except Exception as e:
                error_msg = str(e)
                # Check if it's a lock conflict
                if "LockBusy" in error_msg or "Failed to acquire" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Lock conflict on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Failed to acquire lock after {max_retries} attempts")
                        raise
                else:
                    # Other errors, don't retry
                    logger.error(f"Error deleting chunks: {e}")
                    raise

        return 0

    def _build_index(self, chunks: List[Chunk]) -> List[str]:
        """Build index using producer-consumer pattern

        Args:
            chunks: List of Chunk objects to index

        Returns:
            List of chunk IDs that were added to the index

        Raises:
            RuntimeError: If there's an error during index building
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return []

        if self.tokenizer_manager.custom_preprocess_func is None:
            self._set_tokenizer(chunks)

        # For new indices, reinitialize with dynamic fields based on chunks.
        # Tantivy indices require `meta.json`; lock files alone should not be treated as an existing index.
        meta_path = os.path.join(self.config.index_path, "meta.json")
        index_exists = os.path.exists(meta_path)

        if not index_exists:
            self._index = None  # Reset to force reinitialization with dynamic fields
            self._initialize_index(chunks)

        if self._index is None:
            raise RuntimeError("Index has not been initialized")

        total_docs = len(chunks)
        added_ids, processed_count = [], 0

        # Ensure any previous writer thread is stopped
        if self._writer_thread and self._writer_thread.is_alive():
            self.stop_event.set()
            try:
                self._writer_thread.join(timeout=5.0)
            except Exception:
                pass

        # Clear the queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break

        writer = self._index.writer(heap_size=self.writer_heap_size)

        # Reset stop event for this build operation
        self.stop_event.clear()

        self._writer_thread = threading.Thread(target=self._writer_worker, args=(writer,))
        self._writer_thread.start()

        start_time = time.time()

        try:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", None) or {}
                index_text = metadata.get("index_text")
                token_source = chunk.content or ""
                if isinstance(index_text, str) and index_text.strip():
                    token_source = index_text
                # Optional: inject file/path context into BM25 token text to disambiguate
                # similar products across companies (configured via BM25BuilderConfig).
                try:
                    from encapsulation.database.utils.embedding_text import build_embedding_text, build_prefix_keys

                    cfg = getattr(self, "config", None)
                    prefix_keys = build_prefix_keys(getattr(cfg, "token_text_prefix_keys", None))
                    filename_root = getattr(cfg, "token_text_filename_root", None)
                    sep = getattr(cfg, "token_text_separator", "\n")
                    token_source = build_embedding_text(
                        base_text=str(token_source),
                        metadata=metadata,
                        prefix_keys=prefix_keys,
                        filename_root=filename_root,
                        separator=sep,
                    ) or token_source
                except Exception:
                    pass
                content_tokens = self.tokenizer_manager.get_current_tokenizer()(token_source)
                chunk_id = str(chunk.id) if chunk.id else str(uuid.uuid4())

                tantivy_doc = TantivyDocument()
                tantivy_doc.add_text("id", chunk_id)
                tantivy_doc.add_text("content", chunk.content or "")
                tantivy_doc.add_text("content_tokens", " ".join(content_tokens))

                # Add owner_id for user isolation
                owner_id = chunk.owner_id if hasattr(chunk, "owner_id") and chunk.owner_id else ""
                tantivy_doc.add_text("owner_id", owner_id)

                metadata = chunk.metadata or {}
                tantivy_doc.add_json("metadata", metadata)

                # Dynamically add all string fields from metadata for filtering
                for key, value in metadata.items():
                    if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                        try:
                            tantivy_doc.add_text(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to add field '{key}' to tantivy document: {e}")

                self.processing_queue.put(tantivy_doc)

                added_ids.append(chunk_id)
                processed_count += 1

                if processed_count % self.config.progress_interval == 0:
                    elapsed = time.time() - start_time
                    stats = {
                        "processed": processed_count,
                        "total": total_docs,
                        "elapsed_sec": round(elapsed, 2),
                        "throughput_docs_sec": round(processed_count / elapsed, 2),
                    }
                    logger.info(f"[IndexProgress] {json.dumps(stats, ensure_ascii=False)}")

            # Final progress logging if not already logged at the end
            if processed_count % self.config.progress_interval != 0:
                elapsed = time.time() - start_time
                stats = {
                    "processed": processed_count,
                    "total": total_docs,
                    "elapsed_sec": round(elapsed, 2),
                    "throughput_docs_sec": round(processed_count / elapsed, 2),
                }
                logger.info(f"[IndexProgress] Final: {json.dumps(stats, ensure_ascii=False)}")

            self.processing_queue.put(None)
            self._writer_thread.join()
            writer.commit()
            self._index.reload()

            tokenizer_info = self.tokenizer_manager.get_tokenizer_info()
            logger.info(f"Successfully built index with {len(added_ids)} chunks using {tokenizer_info} tokenizer")

        except Exception as e:
            logger.error(f"Error building index: {e}")
            try:
                writer.rollback()
            except Exception:
                pass
            raise
        finally:
            self.stop_event.set()
            if self.config.enable_gc:
                gc.collect()

        return added_ids

    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Add chunks to the existing index (with ID deduplication)

        Args:
            chunks: List of Chunk objects to add
        Returns:
            List of chunk IDs that were successfully added to the index
        """
        if not chunks:
            logger.warning("No chunks provided for adding")
            return []

        # Check for duplicate IDs and filter
        unique_chunks = []
        duplicate_ids = []
        existing_ids = set()

        # Get existing chunk IDs - directly query from index
        existing_ids = set()
        if self._index is not None:
            try:
                searcher = self._index.searcher()
                # Query all chunk IDs (this can be optimized, but for simplicity, just query directly)
                for chunk in chunks:
                    query = self._index.parse_query(f'id:"{chunk.id}"', ["id"])
                    results = searcher.search(query, 1)
                    if results.hits:
                        existing_ids.add(chunk.id)
                logger.debug(f"Found {len(existing_ids)} existing chunk IDs in index")
            except Exception as e:
                logger.debug(f"Error getting existing chunk IDs: {e}")
                existing_ids = set()

        # Check for duplicate IDs in the list (including with existing chunks)
        seen_ids = set()
        for chunk in chunks:
            if chunk.id in seen_ids:
                # Duplicate within the list
                duplicate_ids.append(chunk.id)
                logger.warning(f"Duplicate chunk ID found: {chunk.id}. Use update_chunks() to update existing chunks.")
            elif chunk.id in existing_ids:
                # Duplicate with existing chunks
                duplicate_ids.append(chunk.id)
                logger.warning(f"Chunk with ID {chunk.id} already exists. Use update() to update existing chunks.")
            else:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)

        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate chunk IDs: {duplicate_ids}")

        if not unique_chunks:
            logger.warning("No unique chunks to add after deduplication")
            return []

        return self._build_index(unique_chunks)

    def save_index(self, index_path: str = None, index_name: str = "index") -> None:
        """Save index state (Tantivy automatically persists)

        Note: Tantivy automatically persists the index, so this method only logs the save location.
        The index_path and index_name parameters are ignored for compatibility.
        """
        self._ensure_index_loaded()
        logger.info(f"Index automatically saved at: {self.config.index_path}")

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build index from chunks (only when index doesn't exist)

        Args:
            chunks: List of Chunk objects to index

        Raises:
            RuntimeError: If index already exists
        """
        # Check if index already exists
        if self.index_exists():
            raise RuntimeError(
                "Index already exists. Use add() to add chunks to existing index, "
                "or delete the existing index first if you want to rebuild it.",
            )

        self.from_chunks(chunks)

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update chunks in index

        Args:
            chunks: List of Chunk objects to update

        Returns:
            Optional[bool]: True if update successful, False otherwise, None if not implemented
        """
        if not chunks:
            return True

        try:
            chunk_ids = [str(chunk.id) for chunk in chunks if chunk.id is not None]
            if chunk_ids:
                logger.info(f"Update mode: attempting to delete {len(chunk_ids)} existing chunks")
                deleted_count = self._delete_chunks_by_ids(chunk_ids)
                logger.info(f"Update mode: successfully deleted {deleted_count} chunks")

                # Ensure index is reloaded after deletion for consistency
                if self._index is not None:
                    self._index.reload()

            self._build_index(chunks)
            return True
        except Exception as e:
            logger.error(f"Error updating chunks: {e}")
            return False

    def delete_index(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete chunks by IDs

        Args:
            ids: List of IDs to delete. If None, delete all. Default is None

        Returns:
            Optional[bool]: True if deletion successful, False otherwise, None if not implemented
        """
        if ids is None:
            try:
                if self._index is not None:
                    self._index = None
                    self._tokenizers_registered = False

                self._initialize_index()
                logger.info("Successfully deleted all chunks by recreating index")
                return True
            except Exception as e:
                logger.error(f"Error deleting all chunks: {e}")
                return False

        if not ids:
            return True

        unique_chunk_ids = list(set(ids))

        try:
            deleted_count = self._delete_chunks_by_ids(unique_chunk_ids)
            # Idempotency: treat missing IDs as already deleted.
            if deleted_count <= 0:
                logger.info("No chunks found to delete")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
