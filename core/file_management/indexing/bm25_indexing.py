import asyncio
import logging
from typing import List, TYPE_CHECKING
from collections import deque
import threading

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig

logger = logging.getLogger(__name__)


class BM25Indexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using BM25.
    Supports batch processing with periodic flushing to avoid lock conflicts.
    """

    def __init__(self, config: "BM25IndexerConfig"):
        """
        Initializes the BM25 indexer and its specific builder.
        """
        super().__init__(config)
        self.bm25_builder = config.index_config.build()

        # Batch processing configuration
        self.batch_size = config.batch_size
        self.flush_interval = config.flush_interval

        # Async lock to ensure only one coroutine writes to the index
        self._write_lock = asyncio.Lock()

        # Thread-safe guard for pending queue (needed because delete runs in executor threads)
        self._pending_lock = threading.Lock()

        # Pending chunks queue for batch processing
        self._pending_chunks: deque[Chunk] = deque()

    async def _flush_pending_chunks(self) -> List[str]:
        """Flush all pending chunks to the index."""
        with self._pending_lock:
            if not self._pending_chunks:
                return []
            chunks_to_index = list(self._pending_chunks)
            self._pending_chunks.clear()

        if not chunks_to_index:
            return []

        # Acquire lock to ensure exclusive write access
        async with self._write_lock:

            logger.info(f"Flushing {len(chunks_to_index)} chunks to BM25 index")

            # Perform the actual indexing in a thread pool
            loop = asyncio.get_running_loop()
            chunk_ids = await loop.run_in_executor(
                None,
                self._build_or_update_index_sync,
                chunks_to_index
            )

            logger.info(f"Successfully flushed {len(chunk_ids)} chunks")
            return chunk_ids

    def _build_or_update_index_sync(self, chunks_list: List[Chunk]) -> List[str]:
        """Synchronous method to build or update index (runs in thread pool)."""
        try:
            # Check if index is already loaded in memory
            if self.bm25_builder._index is not None:
                # Index is loaded, use update_index to update existing chunks
                logger.info(f"Index already loaded, updating {len(chunks_list)} chunks")
                result = self.bm25_builder.update_index(chunks_list)
                if result is not True:
                    raise RuntimeError("BM25 builder update_index returned False")
                return [chunk.id for chunk in chunks_list]

            # Index not loaded, try to load existing index or create new one
            try:
                # Try to load existing index from disk
                self.bm25_builder.load_local()
                logger.info(f"Loaded existing index, updating {len(chunks_list)} chunks")
                result = self.bm25_builder.update_index(chunks_list)
                if result is not True:
                    raise RuntimeError("BM25 builder update_index returned False")
                return [chunk.id for chunk in chunks_list]
            except (FileNotFoundError, RuntimeError) as e:
                # No existing index, create new one
                logger.info(f"No existing index found, creating new index with {len(chunks_list)} chunks")
                self.bm25_builder.from_chunks(chunks_list)
                return [chunk.id for chunk in chunks_list]

        except Exception as e:
            logger.error(f"Failed to build or update index: {e}", exc_info=True)
            raise

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds chunks to the pending queue and immediately flushes them.

        Legacy batch/interval knobs are still wired through the config for backward
        compatibility, but the current deletion guarantees require us to block until
        `_flush_pending_chunks` completes so callers know data is durable.
        """
        if not chunks:
            return []

        # Add chunks to pending queue
        with self._pending_lock:
            self._pending_chunks.extend(chunks)
            total_pending = len(self._pending_chunks)
        logger.info(f"Added {len(chunks)} chunks to pending queue. Total pending: {total_pending}")

        # Flush pending chunks before returning to guarantee durability
        flushed_ids = await self._flush_pending_chunks()
        if flushed_ids:
            return flushed_ids
        return [chunk.id for chunk in chunks]

    async def shutdown(self):
        """Shutdown the indexer and flush any pending chunks."""
        logger.info("Shutting down BM25Indexer...")

        # Flush any remaining pending chunks
        with self._pending_lock:
            pending_count = len(self._pending_chunks)
        if pending_count:
            logger.info(f"Flushing {pending_count} remaining chunks before shutdown")
            await self._flush_pending_chunks()
        logger.info("BM25Indexer shutdown complete")

    def _remove_pending_chunks(self, chunk_ids: List[str]) -> None:
        """Remove chunks from the pending queue prior to deletion."""
        if not self._pending_chunks or not chunk_ids:
            return

        chunk_id_set = set(chunk_ids)
        with self._pending_lock:
            original_length = len(self._pending_chunks)
            self._pending_chunks = deque(
                chunk for chunk in self._pending_chunks
                if chunk.id not in chunk_id_set
            )
            removed = original_length - len(self._pending_chunks)
        if removed > 0:
            logger.info(f"Removed {removed} pending chunks due to delete request")

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the BM25 index (synchronous).
        """
        try:
            self._remove_pending_chunks(chunk_ids)
            # Delete chunks from BM25 index
            result = self.bm25_builder.delete_index(chunk_ids)
            logger.info(f"Deletion result: {result}")
            return result if result is not None else False
        except Exception as e:
            logger.error(f"Failed to delete chunks from BM25 index: {e}")
            return False
