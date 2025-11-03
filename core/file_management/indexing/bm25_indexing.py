import asyncio
import logging
from typing import List, TYPE_CHECKING
from collections import deque
import time

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

        # Pending chunks queue for batch processing
        self._pending_chunks: deque[Chunk] = deque()

        # Last flush timestamp
        self._last_flush_time = time.time()

        # Background flush task
        self._flush_task: asyncio.Task = None
        self._shutdown = False

    def _start_background_flush(self):
        """Start the background flush task if not already running."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._background_flush_worker())
            logger.info("Started background flush worker")

    async def _background_flush_worker(self):
        """Background worker that periodically flushes pending chunks."""
        logger.info(f"Background flush worker started with interval: {self.flush_interval}s")

        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)

                # Check if there are pending chunks and enough time has passed
                if self._pending_chunks:
                    current_time = time.time()
                    time_since_last_flush = current_time - self._last_flush_time

                    if time_since_last_flush >= self.flush_interval:
                        logger.info(f"Background flush triggered: {len(self._pending_chunks)} pending chunks")
                        await self._flush_pending_chunks()

            except asyncio.CancelledError:
                logger.info("Background flush worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background flush worker: {e}", exc_info=True)

    async def _flush_pending_chunks(self) -> List[str]:
        """Flush all pending chunks to the index."""
        if not self._pending_chunks:
            return []

        # Acquire lock to ensure exclusive write access
        async with self._write_lock:
            # Collect all pending chunks
            chunks_to_index = list(self._pending_chunks)
            self._pending_chunks.clear()

            if not chunks_to_index:
                return []

            logger.info(f"Flushing {len(chunks_to_index)} chunks to BM25 index")

            # Perform the actual indexing in a thread pool
            loop = asyncio.get_running_loop()
            chunk_ids = await loop.run_in_executor(
                None,
                self._build_or_update_index_sync,
                chunks_to_index
            )

            # Update last flush time
            self._last_flush_time = time.time()

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
                if result:
                    return [chunk.id for chunk in chunks_list]
                return []

            # Index not loaded, try to load existing index or create new one
            try:
                # Try to load existing index from disk
                self.bm25_builder.load_local()
                logger.info(f"Loaded existing index, updating {len(chunks_list)} chunks")
                result = self.bm25_builder.update_index(chunks_list)
                if result:
                    return [chunk.id for chunk in chunks_list]
                return []
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
        Adds chunks to the pending queue for batch processing.

        This method is NON-BLOCKING - it adds chunks to the queue and returns immediately.
        The actual indexing happens in the background flush worker.

        Flush trigger strategies:
        1. If pending chunks >= batch_size: trigger immediate flush (non-blocking)
        2. Otherwise: wait for periodic flush
        """
        if not chunks:
            return []

        # Start background flush worker if not running
        self._start_background_flush()

        # Add chunks to pending queue
        self._pending_chunks.extend(chunks)
        total_pending = len(self._pending_chunks)
        logger.info(f"Added {len(chunks)} chunks to pending queue. Total pending: {total_pending}")

        # Strategy 1: Batch size reached - trigger immediate flush (non-blocking)
        if total_pending >= self.batch_size:
            logger.info(f"Batch size ({self.batch_size}) reached, triggering immediate flush")
            # Create a flush task but don't wait for it
            asyncio.create_task(self._flush_pending_chunks())

        # Return chunk IDs immediately (they will be indexed by background worker)
        return [chunk.id for chunk in chunks]

    async def shutdown(self):
        """Shutdown the indexer and flush any pending chunks."""
        logger.info("Shutting down BM25Indexer...")
        self._shutdown = True

        # Flush any remaining pending chunks
        if self._pending_chunks:
            logger.info(f"Flushing {len(self._pending_chunks)} remaining chunks before shutdown")
            await self._flush_pending_chunks()

        # Cancel background flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        logger.info("BM25Indexer shutdown complete")

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the BM25 index (synchronous).
        """
        try:
            # Delete chunks from BM25 index
            result = self.bm25_builder.delete_index(chunk_ids)
            logger.info(f"Deletion result: {result}")
            return result if result is not None else False
        except Exception as e:
            logger.error(f"Failed to delete chunks from BM25 index: {e}")
            return False
