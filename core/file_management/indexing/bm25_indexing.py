import asyncio
import logging
from typing import List, TYPE_CHECKING

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig

logger = logging.getLogger(__name__)


class BM25Indexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using BM25.
    """
    
    def __init__(self, config: "BM25IndexerConfig"):
        """
        Initializes the BM25 indexer and its specific builder.
        """
        super().__init__(config)
        self.bm25_builder = config.index_config.build()

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds a batch of chunks to the BM25 index using a thread pool.
        If index doesn't exist, creates a new one.
        """
        loop = asyncio.get_running_loop()

        def build_or_update_index(chunks_list):
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

        # The actual blocking call is executed in a separate thread.
        chunk_ids = await loop.run_in_executor(None, build_or_update_index, chunks)
        return chunk_ids or []

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the BM25 index (synchronous).
        """
        try:
            # Delete chunks from BM25 index
            result = self.bm25_builder.delete_index(chunk_ids)
            return result if result is not None else False
        except Exception as e:
            logger.error(f"Failed to delete chunks from BM25 index: {e}")
            return False
