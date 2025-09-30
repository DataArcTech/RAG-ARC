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
                # Check if index exists
                if self.bm25_builder.index_exists():
                    # Index exists, update it
                    result = self.bm25_builder.update_index(chunks_list)
                    if result:
                        return [chunk.id for chunk in chunks_list]
                    return []
                else:
                    # Index doesn't exist, create new one using add
                    # which handles initialization properly
                    self.bm25_builder.add_chunks(chunks_list)
                    return [chunk.id for chunk in chunks_list]
            except RuntimeError as e:
                if "Index has not been initialized" in str(e):
                    # Force initialization and try again
                    self.bm25_builder._initialize_index(chunks_list) 
                    self.bm25_builder.from_chunks(chunks_list)
                    return [chunk.id for chunk in chunks_list]
                else:
                    raise

        # The actual blocking call is executed in a separate thread.
        chunk_ids = await loop.run_in_executor(None, build_or_update_index, chunks)
        return chunk_ids or []
