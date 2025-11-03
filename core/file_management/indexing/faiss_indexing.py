import asyncio
import logging
from typing import List, TYPE_CHECKING

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig

logger = logging.getLogger(__name__)


class FaissIndexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using FaissVectorDB.
    """

    def __init__(self, config: "FaissIndexerConfig"):
        """
        Initializes the Faiss indexer and its specific database instance.
        """
        super().__init__(config)
        self.faiss_db = config.index_config.build()

        # Load existing index if it exists
        try:
            if hasattr(self.faiss_db.config, 'index_path'):
                self.faiss_db.load_index(self.faiss_db.config.index_path)
                logger.info(f"Loaded existing FAISS index from {self.faiss_db.config.index_path}")
        except Exception as e:
            logger.info(f"No existing FAISS index found or failed to load: {e}. Will create new index when chunks are added.")

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds a batch of chunks to the FAISS index using a thread pool.
        """

        # Update the index
        chunk_ids = self.faiss_db.update_index(chunks)
        # Save the index to disk
        if hasattr(self.faiss_db.config, 'index_path'):
            self.faiss_db.save_index(self.faiss_db.config.index_path)
        return chunk_ids or []

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the FAISS index using soft-delete (synchronous).
        """
        try:
            # Delete chunks from index (soft-delete)
            result = self.faiss_db.delete_index(chunk_ids)
            # Save the index to disk
            if hasattr(self.faiss_db.config, 'index_path'):
                self.faiss_db.save_index(self.faiss_db.config.index_path)
            return result if result is not None else False
        except Exception as e:
            logger.error(f"Failed to delete chunks from FAISS index: {e}")
            return False
