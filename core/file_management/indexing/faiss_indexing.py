import asyncio
import logging
from typing import List

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Document

logger = logging.getLogger(__name__)


class FaissIndexer(BaseIndexer):
    """
    Concrete implementation for indexing documents using FaissVectorDB.
    """

    def __init__(self, config):
        """
        Initializes the Faiss indexer and its specific database instance.
        """
        super().__init__(config)
        self.faiss_db = config.build()

    async def update_index(self, documents: List[Document]) -> List[str]:
        """
        Adds a batch of documents to the FAISS index using a thread pool.
        """
        loop = asyncio.get_running_loop()
        doc_ids = await loop.run_in_executor(
            None, self.faiss_db.update_index, documents
        )
        return doc_ids or []
