import asyncio
import logging
from typing import List

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Document

logger = logging.getLogger(__name__)


class BM25Indexer(BaseIndexer):
    """
    Concrete implementation for indexing documents using BM25.
    """
    
    def __init__(self, config):
        """
        Initializes the BM25 indexer and its specific builder.
        """
        super().__init__(config)
        self.bm25_builder = config.build()

    async def update_index(self, documents: List[Document]) -> List[str]:
        """
        Adds a batch of documents to the BM25 index using a thread pool.
        If index doesn't exist, creates a new one.
        """
        loop = asyncio.get_running_loop()

        def build_or_update_index(docs):
            # Check if index exists
            if self.bm25_builder.index_exists():
                # Index exists, update it
                result = self.bm25_builder.update_index(docs)
                if result:
                    return [doc.id for doc in docs]
                return []
            else:
                # Index doesn't exist, create new one
                self.bm25_builder.build_index(docs)
                return [doc.id for doc in docs]

        # The actual blocking call is executed in a separate thread.
        doc_ids = await loop.run_in_executor(None, build_or_update_index, documents)
        return doc_ids or []
