"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import abstractmethod
from typing import List
import asyncio
import logging

from encapsulation.data_model.data_model import Document, GraphData
from framework.module import AbstractModule

# Set up logger
logger = logging.getLogger(__name__)

class ExtractorBase(AbstractModule):
    """BaseExtractor，只负责单轮抽取和并发控制"""

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.llm_config.build()

    @abstractmethod
    async def extract(self, document: Document) -> GraphData:
        """extract from a single document

        Args:
            document: Document to extract

        Returns:
            GraphData: Extracted graph data
        """
        pass

    async def process_document(self, document: Document) -> Document:
        """process a single document"""
        try:
            graph_data = await self.extract(document)
            document.graph = graph_data
            return document
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}", exc_info=True)
            document.graph = GraphData()  # return empty graph data
            return document

    async def extract_concurrent(self, documents: List[Document]) -> List[Document]:
        """extract from multiple documents concurrently"""
        if not documents:
            return []

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_with_semaphore(doc: Document) -> Document:
            async with semaphore:
                return await self.process_document(doc)

        return await asyncio.gather(*[process_with_semaphore(doc) for doc in documents])

    def __call__(self, documents: List[Document]) -> List[Document]:
        """sync interface for single-threaded execution"""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.extract_concurrent(documents))
        except RuntimeError:
            return asyncio.run(self.extract_concurrent(documents))