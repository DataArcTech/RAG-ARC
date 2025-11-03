"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import abstractmethod
from typing import List
import asyncio
import logging

from encapsulation.data_model.schema import Chunk, GraphData
from framework.module import AbstractModule

logger = logging.getLogger(__name__)

class ExtractorBase(AbstractModule):
    """BaseExtractor: only responsible for single-round extraction and concurrency control"""

    def __init__(self, config):
        super().__init__(config)
        self.llm = config.llm_config.build()
        self.extraction_semaphore = asyncio.Semaphore(config.max_concurrent)

    @abstractmethod
    async def extract(self, chunk: Chunk) -> GraphData:
        """extract from a single chunk

        Args:
            chunk: Chunk to extract

        Returns:
            GraphData: Extracted graph data
        """
        pass

    async def process_chunk(self, chunk: Chunk) -> Chunk:
        """process a single chunk"""
        async with self.extraction_semaphore:
            try:
                graph_data = await self.extract(chunk)
                chunk.graph = graph_data
                return chunk
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}", exc_info=True)
                chunk.graph = GraphData()  # return empty graph data
                return chunk

    async def extract_concurrent(self, chunks: List[Chunk]) -> List[Chunk]:
        """extract from multiple chunks concurrently"""
        if not chunks:
            return []

        logger.info(f"Starting concurrent extraction with max_concurrent={self.config.max_concurrent}")

        # process_chunk handles all exceptions internally, so we don't need return_exceptions=True
        tasks = [self.process_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def __call__(self, chunks: List[Chunk]) -> List[Chunk]:
        return await self.extract_concurrent(chunks)