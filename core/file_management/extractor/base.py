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

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        logger.info(f"Starting concurrent extraction with max_concurrent={self.config.max_concurrent}")

        async def process_with_semaphore(chunk: Chunk) -> Chunk:
            async with semaphore:
                return await self.process_chunk(chunk)

        return await asyncio.gather(*[process_with_semaphore(chunk) for chunk in chunks])

    def __call__(self, chunks: List[Chunk]) -> List[Chunk]:
        """sync interface that handles both sync and async contexts"""
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # If we're already in an event loop, create a new thread with its own event loop
            import concurrent.futures

            # Create a new thread with its own event loop for concurrent processing
            def run_in_thread():
                return asyncio.run(self.extract_concurrent(chunks))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()

        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.extract_concurrent(chunks))