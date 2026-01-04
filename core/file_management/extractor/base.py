"""
Base extractor with simple concurrent extraction functionality.
"""

from abc import abstractmethod
from typing import List
import asyncio
import logging

from encapsulation.data_model.schema import Chunk, GraphData
from framework.module import AbstractModule
from core.file_management.extractor.metadata_keys import EXTRACTION_ERROR_KEY

logger = logging.getLogger(__name__)

_DEFAULT_ERROR_POLICY = "attach"  # attach | raise | empty


def _coerce_error_policy(value: object) -> str:
    policy = str(value or "").strip().lower()
    if policy in {"attach", "raise", "empty"}:
        return policy
    return _DEFAULT_ERROR_POLICY


def _build_error_payload(*, exc: BaseException, chunk_id: str | None, extractor_name: str, llm_model: str | None) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "extractor": extractor_name,
        "llm_model": llm_model,
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
    }
    return {k: v for k, v in payload.items() if v is not None}


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

    async def process_chunk(self, chunk: Chunk, *, semaphore: asyncio.Semaphore) -> Chunk:
        """process a single chunk"""
        async with semaphore:
            error_policy = _coerce_error_policy(getattr(self.config, "error_policy", _DEFAULT_ERROR_POLICY))
            try:
                graph_data = await self.extract(chunk)
                chunk.graph = graph_data
                return chunk
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.id}: {e}", exc_info=True)
                if error_policy == "raise":
                    raise

                chunk_id = getattr(chunk, "id", None)
                llm_config = getattr(self.config, "llm_config", None)
                llm_model = getattr(llm_config, "model_name", None) if llm_config is not None else None
                payload = _build_error_payload(
                    exc=e,
                    chunk_id=str(chunk_id) if chunk_id else None,
                    extractor_name=self.__class__.__name__,
                    llm_model=str(llm_model) if llm_model else None,
                )

                graph = GraphData()
                if error_policy == "attach":
                    graph.metadata[EXTRACTION_ERROR_KEY] = payload
                    chunk.metadata[EXTRACTION_ERROR_KEY] = payload
                chunk.graph = graph  # empty graph data (with attached error if configured)
                return chunk

    async def extract_concurrent(self, chunks: List[Chunk]) -> List[Chunk]:
        """extract from multiple chunks concurrently"""
        if not chunks:
            return []

        logger.info(f"Starting concurrent extraction with max_concurrent={self.config.max_concurrent}")

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        batch_size = getattr(self.config, "batch_size", None)
        try:
            batch_size_int = int(batch_size) if batch_size is not None else 0
        except (TypeError, ValueError):
            batch_size_int = 0

        # process_chunk handles all exceptions internally, so we don't need return_exceptions=True
        if batch_size_int <= 0:
            tasks = [self.process_chunk(chunk, semaphore=semaphore) for chunk in chunks]
            return await asyncio.gather(*tasks)

        out: list[Chunk] = []
        for start in range(0, len(chunks), batch_size_int):
            batch = chunks[start : start + batch_size_int]
            tasks = [self.process_chunk(chunk, semaphore=semaphore) for chunk in batch]
            out.extend(await asyncio.gather(*tasks))
        return out

    async def __call__(self, chunks: List[Chunk]) -> List[Chunk]:
        return await self.extract_concurrent(chunks)
