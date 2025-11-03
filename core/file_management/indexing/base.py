import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import uuid
from abc import ABC, abstractmethod

from framework.module import AbstractModule
from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


class BaseIndexer(AbstractModule, ABC):
    """
    Abstract Base Class for asynchronous, batch-capable indexers.

    This class provides the core, reusable logic for:
    - Handling single file or list of file paths.
    - Concurrently loading and parsing JSON files in a non-blocking way.
    - Transforming file content into Chunk objects.

    Subclasses must implement the `_batch_add_chunks` method to provide
    the specific indexing logic for their backend (e.g., BM25, FAISS).
    """

    @abstractmethod
    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Abstract method to add a batch of chunks to the specific index.
        This is the primary method that subclasses must implement.

        Args:
            chunks: A list of Chunk objects to be indexed.

        Returns:
            A list of chunk IDs that were successfully added.
        """
        pass

    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Abstract method to delete a batch of chunks from the specific index.
        This is a synchronous method to ensure compatibility with synchronous deletion pipelines.

        Args:
            chunk_ids: A list of chunk IDs to be deleted.

        Returns:
            True if deletion was successful, False otherwise.
        """
        pass

    async def index_chunk_files(self, chunk_file_paths: Union[str, List[str]]) -> bool:
        """
        Asynchronously loads and indexes one or more chunk JSON files.
        This is the main public entry point for the indexer.
        """
        if isinstance(chunk_file_paths, str):
            chunk_file_paths = [chunk_file_paths]

        if not chunk_file_paths:
            logger.warning("No file paths provided for indexing.")
            return False

        logger.info(f"Starting batch indexing for {len(chunk_file_paths)} files.")
        
        load_tasks = [self.load_chunk_from_file(path) for path in chunk_file_paths]
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        chunks_to_index = []
        for i, res in enumerate(results):
            if isinstance(res, Chunk):
                chunks_to_index.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Failed to load chunk from {chunk_file_paths[i]}: {res}")
        
        if not chunks_to_index:
            logger.error("All files failed to load. No chunks to index.")
            return False
            
        try:
            logger.info(f"Submitting a batch of {len(chunks_to_index)} chunks to the indexer.")
            chunk_ids = await self.update_index(chunks_to_index)
            
            if chunk_ids:
                logger.info(f"Successfully indexed a batch of {len(chunk_ids)} chunks.")
                return True
            else:
                logger.error("Indexer returned no IDs for the batch, indicating a failure.")
                return False
        except Exception as e:
            logger.error(f"An error occurred during the batch indexing process: {e}")
            return False

    async def load_chunk_from_file(self, file_path: str) -> Chunk:
        """
        Loads a single Chunk from a JSON file in a non-blocking way.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file does not exist or is not a file: {file_path}")

        chunk_data = json.loads(path.read_text(encoding='utf-8'))

        return self.create_chunk_from_chunk_data(chunk_data)

    @staticmethod
    def create_chunk_from_chunk_data(chunk_data: Dict[str, Any]) -> Chunk:
        """
        Creates a Chunk object from a chunk data dictionary.
        """
        content = chunk_data.get('content', '')
        if not content:
            logger.warning("Chunk data contains empty content for a chunk.")

        metadata = chunk_data.get('metadata', {})
        chunk_id = chunk_data.get('id', str(uuid.uuid4()))

        # Extract owner_id from metadata or source_metadata
        owner_id = metadata.get('owner_id', '')
        if not owner_id and 'source_metadata' in chunk_data:
            owner_id = chunk_data['source_metadata'].get('owner_id', '')

        return Chunk(id=chunk_id, content=content, owner_id=owner_id, metadata=metadata)
