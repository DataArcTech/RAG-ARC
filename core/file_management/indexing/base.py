import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import uuid
from abc import ABC, abstractmethod

from framework.module import AbstractModule
from encapsulation.data_model.schema import Document

logger = logging.getLogger(__name__)


class BaseIndexer(AbstractModule, ABC):
    """
    Abstract Base Class for asynchronous, batch-capable indexers.

    This class provides the core, reusable logic for:
    - Handling single file or list of file paths.
    - Concurrently loading and parsing JSON files in a non-blocking way.
    - Transforming file content into Document objects.

    Subclasses must implement the `_batch_add_documents` method to provide
    the specific indexing logic for their backend (e.g., BM25, FAISS).
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def update_index(self, documents: List[Document]) -> List[str]:
        """
        Abstract method to add a batch of documents to the specific index.
        This is the primary method that subclasses must implement.

        Args:
            documents: A list of Document objects to be indexed.

        Returns:
            A list of document IDs that were successfully added.
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
        
        documents_to_index = []
        for i, res in enumerate(results):
            if isinstance(res, Document):
                documents_to_index.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Failed to load document from {chunk_file_paths[i]}: {res}")
        
        if not documents_to_index:
            logger.error("All files failed to load. No documents to index.")
            return False
            
        try:
            logger.info(f"Submitting a batch of {len(documents_to_index)} documents to the indexer.")
            doc_ids = await self.update_index(documents_to_index)
            
            if doc_ids:
                logger.info(f"Successfully indexed a batch of {len(doc_ids)} documents.")
                return True
            else:
                logger.error("Indexer returned no IDs for the batch, indicating a failure.")
                return False
        except Exception as e:
            logger.error(f"An error occurred during the batch indexing process: {e}")
            return False

    async def load_chunk_from_file(self, file_path: str) -> Document:
        """
        Loads a single Document from a JSON file in a non-blocking way.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file does not exist or is not a file: {file_path}")
            
        loop = asyncio.get_running_loop()
        
        content = await loop.run_in_executor(None, path.read_text, 'utf-8')
        chunk_data = await loop.run_in_executor(None, json.loads, content)
        
        return self.create_document_from_chunk(chunk_data)

    @staticmethod
    def create_document_from_chunk(chunk_data: Dict[str, Any]) -> Document:
        """
        Creates a Document object from a chunk data dictionary.
        """
        content = chunk_data.get('content', '')
        if not content:
            logger.warning("Chunk data contains empty content for a document.")
            
        metadata = chunk_data.get('metadata', {})
        doc_id = chunk_data.get('id', str(uuid.uuid4()))
            
        return Document(id=doc_id, content=content, metadata=metadata)
