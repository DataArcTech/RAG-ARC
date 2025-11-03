import asyncio
import logging
from typing import List, TYPE_CHECKING

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig

logger = logging.getLogger(__name__)


class NetworkXGraphIndexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using NetworkX graph database with graph extraction.

    This indexer:
    1. Extracts Chunk data (retrieves chunks that need to be indexed)
    2. Uses GraphExtractor to extract graph data from chunk content
    3. Builds a graph index using the extracted graph data with NetworkX
    """

    def __init__(self, config: "NetworkXGraphIndexerConfig"):
        """
        Initializes the NetworkX graph indexer with extractor and graph store.

        Args:
            config: Configuration object containing extractor and NetworkX store configs
        """
        super().__init__(config)
        self.extractor = config.extractor_config.build()
        self.networkx_store = config.graph_store_config.build()

        logger.info("NetworkX Graph Indexer initialized with extractor and graph store")

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds a batch of chunks to the NetworkX graph index.

        This method:
        1. Extracts graph data from chunks using GraphExtractor
        2. Adds chunks and their graph data to NetworkX store
        3. Saves the index to disk if configured

        Args:
            chunks: A list of Chunk objects to be indexed

        Returns:
            A list of chunk IDs that were successfully added
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return []

        try:
            # Step 1: Extract graph data from chunks using GraphExtractor
            logger.info(f"Extracting graph data from {len(chunks)} chunks...")
            extracted_chunks = await self._extract_graph_data(chunks)

            if not extracted_chunks:
                logger.error("Graph extraction failed for all chunks")
                return []

            logger.info(f"Successfully extracted graph data from {len(extracted_chunks)} chunks")

            # Step 2: Add chunks and graph data to NetworkX store
            chunk_ids = self._add_to_graph_store(extracted_chunks)

            # Step 3: Save index if storage path is configured
            if hasattr(self.networkx_store, 'storage_path') and self.networkx_store.storage_path:
                self.networkx_store.save_index(self.networkx_store.storage_path, self.networkx_store.index_name)
                logger.info(f"Saved graph index to {self.networkx_store.storage_path}")

            return chunk_ids

        except Exception as e:
            logger.error(f"Error during NetworkX graph indexing: {e}", exc_info=True)
            return []

    async def _extract_graph_data(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Extract graph data from chunks using GraphExtractor.

        The GraphExtractor processes chunks concurrently based on its max_concurrent setting.

        Args:
            chunks: List of chunks to extract graph data from

        Returns:
            List of chunks with extracted graph data
        """
        try:
            # Use the extractor's __call__ method which handles concurrent extraction
            # This internally calls extract_concurrent() with proper semaphore control
            extracted_chunks = self.extractor(chunks)

            # Filter out chunks that failed extraction (empty graph data)
            valid_chunks = []
            for chunk in extracted_chunks:
                if chunk.graph and not chunk.graph.is_empty():
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"Chunk {chunk.id} has empty graph data, skipping")

            return valid_chunks

        except Exception as e:
            logger.error(f"Error during graph extraction: {e}", exc_info=True)
            return []

    def _add_to_graph_store(self, chunks: List[Chunk]) -> List[str]:
        """
        Add chunks and their graph data to the NetworkX store.

        This is a blocking operation that runs in a thread pool executor.

        Args:
            chunks: List of chunks with extracted graph data

        Returns:
            List of successfully added chunk IDs
        """
        added_chunk_ids = []

        for chunk in chunks:
            try:
                # Add chunk to the graph store
                self.networkx_store.add_chunk(chunk)

                # Add graph data (entities and relations) to the store
                if chunk.graph and not chunk.graph.is_empty():
                    self.networkx_store.add_graph_data(chunk.graph, chunk.id)
                    added_chunk_ids.append(chunk.id)
                    logger.debug(f"Added chunk {chunk.id} with {len(chunk.graph.entities)} entities "
                               f"and {len(chunk.graph.relations)} relations")
                else:
                    logger.warning(f"Chunk {chunk.id} has no graph data to add")

            except Exception as e:
                logger.error(f"Failed to add chunk {chunk.id} to graph store: {e}")
                continue

        logger.info(f"Successfully added {len(added_chunk_ids)} chunks to NetworkX graph store")
        return added_chunk_ids

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the NetworkX graph index (synchronous).

        Args:
            chunk_ids: A list of chunk IDs to be deleted

        Returns:
            True if deletion was successful, False otherwise
        """
        if not chunk_ids:
            logger.warning("No chunk IDs provided for deletion")
            return False

        try:
            # Delete chunks from NetworkX store
            result = self.networkx_store.delete_index(chunk_ids)

            # Save index if storage path is configured
            if result and hasattr(self.networkx_store, 'storage_path') and self.networkx_store.storage_path:
                self.networkx_store.save_index(
                    self.networkx_store.storage_path,
                    self.networkx_store.index_name
                )
                logger.info(f"Saved graph index after deletion to {self.networkx_store.storage_path}")

            return result

        except Exception as e:
            logger.error(f"Error during NetworkX graph deletion: {e}", exc_info=True)
            return False
