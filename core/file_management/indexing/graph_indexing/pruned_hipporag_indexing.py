import asyncio
import logging
from typing import List, TYPE_CHECKING

from core.file_management.indexing.base import BaseIndexer
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.file_management.indexing.graph_indexing.pruned_hipporag_indexing_config import PrunedHippoRAGIndexerConfig

logger = logging.getLogger(__name__)


class PrunedHippoRAGIndexer(BaseIndexer):
    """
    Concrete implementation for indexing chunks using Pruned HippoRAG graph database with graph extraction.

    This indexer:
    1. Extracts Chunk data (retrieves chunks that need to be indexed)
    2. Uses HippoRAG2Extractor to extract graph data from chunk content
    3. Builds a graph index using the extracted graph data with Pruned HippoRAG Optimized Store
    """

    def __init__(self, config: "PrunedHippoRAGIndexerConfig"):
        """
        Initializes the Pruned HippoRAG indexer with extractor and graph store.

        Args:
            config: Configuration object containing extractor and Pruned HippoRAG store configs
        """
        super().__init__(config)
        self.extractor = config.extractor_config.build()
        self.graph_store = config.graph_store_config.build()

        logger.info("Pruned HippoRAG Indexer initialized with extractor and graph store")

    async def update_index(self, chunks: List[Chunk]) -> List[str]:
        """
        Adds a batch of chunks to the Pruned HippoRAG graph index.

        This method:
        1. Extracts graph data from chunks using HippoRAG2Extractor
        2. Adds chunks and their graph data to Pruned HippoRAG store using update_index
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
            # Step 1: Extract graph data from chunks using HippoRAG2Extractor
            logger.info(f"Extracting graph data from {len(chunks)} chunks...")
            extracted_chunks = await self._extract_graph_data(chunks)

            if not extracted_chunks:
                logger.error("Graph extraction failed for all chunks")
                return []

            logger.info(f"Successfully extracted graph data from {len(extracted_chunks)} chunks")

            # Step 2: Use graph_store.update_index() which handles:
            # - Adding chunks and graph data
            # - Batch generating embeddings
            # - Computing synonymy edges (if enabled)
            # - Building graph structure
            # - Rebuilding chunk embeddings array
            success = self.graph_store.update_index(extracted_chunks)

            if not success:
                logger.error("Failed to update graph store")
                return []

            # Collect successfully added chunk IDs
            chunk_ids = [chunk.id for chunk in extracted_chunks]

            # Step 3: Save index if storage path is configured
            if hasattr(self.graph_store, 'storage_path') and self.graph_store.storage_path:
                self.graph_store.save_index(self.graph_store.storage_path, self.graph_store.index_name)
                logger.info(f"Saved graph index to {self.graph_store.storage_path}")

            return chunk_ids

        except Exception as e:
            logger.error(f"Error during Pruned HippoRAG graph indexing: {e}", exc_info=True)
            return []

    async def _extract_graph_data(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Extract graph data from chunks using HippoRAG2Extractor.

        The HippoRAG2Extractor processes chunks concurrently based on its max_concurrent setting.

        Args:
            chunks: List of chunks to extract graph data from

        Returns:
            List of chunks with extracted graph data
        """
        try:
            # Use the extractor's __call__ method which handles concurrent extraction
            # This internally calls extract_concurrent() with proper semaphore control
            extracted_chunks = await self.extractor(chunks)

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

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes a batch of chunks from the Pruned HippoRAG graph index (synchronous).

        Args:
            chunk_ids: A list of chunk IDs to be deleted

        Returns:
            True if deletion was successful, False otherwise
        """
        if not chunk_ids:
            logger.warning("No chunk IDs provided for deletion")
            return False

        try:
            # Delete chunks from Pruned HippoRAG store
            result = self.graph_store.delete_index(chunk_ids)

            # Save index if storage path is configured
            if result and hasattr(self.graph_store, 'storage_path') and self.graph_store.storage_path:
                self.graph_store.save_index(
                    self.graph_store.storage_path,
                    self.graph_store.index_name
                )
                logger.info(f"Saved graph index after deletion to {self.graph_store.storage_path}")

            return result

        except Exception as e:
            logger.error(f"Error during Pruned HippoRAG graph deletion: {e}", exc_info=True)
            return False

