import asyncio
import logging
import time
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

    async def update_index(self, chunks: List[Chunk]):  # type: ignore[override]
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
            t_extract0 = time.perf_counter()
            extracted_chunks = await self._extract_graph_data(chunks)
            t_extract = time.perf_counter() - t_extract0

            if not extracted_chunks:
                raise RuntimeError("Graph extraction failed for all chunks")

            logger.info(f"Successfully extracted graph data from {len(extracted_chunks)} chunks")

            # Step 2: Use graph_store.update_index() which handles:
            # - Adding chunks and graph data
            # - Batch generating embeddings
            # - Computing synonymy edges (if enabled)
            # - Building graph structure
            # - Rebuilding chunk embeddings array
            loop = asyncio.get_running_loop()
            t_update0 = time.perf_counter()
            store_stats = None
            update_with_stats = getattr(self.graph_store, "update_index_with_stats", None)
            if callable(update_with_stats):
                success, store_stats = await loop.run_in_executor(None, update_with_stats, extracted_chunks)
            else:
                success = await loop.run_in_executor(None, self.graph_store.update_index, extracted_chunks)
            t_update = time.perf_counter() - t_update0

            if not success:
                raise RuntimeError("Failed to update graph store")

            # Collect successfully added chunk IDs
            chunk_ids = [chunk.id for chunk in extracted_chunks]

            # Step 3: Save index if storage path is configured
            t_save = 0.0
            if hasattr(self.graph_store, 'storage_path') and self.graph_store.storage_path:
                t_save0 = time.perf_counter()
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    self.graph_store.save_index,
                    self.graph_store.storage_path,
                    self.graph_store.index_name,
                )
                t_save = time.perf_counter() - t_save0
                logger.info(f"Saved graph index to {self.graph_store.storage_path}")

            # Return a richer payload (handled by IndexManagerPipeline) so perf experiments can attribute costs.
            return {
                "indexed_ids": chunk_ids,
                "metadata": {
                    "timings_s": {
                        "extract_graph": float(t_extract),
                        "graph_store_update_index": float(t_update),
                        "graph_store_save_index": float(t_save),
                    },
                    "graph_store": store_stats,
                },
            }

        except Exception as e:
            logger.error(f"Error during Pruned HippoRAG graph indexing: {e}", exc_info=True)
            raise

    async def _extract_graph_data(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Extract graph data from chunks using HippoRAG2Extractor.

        The HippoRAG2Extractor processes chunks concurrently based on its max_concurrent setting.

        Args:
            chunks: List of chunks to extract graph data from

        Returns:
            List of chunks with extracted graph data
        """
        # Use the extractor's __call__ method which handles concurrent extraction.
        # NOTE (P0): Do NOT filter out empty graphs or extraction failures here.
        # Empty/failed chunks must still be ingested so downstream graph stores can:
        # - keep dense fallback over chunk embeddings covering all chunks
        # - record `chunks_graph_empty` / `chunks_extraction_failed` in ingest meta
        try:
            return await self.extractor(chunks)
        except Exception as exc:
            logger.error("Error during graph extraction", exc_info=True)
            raise exc

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
