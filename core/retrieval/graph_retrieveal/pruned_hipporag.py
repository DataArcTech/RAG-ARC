import logging
import os
import threading
import uuid
from typing import TYPE_CHECKING

from core.retrieval.graph_retrieveal.base import BaseGraphRetriever
from core.retrieval.graph_retrieveal.pruned_hipporag_dense import _PrunedHippoRAGDenseMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_facts import _PrunedHippoRAGFactsMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_graph import _PrunedHippoRAGGraphMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_ppr import _PrunedHippoRAGPPRMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_retrieve import _PrunedHippoRAGRetrieveMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_tls import _PrunedHippoRAGTLSMixin

if TYPE_CHECKING:
    from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig

logger = logging.getLogger(__name__)


class PrunedHippoRAGRetriever(
    _PrunedHippoRAGTLSMixin,
    _PrunedHippoRAGRetrieveMixin,
    _PrunedHippoRAGFactsMixin,
    _PrunedHippoRAGGraphMixin,
    _PrunedHippoRAGPPRMixin,
    _PrunedHippoRAGDenseMixin,
    BaseGraphRetriever,
):
    """
    Pruned HippoRAG Retrieval System.

    This retriever implements a graph-based retrieval approach that:
    1. Retrieves relevant facts from a knowledge graph using dense retrieval (FAISS)
    2. Optionally reranks facts using an LLM to improve relevance
    3. Extracts seed entities from top-ranked facts
    4. Expands a subgraph around seed entities with optional pruning
    5. Performs Personalized PageRank (PPR) on the subgraph to rank passages

    The "pruned" aspect refers to limiting the number of neighbors during graph expansion
    to balance retrieval quality and computational efficiency.
    """

    def __init__(self, config: "PrunedHippoRAGRetrievalConfig"):
        """
        Initialize the Pruned HippoRAG Retriever.

        Args:
            config: Configuration object containing all retrieval parameters
        """
        self._tls = threading.local()
        super().__init__(config)

        # Build and load graph store
        self.graph_store = config.graph_config.build()

        storage_path = config.graph_config.storage_path
        index_name = config.graph_config.index_name
        if os.path.exists(os.path.join(storage_path, f"{index_name}_graph.pkl")):
            logger.info(f"Loading existing graph index from {storage_path}...")
            self.graph_store.load_index(storage_path, index_name)
        else:
            logger.info(f"No existing index found at {storage_path}, starting with empty graph")

        self.embedding_model = self.graph_store.embedding_model

        # Initialize optional LLM client for fact reranking
        self.llm_client = None
        if config.llm_config is not None:
            try:
                self.llm_client = config.llm_config.build()
                logger.info("LLM client initialized for fact filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Will use fallback filtering.")

        # Build initial node mappings
        self._build_node_mappings()

        logger.info("Pruned HippoRAG Retrieval System initialized")
        logger.info(f"  Expansion hops: {config.expansion_hops}")
        logger.info(f"  Include chunk neighbors: {config.include_chunk_neighbors}")
        logger.info(f"  Enable query-aware pruning: {config.enable_pruning}")
        if config.enable_pruning:
            logger.info(f"    Base max neighbors: {config.max_neighbors}")
            logger.info(f"    Query-aware multiplier: {config.query_aware_multiplier}")
            logger.info(f"    Min/Max neighbors: {config.query_aware_min_k}/{config.query_aware_max_k}")

