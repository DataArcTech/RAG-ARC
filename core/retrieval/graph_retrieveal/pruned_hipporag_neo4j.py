import logging
import threading
from typing import TYPE_CHECKING

from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_cache import _PrunedHippoRAGNeo4jCacheMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_facts import _PrunedHippoRAGNeo4jFactsMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_graph import _PrunedHippoRAGNeo4jGraphMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_ppr import _PrunedHippoRAGNeo4jPPRMixin
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j_retrieve import _PrunedHippoRAGNeo4jRetrieveMixin

if TYPE_CHECKING:
    from config.core.retrieval.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jRetrievalConfig

logger = logging.getLogger(__name__)

class PrunedHippoRAGNeo4jRetriever(
    _PrunedHippoRAGNeo4jCacheMixin,
    _PrunedHippoRAGNeo4jGraphMixin,
    _PrunedHippoRAGNeo4jPPRMixin,
    _PrunedHippoRAGNeo4jRetrieveMixin,
    _PrunedHippoRAGNeo4jFactsMixin,
    PrunedHippoRAGRetriever,
):
    """
    Pruned HippoRAG Retrieval System with Neo4j backend.

    This retriever extends the base PrunedHippoRAGRetriever to work with Neo4j
    as the graph database backend instead of SQLite + igraph.

    Key differences:
    - Graph queries use Neo4j Cypher instead of igraph methods
    - Node mappings are built from Neo4j instead of SQLite
    - Subgraph extraction uses Neo4j queries
    - PageRank still uses igraph (extracted subgraph)
    """

    def __init__(self, config: "PrunedHippoRAGNeo4jRetrievalConfig"):
        """
        Initialize the Pruned HippoRAG Retriever with Neo4j backend.

        Args:
            config: Configuration object containing all retrieval parameters
        """
        self._tls = threading.local()
        # Call parent __init__ but skip some igraph-specific initialization
        self.config = config

        # Build and load graph store (Neo4j version)
        self.graph_store = config.graph_config.build()

        self.embedding_model = self.graph_store.embedding_model

        # Initialize optional LLM client for fact reranking
        self.llm_client = None
        if config.llm_config is not None:
            try:
                self.llm_client = config.llm_config.build()
                logger.info("LLM client initialized for fact filtering")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Will use fallback filtering.")

        # Cache for node mappings to avoid rebuilding on every query
        self._cached_owner_id = None
        self._cached_store_version = None  # Track graph store cache version
        self.passage_node_keys = []
        self.passage_embeddings_array = None

        # PPR backend selection: 'push' (default, faster) or 'igraph' (fallback)
        self.ppr_backend = getattr(config, "ppr_backend", "push")
        # Default is undirected PPR for retrieval stability; deepsearch/fast tools can override.
        self.ppr_directed_mode = getattr(config, "ppr_directed_mode", "off")

        # Build initial node mappings
        self._build_node_mappings()

        logger.info("Pruned HippoRAG Retrieval System (Neo4j) initialized")
        logger.info(f"  PPR backend: {self.ppr_backend}")
        logger.info(f"  Expansion hops: {config.expansion_hops}")
        logger.info(f"  Include chunk neighbors: {config.include_chunk_neighbors}")
        logger.info(f"  Enable query-aware pruning: {config.enable_pruning}")
        if config.enable_pruning:
            logger.info(f"    Base max neighbors: {config.max_neighbors}")
            logger.info(f"    Query-aware multiplier: {config.query_aware_multiplier}")
        logger.info(f"    Min/Max neighbors: {config.query_aware_min_k}/{config.query_aware_max_k}")
