"""
Pruned HippoRAG Graph Store (igraph backend).

Design note (important):
- This backend builds an **undirected** igraph (`igraph.Graph(directed=False)`) for a lightweight, local/test-friendly
  retrieval path.
- Direction-sensitive KG semantics (e.g., OWNS, creditor/debtor, before/after) require the Neo4j backend, which
  preserves per-predicate directionality and supports Cypher-backed deterministic DeepSearch tools.
"""

import os
import json
import logging
import pickle
import sqlite3
import re
import threading
from typing import List, Dict, Any, Optional, Sequence, TYPE_CHECKING
from collections import defaultdict
import numpy as np
import igraph as ig
import faiss

from encapsulation.database.graph_db.base import GraphStore
from encapsulation.database.graph_db.pruned_hipporag_igraph_embeddings import _PrunedHippoRAGIGraphEmbeddingsMixin
from encapsulation.database.graph_db.pruned_hipporag_igraph_indexing import _PrunedHippoRAGIGraphIndexingMixin
from encapsulation.database.graph_db.pruned_hipporag_igraph_sqlite import _PrunedHippoRAGIGraphSQLiteMixin
from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from encapsulation.database.utils.sqlite_threadlocal import ThreadLocalSQLiteConnection
from core.utils.path_guard import ensure_writable_dir
from core.utils.rwlock import RWLock
from framework.shared_module_decorator import shared_module
from framework.virtual_paths import is_io_path, resolve_io_to_local_path

if TYPE_CHECKING:
    from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig

logger = logging.getLogger(__name__)


@shared_module
class PrunedHippoRAGIGraphStore(
    _PrunedHippoRAGIGraphEmbeddingsMixin,
    _PrunedHippoRAGIGraphSQLiteMixin,
    _PrunedHippoRAGIGraphIndexingMixin,
    GraphStore,
):
    """
    Pruned HippoRAG Graph Store using hybrid storage.

    This graph store implements a multi-component storage system:

    1. **Facts**: FAISS Flat index for exact similarity search
       - Stores fact embeddings for dense retrieval
       - Used to find relevant facts given a query

    2. **Entities**: FAISS HNSW index for approximate nearest neighbor search
       - Stores entity embeddings
       - Used to compute synonymy edges between similar entities

    3. **Chunks**: In-memory numpy array for brute-force search
       - Stores chunk embeddings
       - Used for dense passage retrieval fallback

    4. **Metadata**: SQLite database
       - Stores chunks, entities, facts, and their relationships
       - Provides efficient querying and filtering

    5. **Graph**: igraph undirected graph
       - Represents the knowledge graph structure
       - Nodes: chunks and entities
       - Edges: chunk-entity relations, entity-entity relations (facts), synonymy edges
       - Used for Personalized PageRank during retrieval
    """

    OWNER_GLOBAL_KEY = "__GLOBAL__"

    @staticmethod
    def _normalize_owner_id(owner_id: Optional[Any]) -> Optional[str]:
        if owner_id is None:
            return None
        return str(owner_id)

    @classmethod
    def _owner_key(cls, owner_id: Optional[Any]) -> str:
        normalized = cls._normalize_owner_id(owner_id)
        return normalized if normalized else cls.OWNER_GLOBAL_KEY

    @classmethod
    def _restore_owner_id(cls, owner_id: Optional[str]) -> Optional[str]:
        if not owner_id or owner_id == cls.OWNER_GLOBAL_KEY:
            return None
        return owner_id

    def __init__(self, config: "PrunedHippoRAGIGraphConfig"):
        """
        Initialize the Pruned HippoRAG Graph Store.

        Args:
            config: Configuration object containing all storage parameters
        """
        super().__init__(config)
        self._rwlock = RWLock()
        self._chunk_embeddings_lock = threading.Lock()

        # Initialize embedding model
        self.embedding_model = config.embedding.build()

        # Initialize undirected graph
        self.graph = ig.Graph(directed=False)

        # Storage configuration (resolve virtual io:// into a local directory early).
        storage_path = getattr(config, "storage_path", "io://graph_index")
        if is_io_path(storage_path):
            storage_path = str(resolve_io_to_local_path(storage_path))
        fallback_root = os.getenv("RAGARC_RUNTIME_DIR", "io://runtime")
        if is_io_path(fallback_root):
            fallback_root = str(resolve_io_to_local_path(fallback_root))
        fallback_storage = os.path.join(str(fallback_root), "graph_index")
        resolved_storage = ensure_writable_dir(str(storage_path), fallback_storage)
        self.storage_path = resolved_storage
        setattr(self.config, "storage_path", resolved_storage)
        self.index_name = getattr(config, "index_name", "index")

        # Initialize FAISS indices for facts and entities
        self._init_faiss_indices()

        # Initialize SQLite database for metadata
        self._init_sqlite_db()

        # In-memory chunk embeddings (not stored in FAISS)
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None

        # Graph node mappings
        self.node_to_idx = {}  # node_id -> graph_index
        self.idx_to_node = {}  # graph_index -> node_id
        self.node_to_node_stats = defaultdict(float)  # (node_id, node_id) -> edge_weight

        # Synonymy edge configuration
        self.add_synonymy_edges = getattr(config, 'add_synonymy_edges', False)
        self.synonymy_edge_topk = getattr(config, 'synonymy_edge_topk', 100)
        self.synonymy_edge_sim_threshold = getattr(config, 'synonymy_edge_sim_threshold', 0.8)

        logger.info("Pruned HippoRAG graph store initialized")
        logger.info(f"  - Fact index: FAISS Flat (exact search)")
        logger.info(f"  - Entity index: FAISS HNSW (synonymy edges)")
        logger.info(f"  - Chunk index: numpy array (brute-force search)")
        logger.info(f"  - Metadata: SQLite")
        logger.info(f"  - Graph: igraph")

    def read_lock(self):
        return self._rwlock.read_lock()

    def write_lock(self):
        return self._rwlock.write_lock()
