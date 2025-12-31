import os
import json
import logging
import pickle
import re
import random
import threading
import time
from typing import List, Dict, Any, Optional, Sequence, Set, Tuple, TYPE_CHECKING
import numpy as np
import igraph as ig
import warnings

warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)

import faiss
import neo4j

from encapsulation.database.graph_db.base import GraphStore
from encapsulation.database.graph_db.pruned_hipporag_neo4j_cache import _PrunedHippoRAGNeo4jCacheMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _PrunedHippoRAGNeo4jIndexingMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_persistence import _PrunedHippoRAGNeo4jPersistenceMixin
from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from core.utils.path_guard import ensure_writable_dir
from core.utils.rwlock import RWLock
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig

logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


@shared_module
class PrunedHippoRAGNeo4jStore(
    _PrunedHippoRAGNeo4jEmbeddingsMixin,
    _PrunedHippoRAGNeo4jCacheMixin,
    _PrunedHippoRAGNeo4jIndexingMixin,
    _PrunedHippoRAGNeo4jPersistenceMixin,
    GraphStore,
):
    """
    Pruned HippoRAG Graph Store using Neo4j backend.

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

    4. **Metadata & Graph**: Neo4j database
       - Stores chunks, entities, facts, and their relationships
       - Provides native graph queries and traversal
       - Nodes: Chunk, Entity
       - Relationships: MENTIONS (chunk->entity), RELATES_TO (entity->entity), SIMILAR_TO (entity->entity)

    5. **PageRank**: Extracted subgraph to igraph for computation
       - Subgraph extracted from Neo4j based on query
       - PageRank computed in-memory using igraph
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

    def __init__(self, config: "PrunedHippoRAGNeo4jConfig"):
        """
        Initialize the Pruned HippoRAG Graph Store with Neo4j backend.

        Args:
            config: Configuration object containing all storage parameters
        """
        super().__init__(config)
        self._rwlock = RWLock()

        # Initialize embedding model
        self.embedding_model = config.embedding.build()

        # Initialize Neo4j connection with notifications disabled
        self._driver = None
        try:
            self._driver: neo4j.Driver = neo4j.GraphDatabase.driver(
                config.url,
                auth=(config.username, config.password),
                notifications_min_severity="OFF"  # Disable all notifications
            )
            logger.info(f"Successfully connected to Neo4j database: {config.url}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
            raise

        self.database = config.database

        # Add conn property for compatibility with base class
        # (Neo4j doesn't use SQLite conn, but base class expects it)
        self.conn = None

        # Ensure storage path is writable
        storage_path = getattr(config, 'storage_path', './data/graph_index_neo4j')
        fallback_storage = os.path.join(
            os.getenv("RAGARC_RUNTIME_DIR", "./local/runtime"),
            "graph_index_neo4j"
        )
        resolved_storage = ensure_writable_dir(storage_path, fallback_storage)
        self.storage_path = resolved_storage
        setattr(self.config, 'storage_path', resolved_storage)

        # Initialize Neo4j schema (constraints and indices)
        self._init_neo4j_schema()

        # Initialize FAISS indices for facts and entities
        self._init_faiss_indices()

        # In-memory chunk embeddings (not stored in FAISS)
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None

        # Chunk embeddings optimization settings
        self.use_float16_embeddings = getattr(config, 'use_float16_embeddings', True)
        self.normalize_chunk_embeddings = getattr(config, 'normalize_chunk_embeddings', True)

        # In-memory graph cache for fast neighbor lookups
        self._graph_cache: Optional[Dict[str, Dict[str, List[Tuple[str, float]]]]] = None
        self._cache_loaded = False

        # Entity chunk count cache (computed from graph cache)
        self._entity_chunk_count_cache: Optional[Dict[str, Dict[str, int]]] = None
        
        # Cache version counter - incremented on any modification (add/delete)
        self._cache_version: int = 0

        self.index_name = getattr(config, 'index_name', 'index')

        # Synonymy edge configuration
        self.add_synonymy_edges = getattr(config, 'add_synonymy_edges', False)
        self.synonymy_edge_topk = getattr(config, 'synonymy_edge_topk', 100)
        self.synonymy_edge_sim_threshold = getattr(config, 'synonymy_edge_sim_threshold', 0.8)

        # Load graph structure into memory for fast lookups
        self._load_graph_cache()

        # Load chunk embeddings from disk if available
        self._load_chunk_embeddings()

        logger.info("Pruned HippoRAG Neo4j graph store initialized")
        logger.info(f"  - Fact index: FAISS Flat (exact search)")
        logger.info(f"  - Entity index: FAISS HNSW (synonymy edges)")
        logger.info(f"  - Chunk index: numpy array (brute-force search)")
        logger.info(f"  - Metadata & Graph: Neo4j (cached in memory)")
        logger.info(f"  - PageRank: igraph (extracted subgraph)")

    def read_lock(self):
        return self._rwlock.read_lock()

    def write_lock(self):
        return self._rwlock.write_lock()

    def _init_neo4j_schema(self):
        """
        Initialize Neo4j schema with constraints and indices.

        Creates:
        - Unique constraints on Chunk.chunk_id and Entity.entity_id
        - Indices on frequently queried properties
        - Note: Facts are now stored as relationships, not nodes
        """
        with self._driver.session(database=self.database) as session:
            # Chunk constraints and indices
            try:
                session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                session.run("CREATE INDEX chunk_owner IF NOT EXISTS FOR (c:Chunk) ON (c.owner_id)")
                logger.info("Created Chunk constraints and indices")
            except Exception as e:
                logger.warning(f"Failed to create Chunk constraints: {e}")

            # Entity constraints and indices
            try:
                session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)")
                logger.info("Created Entity constraints and indices")
            except Exception as e:
                logger.warning(f"Failed to create Entity constraints: {e}")

            # Note: Facts are now stored as :Fact relationships between entities
            # No separate Fact nodes needed
            logger.info("Schema initialized (Facts stored as relationships)")

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on Neo4j.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]



