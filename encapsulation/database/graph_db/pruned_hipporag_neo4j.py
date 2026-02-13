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

import faiss
import neo4j

from core.knowledge_graph.schema import KGSchemaConfig, load_schema_from_yaml
from encapsulation.database.graph_db.base import GraphStore
from encapsulation.database.graph_db.pruned_hipporag_neo4j_cache import _PrunedHippoRAGNeo4jCacheMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _PrunedHippoRAGNeo4jIndexingMixin
from encapsulation.database.graph_db.pruned_hipporag_neo4j_kg_maintenance import _PrunedHippoRAGNeo4jKGMaintenanceMixin
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
    _PrunedHippoRAGNeo4jKGMaintenanceMixin,
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

        # KG schema governance (relation whitelist/normalization/provenance knobs).
        self.kg_schema: KGSchemaConfig | None = None
        schema_path = getattr(config, "kg_schema_path", None)
        if schema_path:
            try:
                if os.path.exists(str(schema_path)):
                    self.kg_schema = load_schema_from_yaml(schema_path)
                    logger.info("Loaded KG schema: path=%s version=%s", schema_path, self.kg_schema.version)
                else:
                    logger.warning("KG schema file not found (skipping): path=%s", schema_path)
            except Exception as exc:
                logger.error("Failed to load KG schema (skipping): path=%s error=%s", schema_path, exc)
        self.kg_schema_loaded = self.kg_schema is not None
        self.kg_schema_version = getattr(self.kg_schema, "version", None) if self.kg_schema_loaded else None
        logger.info(
            "KG schema status: loaded=%s version=%s path=%s",
            self.kg_schema_loaded,
            self.kg_schema_version,
            schema_path,
        )

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
        storage_path = config.storage_path
        fallback_storage = os.path.join(
            os.getenv("RAGARC_RUNTIME_DIR", "io://runtime"),
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
        # Owner bookkeeping for chunk embedding persistence (IO sharding).
        self._chunk_embedding_owner_by_chunk_id: Dict[str, Optional[str]] = {}
        self._chunk_ids_by_owner: Dict[Optional[str], Set[str]] = {}
        self._chunk_embeddings_dirty_owners: Set[Optional[str]] = set()

        # Chunk embeddings optimization settings
        self.use_float16_embeddings = config.use_float16_embeddings
        self.normalize_chunk_embeddings = config.normalize_chunk_embeddings

        # In-memory graph cache for fast neighbor lookups
        self._graph_cache: Optional[Dict[str, Dict[str, List[Tuple[str, float]]]]] = None
        self._cache_loaded = False

        # Entity chunk count cache (computed from graph cache)
        self._entity_chunk_count_cache: Optional[Dict[str, Dict[str, int]]] = None
        
        # Cache version counter - incremented on any modification (add/delete)
        self._cache_version: int = 0

        self.index_name = config.index_name

        # Synonymy edge configuration
        self.add_synonymy_edges = config.add_synonymy_edges
        self.synonymy_edge_topk = config.synonymy_edge_topk
        self.synonymy_edge_sim_threshold = config.synonymy_edge_sim_threshold
        self.synonymy_edge_min_entity_chars = config.synonymy_edge_min_entity_chars

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
            include_existence = self._supports_existence_constraints(session)
            for stmt in self.neo4j_schema_statements(include_existence_constraints=include_existence):
                try:
                    session.run(stmt)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to apply Neo4j schema statement: %s error=%s", stmt, exc)
            logger.info("Neo4j schema initialized/verified (Facts stored as relationships)")
        
        # Auto-migrate: extract source_file_id from metadata JSON for existing chunks
        self._migrate_source_file_id_from_metadata()

    @staticmethod
    def neo4j_schema_statements(*, include_existence_constraints: bool = True) -> List[str]:
        """
        Return Neo4j DDL statements required by HippoRAG + DeepSearch Cypher tools.

        Notes:
        - Neo4j cannot enforce relationship-level uniqueness constraints; `fact_id` uniqueness is ensured
          by construction (owner-scoped hash) and MERGE patterns. We still create relationship indexes
          and existence constraints for query stability and diagnostics.
        """
        statements = [
            # Chunk nodes
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX chunk_owner IF NOT EXISTS FOR (c:Chunk) ON (c.owner_id)",
            "CREATE INDEX chunk_source_file_id IF NOT EXISTS FOR (c:Chunk) ON (c.source_file_id)",
            "CREATE INDEX chunk_owner_source_page_start IF NOT EXISTS FOR (c:Chunk) ON (c.owner_id, c.source_file_id, c.page_start)",
            "CREATE INDEX chunk_owner_source_page_end IF NOT EXISTS FOR (c:Chunk) ON (c.owner_id, c.source_file_id, c.page_end)",
            # Entity nodes
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            # Legacy constraint (owner_id, entity_name) prevented representing same-name different-type entities.
            "DROP CONSTRAINT entity_owner_name_unique IF EXISTS",
            "CREATE CONSTRAINT entity_owner_name_type_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.owner_id, e.entity_name_normalized, e.entity_type_key) IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)",
            "CREATE INDEX entity_owner_name IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_name)",
            "CREATE INDEX entity_owner_name_normalized IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_name_normalized)",
            "CREATE INDEX entity_owner_type IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_type)",
            "CREATE INDEX entity_owner_type_key IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_type_key)",
            "CREATE INDEX entity_owner_canonical_name IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_canonical_name)",
            "CREATE INDEX entity_owner_canonical_key IF NOT EXISTS FOR (e:Entity) ON (e.owner_id, e.entity_canonical_key)",
            # Relationships: MENTIONS
            "CREATE INDEX mentions_owner IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.owner_id)",
            # Relationships: RELATES_TO (facts)
            "CREATE INDEX relates_fact_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.fact_id)",
            "CREATE INDEX relates_owner_fact_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.owner_id, r.fact_id)",
            "CREATE INDEX relates_owner_predicate IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.owner_id, r.predicate)",
            # Relationships: SIMILAR_TO (synonymy)
            "CREATE INDEX similar_owner IF NOT EXISTS FOR ()-[r:SIMILAR_TO]-() ON (r.owner_id)",
            # Nodes: KG ingest metadata (per owner).
            "CREATE CONSTRAINT kg_ingest_meta_owner_unique IF NOT EXISTS FOR (m:KGIngestMeta) REQUIRE m.owner_id IS UNIQUE",
            # Nodes: canonical entity keys (queryable alias/canonicalization layer).
            "CREATE CONSTRAINT entity_canonical_id_unique IF NOT EXISTS FOR (c:EntityCanonical) REQUIRE c.canonical_id IS UNIQUE",
            "CREATE INDEX entity_canonical_owner_key IF NOT EXISTS FOR (c:EntityCanonical) ON (c.owner_id, c.canonical_key)",
            "CREATE INDEX entity_canonical_owner_name IF NOT EXISTS FOR (c:EntityCanonical) ON (c.owner_id, c.canonical_name)",
            # Nodes: entity aliases (queryable alias resolution).
            "CREATE CONSTRAINT entity_alias_id_unique IF NOT EXISTS FOR (a:EntityAlias) REQUIRE a.alias_id IS UNIQUE",
            "CREATE INDEX entity_alias_owner_text IF NOT EXISTS FOR (a:EntityAlias) ON (a.owner_id, a.alias_text_normalized)",
            # Relationships: CANONICAL_OF / ALIAS_OF.
            "CREATE INDEX canonical_of_owner IF NOT EXISTS FOR ()-[r:CANONICAL_OF]-() ON (r.owner_id)",
            "CREATE INDEX alias_of_owner IF NOT EXISTS FOR ()-[r:ALIAS_OF]-() ON (r.owner_id)",
            # Nodes: entity mentions (occurrence materialization, used by KG maintenance).
            "CREATE CONSTRAINT entity_mention_id_unique IF NOT EXISTS FOR (m:EntityMention) REQUIRE m.mention_id IS UNIQUE",
            "CREATE INDEX entity_mention_owner_chunk IF NOT EXISTS FOR (m:EntityMention) ON (m.owner_id, m.chunk_id)",
            "CREATE INDEX entity_mention_owner_surface IF NOT EXISTS FOR (m:EntityMention) ON (m.owner_id, m.surface_entity_id)",
            # Nodes: entity identities (disambiguation clusters / real-world entity centers).
            "CREATE CONSTRAINT entity_identity_id_unique IF NOT EXISTS FOR (i:EntityIdentity) REQUIRE i.identity_id IS UNIQUE",
            "CREATE INDEX entity_identity_owner_type IF NOT EXISTS FOR (i:EntityIdentity) ON (i.owner_id, i.entity_type_key)",
            "CREATE INDEX entity_identity_owner_surface IF NOT EXISTS FOR (i:EntityIdentity) ON (i.owner_id, i.surface_entity_id)",
            # Relationships: RESOLVED_TO / SAME_AS.
            "CREATE INDEX resolved_to_owner IF NOT EXISTS FOR ()-[r:RESOLVED_TO]-() ON (r.owner_id)",
            "CREATE INDEX same_as_owner IF NOT EXISTS FOR ()-[r:SAME_AS]-() ON (r.owner_id)",
        ]
        if not include_existence_constraints:
            return statements

        existence = [
            "CREATE CONSTRAINT chunk_owner_required IF NOT EXISTS FOR (c:Chunk) REQUIRE c.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_owner_required IF NOT EXISTS FOR (e:Entity) REQUIRE e.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_type_required IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_type IS NOT NULL",
            "CREATE CONSTRAINT entity_type_key_required IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_type_key IS NOT NULL",
            "CREATE CONSTRAINT entity_name_normalized_required IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_name_normalized IS NOT NULL",
            "CREATE CONSTRAINT mentions_owner_required IF NOT EXISTS FOR ()-[r:MENTIONS]-() REQUIRE r.owner_id IS NOT NULL",
            "CREATE CONSTRAINT relates_owner_required IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.owner_id IS NOT NULL",
            "CREATE CONSTRAINT relates_fact_id_required IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.fact_id IS NOT NULL",
            "CREATE CONSTRAINT relates_predicate_required IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.predicate IS NOT NULL",
            "CREATE CONSTRAINT similar_owner_required IF NOT EXISTS FOR ()-[r:SIMILAR_TO]-() REQUIRE r.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_canonical_owner_required IF NOT EXISTS FOR (c:EntityCanonical) REQUIRE c.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_canonical_key_required IF NOT EXISTS FOR (c:EntityCanonical) REQUIRE c.canonical_key IS NOT NULL",
            "CREATE CONSTRAINT entity_alias_owner_required IF NOT EXISTS FOR (a:EntityAlias) REQUIRE a.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_alias_text_required IF NOT EXISTS FOR (a:EntityAlias) REQUIRE a.alias_text_normalized IS NOT NULL",
            "CREATE CONSTRAINT canonical_of_owner_required IF NOT EXISTS FOR ()-[r:CANONICAL_OF]-() REQUIRE r.owner_id IS NOT NULL",
            "CREATE CONSTRAINT alias_of_owner_required IF NOT EXISTS FOR ()-[r:ALIAS_OF]-() REQUIRE r.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_mention_owner_required IF NOT EXISTS FOR (m:EntityMention) REQUIRE m.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_mention_chunk_required IF NOT EXISTS FOR (m:EntityMention) REQUIRE m.chunk_id IS NOT NULL",
            "CREATE CONSTRAINT entity_mention_surface_required IF NOT EXISTS FOR (m:EntityMention) REQUIRE m.surface_entity_id IS NOT NULL",
            "CREATE CONSTRAINT entity_identity_owner_required IF NOT EXISTS FOR (i:EntityIdentity) REQUIRE i.owner_id IS NOT NULL",
            "CREATE CONSTRAINT entity_identity_type_required IF NOT EXISTS FOR (i:EntityIdentity) REQUIRE i.entity_type_key IS NOT NULL",
        ]
        return statements + existence

    @staticmethod
    def _supports_existence_constraints(session) -> bool:  # noqa: ANN001
        """Return True when the connected Neo4j supports property existence constraints.

        Neo4j Community does not support them; Enterprise does.
        """

        try:
            records = list(session.run("CALL dbms.components() YIELD edition RETURN edition AS edition"))
            for record in records:
                try:
                    edition = str(record.get("edition") or "").strip().lower()
                except Exception:
                    continue
                if edition:
                    return "enterprise" in edition
        except Exception:
            # Conservative default: skip existence constraints to avoid noisy warnings.
            return False
        return False

    def _migrate_source_file_id_from_metadata(self) -> None:
        """Auto-migrate: backfill `Chunk.source_file_id` from `Chunk.metadata` JSON.

        This is best-effort and capped to avoid long startup stalls; repeated startups can gradually
        migrate large graphs.
        """

        try:
            with self._driver.session(database=self.database) as session:
                query = """
                MATCH (c:Chunk)
                WHERE c.metadata IS NOT NULL
                  AND c.source_file_id IS NULL
                RETURN c.chunk_id AS chunk_id, c.metadata AS metadata
                LIMIT 1000
                """
                results = session.run(query)
                chunks_to_update: list[dict[str, str]] = []
                for record in results:
                    chunk_id = record.get("chunk_id")
                    if not chunk_id:
                        continue
                    metadata_raw = record.get("metadata")
                    if not metadata_raw:
                        continue
                    try:
                        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    source_file_id = str(
                        metadata.get("source_file_id")
                        or metadata.get("sourceFileId")
                        or metadata.get("file_id")
                        or metadata.get("fileId")
                        or metadata.get("document_id")
                        or metadata.get("documentId")
                        or metadata.get("doc_id")
                        or metadata.get("docId")
                        or ""
                    ).strip()
                    if source_file_id:
                        chunks_to_update.append({"chunk_id": chunk_id, "source_file_id": source_file_id})

                if not chunks_to_update:
                    logger.debug("No chunks need source_file_id migration")
                    return

                update_query = """
                UNWIND $chunks AS chunk
                MATCH (c:Chunk {chunk_id: chunk.chunk_id})
                SET c.source_file_id = chunk.source_file_id
                """
                session.run(update_query, {"chunks": chunks_to_update})
                logger.info("Auto-migrated source_file_id for %d chunks", len(chunks_to_update))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to migrate source_file_id from metadata: %s", exc)

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

    @staticmethod
    def _coerce_source_file_name_from_metadata(metadata: Any) -> str:
        if not metadata:
            return ""
        try:
            meta = json.loads(metadata) if isinstance(metadata, str) else metadata
        except Exception:
            return ""
        if not isinstance(meta, dict):
            return ""
        for key in ("source_file_name", "filename", "source_file", "file_path", "path"):
            val = str(meta.get(key) or "").strip()
            if val:
                return val
        return ""

    def get_batch_chunk_source_file_names(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Resolve `source_file_name` (or equivalent) for a set of Chunk IDs.

        This is used by retrieval-time heuristics (e.g., dense_file_prior gating) and should be
        low-latency: a single UNWIND query and lightweight metadata parsing.
        """
        ids = [str(cid) for cid in (chunk_ids or []) if str(cid).strip()]
        if not ids:
            return {}

        query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {chunk_id: chunk_id})
        RETURN c.chunk_id AS chunk_id, c.metadata AS metadata
        """
        rows = self._execute_query(query, {"chunk_ids": ids}) or []
        out: Dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("chunk_id") or "").strip()
            if not cid:
                continue
            out[cid] = self._coerce_source_file_name_from_metadata(row.get("metadata"))
        return out
