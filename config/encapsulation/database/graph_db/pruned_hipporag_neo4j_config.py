import os
from typing import Annotated, Literal, Union, Optional, Dict, Any, List
from pydantic import Field, model_validator

from framework.config import AbstractConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore


class GraphEmbeddingCacheConfig(AbstractConfig):
    """
    Owner-scoped embedding cache for graph indexing.

    Goals
    - Speed up repeated ingestion for the same tenant when content overlaps across files/versions/runs.
    - Never share cached embeddings across owners (multi-tenant isolation).
    - Keep IO low: batch reads/writes, single sqlite file per owner under storage_path.
    """

    type: Literal["graph_embedding_cache"] = "graph_embedding_cache"

    enabled: bool = Field(
        default=True,
        description="Enable the owner-scoped sqlite embedding cache for graph indexing.",
    )
    dirname: str = Field(
        default="embedding_cache",
        description="Subdirectory under storage_path for embedding cache DB files.",
    )
    db_filename: str = Field(
        default="embeddings.sqlite3",
        description="Filename for each owner-scoped sqlite DB.",
    )
    max_in_keys_per_query: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Chunk sqlite IN (...) lookups to avoid oversized queries.",
    )


class KGMaintenanceL0BudgetConfig(AbstractConfig):
    """
    L0 KG maintenance budget policy for the "new file arrived" hot path.

    Goal
    - Improve graph quality quickly without degrading user-perceived indexing readiness.
    - Use a relative budget (ratio) so laptop/server deployments can share defaults.

    Policy
    - Estimate expected seconds from workload size / throughput_estimate.
    - Allow up to expected * ratio, clamped to [min_seconds, max_seconds].
    - If we hit the budget mid-run, we stop and defer the remainder to background L1/L2.
    """

    type: Literal["kg_maintenance_l0_budget"] = "kg_maintenance_l0_budget"

    ratio: float = Field(
        default=1.2,
        ge=1.0,
        le=10.0,
        description="Defer L0 when actual runtime exceeds expected_seconds * ratio (after clamping).",
    )
    estimator: Literal["mentions_throughput_ema"] = Field(
        default="mentions_throughput_ema",
        description="How to estimate expected runtime for L0.",
    )
    min_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Lower bound for computed L0 budget seconds (stability against jitter).",
    )
    max_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=600.0,
        description="Upper bound for computed L0 budget seconds (protects readiness latency).",
    )
    ema_alpha: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description="EMA alpha for updating mentions/sec throughput estimate per deployment/process.",
    )


class KGMaintenanceL0Config(AbstractConfig):
    """
    L0 KG maintenance: create entity mentions (occurrences) for a new file quickly.

    Notes
    - This is hot-path and must be budgeted.
    - L0 does NOT run disambiguation/identity alignment; it only materializes mention evidence and
      produces a bounded "affected surface set" for later L1.
    """

    type: Literal["kg_maintenance_l0"] = "kg_maintenance_l0"

    enabled: bool = Field(default=True, description="Enable L0 mention materialization after file indexing.")
    budget: KGMaintenanceL0BudgetConfig = Field(default_factory=KGMaintenanceL0BudgetConfig)

    batch_size: int = Field(
        default=0,
        ge=0,
        le=5000,
        description=(
            "UNWIND batch size for L0 mention upserts into Neo4j. "
            "0 means 'reuse neo4j_ingest_chunk_batch_size'."
        ),
    )
    max_mentions_to_write: int = Field(
        default=0,
        ge=0,
        le=5_000_000,
        description="Hard cap on mention writes per L0 run. 0 disables the cap (use budget_ratio only).",
    )

    # Optional: generate a bounded "affected surfaces" set for later L1 (not used by L0 upserts directly).
    max_surfaces: int = Field(
        default=0,
        ge=0,
        le=200_000,
        description="Cap the number of surface entities returned as the affected set for later maintenance. 0 disables.",
    )
    neighbor_hops: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Optional neighbor expansion hops for the affected set (0 disables expansion).",
    )
    max_neighbors: int = Field(
        default=0,
        ge=0,
        le=200_000,
        description="Cap expanded neighbors when neighbor_hops > 0. 0 disables.",
    )

    # Metadata extraction policy (time/version).
    source_version_metadata_keys: List[str] = Field(
        default_factory=lambda: ["source_version", "version", "doc_version", "edition", "revision"],
        description="Metadata keys (in order) used to extract a document/source version label for mentions.",
    )


class KGMaintenanceConfig(AbstractConfig):
    type: Literal["kg_maintenance"] = "kg_maintenance"

    l0: KGMaintenanceL0Config = Field(default_factory=KGMaintenanceL0Config)

    # L1/L2 are background quality maintenance and are disabled by default until tuned.
    # They can be executed via scripts or scheduled jobs.
    l1_enabled: bool = Field(
        default=True,
        description="Enable L1 background entity disambiguation + identity alignment maintenance.",
    )
    l1_disambiguation_min_cluster_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity to assign a mention embedding to an existing identity cluster during "
            "greedy disambiguation. When None, reuse synonymy_edge_sim_threshold."
        ),
    )
    l1_min_mentions_to_split: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        description="Only attempt splitting a surface entity when it has at least this many mentions. 0 means no minimum.",
    )
    l1_alignment_nn_topk: int | None = Field(
        default=None,
        ge=0,
        le=1000,
        description=(
            "Top-K nearest neighbor identities to consider for SAME_AS alignment. "
            "When None, reuse synonymy_edge_topk. 0 disables alignment."
        ),
    )
    l1_alignment_min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for creating SAME_AS edges. When None, reuse synonymy_edge_sim_threshold.",
    )
    l1_alignment_gradient_mink: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Gradient truncation: mink. 0 means 'no gradient truncation' (use topk only).",
    )
    l1_alignment_gradient_g: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Gradient truncation: g. Only used when gradient_mink > 0.",
    )
    l1_require_time_overlap_for_merge: bool = Field(
        default=True,
        description="Guardrail: if time overlap is decisively false, avoid strong merge actions (we still allow SAME_AS).",
    )


class PrunedHippoRAGNeo4jConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Graph Store with Neo4j backend.

    This graph store uses a hybrid storage approach:
    - Facts: FAISS Flat index (exact search for fact retrieval)
    - Entities: FAISS HNSW index (approximate nearest neighbor for synonymy edges)
    - Chunks: In-memory numpy array (brute-force dense retrieval)
    - Metadata & Graph: Neo4j database (chunks, entities, facts, relations, graph structure)
    - PageRank: Extracted to igraph for computation

    The graph structure in Neo4j connects:
    - Chunks to entities (extracted from chunk content)
    - Entities to entities (via facts/relations)
    - Entities to entities (via synonymy edges based on embedding similarity)
    """
    type: Literal["pruned_hipporag_neo4j"] = "pruned_hipporag_neo4j"

    # Neo4j connection configuration
    url: str = Field(
        default_factory=lambda: os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        description="Neo4j database connection URL, e.g.: bolt://localhost:7687",
    )
    username: str = Field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"),
        description="Database username",
    )
    password: str = Field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""),
        description="Database password",
    )
    database: str = Field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"),
        description="Database name",
    )

    # Storage configuration for FAISS indices
    storage_path: str = Field(
        default_factory=lambda: os.getenv("GRAPH_STORAGE_PATH", "io://graph_index_neo4j"),
        description="Directory path for storing FAISS index files"
    )
    index_name: str = Field(
        default_factory=lambda: os.getenv("GRAPH_INDEX_NAME", "index"),
        description="Name prefix for index files"
    )

    chunk_embeddings_owner_sharded: bool = Field(
        default=True,
        description=(
            "When true (default), persist chunk embeddings as owner-scoped shards under "
            "`storage_path/<chunk_embeddings_dirname>/` to avoid a single ever-growing pickle file. "
            "This reduces write amplification and improves multi-owner isolation."
        ),
    )
    chunk_embeddings_dirname: str = Field(
        default="chunk_embeddings",
        description="Subdirectory name under storage_path for persisted chunk embedding shards.",
    )

    # KG schema governance (YAML)
    kg_schema_path: str = Field(
        default_factory=lambda: os.getenv("KG_SCHEMA_PATH", "./kg_schema.yml"),
        description="Path to KG schema YAML (relation whitelist/normalization/provenance knobs)",
    )

    fact_provenance_max_source_chunks: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Max chunk ids to retain per fact edge in Neo4j (`RELATES_TO.source_chunk_ids`). 0 disables storage.",
    )

    neo4j_ingest_chunk_batch_size: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description=(
            "Chunk batch size per Neo4j ingestion transaction during graph indexing. "
            "Lower values reduce single-tx pressure under concurrency; higher values reduce overhead."
        ),
    )

    chunk_upsert_policy: Literal["replace", "append"] = Field(
        default_factory=lambda: (os.getenv("KG_CHUNK_UPSERT_POLICY", "replace") or "replace").strip().lower(),
        description=(
            "When updating existing chunk_id(s), choose how to handle old graph evidence:\n"
            "- replace: remove old evidence contributed by those chunk_id(s) (MENTIONS + RELATES_TO provenance cleanup)\n"
            "- append: keep old evidence and only add new (legacy behavior; can cause drift)\n"
            "Note: replace requires fact_provenance_max_source_chunks > 0 to be correct."
        ),
    )

    enable_endpoint_canonical_fallback: bool = Field(
        default=True,
        description=(
            "When a triple endpoint is not found in the extracted NER entities, attempt a precision-first "
            "fallback match using KG schema entity canonicalization (e.g., suffix stripping). "
            "Requires an unambiguous unique match; otherwise the triple is still dropped."
        ),
    )

    enable_schema_layer_nodes: bool = Field(
        default=False,
        description=(
            "Persist schema-layer nodes derived from chunk mindmaps into Neo4j (Concept/Process/Instance scaffolding). "
            "Disabled by default to avoid extra writes for general-domain deployments."
        ),
    )

    schema_layer_max_nodes_per_chunk: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Max mindmap nodes per chunk to convert into SchemaNode nodes when enable_schema_layer_nodes is true.",
    )

    enable_sdf_schema: bool = Field(
        default=False,
        description=(
            "Persist SDF process schema nodes into Neo4j (SDFEvent + SDF_BEFORE/HAS_SUBEVENT). "
            "Disabled by default for general-domain deployments."
        ),
    )

    sdf_max_events_per_chunk: int = Field(
        default=80,
        ge=1,
        le=2000,
        description="Max SDF events per chunk to persist when enable_sdf_schema is true.",
    )

    sdf_max_relations_per_chunk: int = Field(
        default=200,
        ge=0,
        le=5000,
        description="Max SDF relations per chunk to persist when enable_sdf_schema is true.",
    )

    sdf_provenance_max_source_chunks: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Max chunk ids to retain per SDF node/edge for provenance (`source_chunk_ids`). 0 disables storage.",
    )

    # Embedding model configuration
    embedding: Annotated[
        Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig],
        Field(discriminator="type")
    ]

    embedding_cache: GraphEmbeddingCacheConfig = Field(
        default_factory=GraphEmbeddingCacheConfig,
        description="Owner-scoped embedding cache settings for graph indexing.",
    )

    kg_maintenance: KGMaintenanceConfig = Field(
        default_factory=KGMaintenanceConfig,
        description="Long-term KG maintenance settings (L0/L1/L2; L0 is hot-path).",
    )

    # Synonymy edge configuration
    add_synonymy_edges: bool = Field(
        default=True,
        description="Whether to add synonymy edges between similar entities"
    )
    synonymy_edge_topk: int = Field(
        default=100,
        description="Number of top similar entities to consider for synonymy edges"
    )
    synonymy_edge_sim_threshold: float = Field(
        default=0.8,
        description="Minimum cosine similarity threshold for creating synonymy edges"
    )
    synonymy_edge_min_entity_chars: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "Minimum normalized character length for an entity to be considered for synonymy edges. "
            "Normalization removes punctuation/symbols and collapses whitespace; this works for non-Latin scripts."
        ),
    )

    # HNSW index parameters for entity embeddings
    hnsw_M: int = Field(
        default=32,
        description="HNSW parameter M: number of bi-directional links per node"
    )
    hnsw_ef_construction: int = Field(
        default=200,
        description="HNSW parameter efConstruction: size of dynamic candidate list during construction"
    )
    hnsw_ef_search: int = Field(
        default=100,
        description="HNSW parameter efSearch: size of dynamic candidate list during search"
    )

    # Chunk embeddings optimization
    use_float16_embeddings: bool = Field(
        default=True,
        description="Use float16 for chunk embeddings to reduce memory usage (recommended)"
    )
    normalize_chunk_embeddings: bool = Field(
        default=True,
        description="Normalize chunk embeddings to unit vectors for cosine similarity"
    )

    chunk_embedding_text_prefix_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Optional metadata keys (from stored chunk metadata JSON) to prefix to the embedded chunk text, "
            "e.g. ['filename']. This helps disambiguate similar products across companies by injecting file context."
        ),
    )
    chunk_embedding_filename_root: str | None = Field(
        default=None,
        description=(
            "If set and `filename` is used as a chunk embedding prefix key, trim the filename/path to start "
            "from this token (e.g. 'RAG-ARC')."
        ),
    )
    chunk_embedding_text_separator: str = Field(
        default="\n",
        description="Separator used when prefixing chunk metadata fields to the chunk embedding text.",
    )

    shared_instance: bool = Field(
        default=True,
        description=(
            "Whether to reuse a process-level shared PrunedHippoRAGNeo4jStore instance for identical configs "
            "(shared_module). Disable for strict isolation in tests/multi-tenant workers."
        ),
    )

    @model_validator(mode="after")
    def _validate_chunk_upsert_policy(self):
        if self.chunk_upsert_policy == "replace" and int(self.fact_provenance_max_source_chunks) == 0:
            raise ValueError(
                "chunk_upsert_policy=replace requires fact_provenance_max_source_chunks > 0 "
                "(otherwise the store cannot remove stale chunk evidence from RELATES_TO edges)."
            )
        return self

    def build(self):
        """Build and return a PrunedHippoRAGNeo4jStore instance."""
        store_cls = PrunedHippoRAGNeo4jStore
        if not bool(self.shared_instance):
            store_cls = getattr(PrunedHippoRAGNeo4jStore, "__wrapped__", PrunedHippoRAGNeo4jStore)
        return store_cls(config=self)
