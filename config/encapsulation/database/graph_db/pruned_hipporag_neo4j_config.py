import os
from typing import Annotated, Literal, Union
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore


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
        default_factory=lambda: os.getenv("GRAPH_STORAGE_PATH", "./data/graph_index_neo4j"),
        description="Directory path for storing FAISS index files"
    )
    index_name: str = Field(
        default_factory=lambda: os.getenv("GRAPH_INDEX_NAME", "index"),
        description="Name prefix for index files"
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

    # Embedding model configuration
    embedding: Annotated[
        Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig],
        Field(discriminator="type")
    ]

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

    def build(self):
        """Build and return a PrunedHippoRAGNeo4jStore instance."""
        return PrunedHippoRAGNeo4jStore(config=self)
