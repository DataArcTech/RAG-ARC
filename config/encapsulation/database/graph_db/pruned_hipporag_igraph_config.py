"""
Configuration for Pruned HippoRAG Graph Store (igraph + FAISS + SQLite)

This uses:
- FAISS Flat for fact retrieval (exact search)
- FAISS HNSW for entity synonymy edges (approximate search)
- numpy arrays for chunk embeddings (brute-force search)
- SQLite for metadata storage
- igraph for graph structure and PPR
"""

from typing import Union, Annotated, Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.database.graph_db.pruned_hipporag_igraph import PrunedHippoRAGIGraphStore


class PrunedHippoRAGIGraphConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Graph Store

    This graph store uses:
    1. FAISS Flat for fact retrieval (exact search, need all scores)
    2. FAISS HNSW for entity synonymy edges (approximate search, speed up)
    3. numpy arrays for chunk embeddings (brute-force search in small subgraphs)
    4. SQLite for metadata storage (SQL queries, transactions, reference counting)
    5. igraph for graph structure (PPR computation)
    """

    type: Literal["pruned_hipporag_igraph"] = "pruned_hipporag_igraph"

    # Storage configuration
    storage_path: str = Field(
        default="./data/graph_index",
        description="Path to store the graph index"
    )
    index_name: str = Field(
        default="index",
        description="Name of the index"
    )

    # Embedding configuration
    embedding: Annotated[
        Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig],
        Field(discriminator="type")
    ] = Field(
        description="Embedding model configuration"
    )

    # Synonymy edge configuration
    add_synonymy_edges: bool = Field(
        default=True,
        description="Whether to add synonymy edges between similar entities"
    )
    synonymy_edge_topk: int = Field(
        default=100,
        description="Number of top-k similar entities to connect with synonymy edges"
    )
    synonymy_edge_sim_threshold: float = Field(
        default=0.8,
        description="Minimum similarity threshold for synonymy edges"
    )

    # HNSW configuration (for entity synonymy edges)
    hnsw_M: int = Field(
        default=32,
        description="HNSW M parameter (number of connections per layer)"
    )
    hnsw_ef_construction: int = Field(
        default=200,
        description="HNSW ef_construction parameter (search depth during construction)"
    )
    hnsw_ef_search: int = Field(
        default=100,
        description="HNSW ef_search parameter (search depth during query)"
    )

    def build(self):
        """Build the graph store"""
        return PrunedHippoRAGIGraphStore(config=self)

