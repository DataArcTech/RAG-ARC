from typing import Union, Annotated, Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.database.graph_db.pruned_hipporag_igraph import PrunedHippoRAGIGraphStore


class PrunedHippoRAGIGraphConfig(AbstractConfig):
    """
    Configuration for Pruned HippoRAG Graph Store.

    This graph store uses a hybrid storage approach:
    - Facts: FAISS Flat index (exact search for fact retrieval)
    - Entities: FAISS HNSW index (approximate nearest neighbor for synonymy edges)
    - Chunks: In-memory numpy array (brute-force dense retrieval)
    - Metadata: SQLite database (chunks, entities, facts, relations)
    - Graph: igraph (undirected graph for Personalized PageRank)

    The graph structure connects:
    - Chunks to entities (extracted from chunk content)
    - Entities to entities (via facts/relations)
    - Entities to entities (via synonymy edges based on embedding similarity)
    """
    type: Literal["pruned_hipporag_igraph"] = "pruned_hipporag_igraph"

    # Storage configuration
    storage_path: str = Field(
        default="./data/graph_index",
        description="Directory path for storing graph index files"
    )
    index_name: str = Field(
        default="index",
        description="Name prefix for index files"
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

    def build(self):
        """Build and return a PrunedHippoRAGIGraphStore instance."""
        return PrunedHippoRAGIGraphStore(config=self)

