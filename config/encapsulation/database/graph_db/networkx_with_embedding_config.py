from typing import Annotated, Union, Literal, Optional
from pydantic import Field
from encapsulation.database.graph_db.networkx_with_embedding import NetworkXVectorGraphStore
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from framework.config import AbstractConfig


class NetworkXVectorConfig(AbstractConfig):
    """NetworkX Vector Graph Store Configuration Class with embedding support"""
    type: Literal["networkx_vector"] = "networkx_vector"

    # Optional storage configuration for persistence
    storage_path: Optional[str] = Field(
        default=None,
        description="Optional path for saving/loading graph data. If None, graph is memory-only"
    )
    
    # Optional index name for persistence
    index_name: str = Field(
        default="networkx_vector_index",
        description="Name for the saved index files"
    )
    
    # Performance tuning options
    auto_save: bool = Field(
        default=False,
        description="Whether to automatically save changes to disk"
    )
    
    # Graph configuration
    allow_self_loops: bool = Field(
        default=True,
        description="Whether to allow self-loops in the graph"
    )
    
    allow_parallel_edges: bool = Field(
        default=True,
        description="Whether to allow parallel edges between nodes"
    )
    
    # Vector search configuration
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for search results",
        ge=0.0,
        le=1.0
    )
    
    max_search_results: int = Field(
        default=100,
        description="Maximum number of results to return in similarity search",
        gt=0
    )
    
    # Embedding caching options
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache generated embeddings in memory"
    )
    
    embedding_cache_size: int = Field(
        default=10000,
        description="Maximum number of embeddings to cache",
        gt=0
    )

    # Embedding configuration - supports same models as Neo4j
    embedding: Annotated[Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig], Field(discriminator="type")]

    def build(self):
        return NetworkXVectorGraphStore(self)