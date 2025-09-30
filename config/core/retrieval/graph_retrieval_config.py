from typing import Optional, Literal, Union, Annotated
from pydantic import Field


from framework.config import AbstractConfig
from config.encapsulation.database.graph_db.neo4j_with_embedding_config import Neo4jVectorConfig
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.retrieval.graph_retrieveal.graph_retrieval import GraphRetrieval
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig



class GraphRetrievalConfig(AbstractConfig):
    """Graph Retrieval Configuration"""
    type: Literal["graph_retrieval"] = "graph_retrieval"

    # Graph database configuration (supports both Neo4j and NetworkX)
    graph_config: Annotated[Union[Neo4jVectorConfig, NetworkXVectorConfig], Field(discriminator="type")]

    embedding_config: Annotated[Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig], Field(discriminator="type")]

    # LLM configuration for entity filtering (optional)
    llm_config: Optional[OpenAIChatConfig] = Field(default=None, description="LLM configuration for entity filtering")

    # Retrieval parameters
    k1_chunks: int = Field(default=20, description="Number of chunks to retrieve in semantic search")
    k2_entities: int = Field(default=5, description="Number of entities to retrieve in semantic search")
    
    # Subgraph construction parameters
    max_hops: int = Field(default=5, description="Maximum hops for subgraph expansion")
    beam_size: int = Field(default=20, description="Beam size for subgraph expansion")
    
    # PPR parameters
    damping_factor: float = Field(default=0.85, description="PPR damping factor")
    max_iterations: int = Field(default=50, description="Maximum PPR iterations")
    tolerance: float = Field(default=1e-6, description="PPR convergence tolerance")
    
    # Scoring parameters
    beta1: float = Field(default=0.7, description="Weight for entity similarity in personalization")
    beta2: float = Field(default=0.3, description="Weight for triple boost in personalization")
    
    # Chunk scoring parameters
    mu1: float = Field(default=0.3, description="Weight for mention count in chunk scoring")
    mu2: float = Field(default=0.3, description="Weight for TF-IDF in chunk scoring")
    mu3: float = Field(default=0.4, description="Weight for embedding similarity in chunk scoring")
    
    # Path scoring parameters
    gamma1: float = Field(default=0.4, description="Weight for path length in path scoring")
    gamma2: float = Field(default=0.3, description="Weight for edge weights in path scoring")
    gamma3: float = Field(default=0.3, description="Weight for entity specificity in path scoring")
    
    # Entity scoring parameters
    lambda1: float = Field(default=0.6, description="Weight for path score in entity scoring")
    lambda2: float = Field(default=0.4, description="Weight for PPR score in entity scoring")
    
    # Coverage parameters
    eta: float = Field(default=0.2, description="Coverage boost factor")
    top_k_entities: int = Field(default=10, description="Top K entities for coverage calculation")
    
    # Final fusion parameters
    alpha: float = Field(default=0.6, description="Weight for graph score in final fusion")
    beta: float = Field(default=0.4, description="Weight for embedding score in final fusion")
    
    # Chunk backtracking parameters
    chunks_per_entity: int = Field(default=10, description="Number of chunks to retrieve per entity")
    
    def build(self):
        return GraphRetrieval(self)