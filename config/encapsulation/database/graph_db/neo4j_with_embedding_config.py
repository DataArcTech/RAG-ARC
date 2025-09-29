
from typing import Annotated, Union, Literal
from pydantic import Field
from encapsulation.database.graph_db.neo4j_with_embedding import Neo4jVectorGraphStore
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from framework.config import AbstractConfig


class Neo4jVectorConfig(AbstractConfig):

    """Neo4j Vector Graph Store Configuration Class with embedding support"""
    type: Literal["neo4j_vector"] = "neo4j_vector"

    # Database connection configuration
    url: str = Field(
        description="Neo4j database connection URL, e.g.: bolt://localhost:7687"
    )
    username: str = Field(
        description="Database username"
    )
    password: str = Field(
        description="Database password"
    )
    database: str = Field(
        default="neo4j",
        description="Database name"
    )

    # Embedding configuration
    embedding: Annotated[Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig], Field(discriminator="type")]

    def build(self):
        return Neo4jVectorGraphStore(self)