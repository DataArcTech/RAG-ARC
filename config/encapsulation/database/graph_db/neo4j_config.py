from framework.config import AbstractConfig
from encapsulation.database.graph_db.neo4j import Neo4jGraphStore
from typing import Literal
from pydantic import Field

class Neo4jConfig(AbstractConfig):
    """Neo4j Graph Store Configuration Class"""
    type: Literal["neo4j"] = "neo4j"

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

    def build(self):
        return Neo4jGraphStore(self)