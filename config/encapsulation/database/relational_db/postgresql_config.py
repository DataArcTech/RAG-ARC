"""Configuration for PostgreSQL relational database"""

from framework.config import AbstractConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from typing import Literal


class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL relational database"""
    # Discriminator for config type identification
    type: Literal["postgresql"] = "postgresql"

    # Database connection configuration
    host: str = "localhost"  # PostgreSQL server host
    port: int = 5432  # PostgreSQL server port
    database: str = "rag_arc_filestore_test"  # Database name
    user: str = "postgres"  # Database username
    password: str = "123"  # Database password

    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)