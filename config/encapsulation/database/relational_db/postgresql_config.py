"""Configuration for PostgreSQL relational database"""

import os
from framework.config import AbstractConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from typing import Literal


class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL relational database"""
    # Discriminator for config type identification
    type: Literal["postgresql"] = "postgresql"

    # Database connection configuration (read from environment variables)
    host: str = os.getenv("POSTGRES_HOST", "localhost")  # PostgreSQL server host
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))  # PostgreSQL server port
    database: str = os.getenv("POSTGRES_DB", "rag_arc")  # Database name
    user: str = os.getenv("POSTGRES_USER", "postgres")  # Database username
    password: str = os.getenv("POSTGRES_PASSWORD", "123")  # Database password

    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)