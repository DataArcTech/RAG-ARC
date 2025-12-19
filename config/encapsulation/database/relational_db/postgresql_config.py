"""Configuration for PostgreSQL relational database"""

import os
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from typing import Literal


class PostgreSQLConfig(AbstractConfig):
    """Configuration for PostgreSQL relational database"""
    # Discriminator for config type identification
    type: Literal["postgresql"] = "postgresql"

    # Database connection configuration (read from environment variables)
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: str = Field(default_factory=lambda: os.getenv("POSTGRES_PORT", "5555"))
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "rag_arc"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "123"))

    def build(self) -> PostgreSQLDB:
        return PostgreSQLDB(self)
