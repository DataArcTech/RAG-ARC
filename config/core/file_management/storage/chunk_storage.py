"""Configuration for ChunkStorage (Core Layer)"""

from typing import Annotated, Literal, Union

from framework.config import AbstractConfig
from pydantic import Field

from core.file_management.storage.chunk import ChunkStorage
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.file_db.minio_config import MinIOConfig
from config.encapsulation.database.file_db.oss_config import OSSConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig


class ChunkStorageConfig(AbstractConfig):
    """Configuration for ChunkStorage - coordinates blob storage with metadata database for chunks"""
    type: Literal["chunk_storage"] = "chunk_storage"

    # Direct sub-configurations (no intermediate file_store layer)
    file_db_config: Annotated[
        Union[LocalDBConfig, MinIOConfig, OSSConfig],
        Field(discriminator="type"),
    ]
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> ChunkStorage:
        return ChunkStorage(self)
