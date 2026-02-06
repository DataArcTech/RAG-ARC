"""Configuration for ParsedContentStorage (Core Layer)"""

from typing import Annotated, Literal, Union

from framework.config import AbstractConfig
from pydantic import Field

from core.file_management.storage.parsed_content import ParsedContentStorage
from config.encapsulation.database.file_db.local_config import LocalDBConfig
from config.encapsulation.database.file_db.minio_config import MinIOConfig
from config.encapsulation.database.file_db.oss_config import OSSConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig


class ParsedContentStorageConfig(AbstractConfig):
    """Configuration for ParsedContentStorage - coordinates blob storage with metadata database for parsed content"""
    type: Literal["parsed_content_storage"] = "parsed_content_storage"

    # Direct sub-configurations (no intermediate file_store layer)
    file_db_config: Annotated[
        Union[LocalDBConfig, MinIOConfig, OSSConfig],
        Field(discriminator="type"),
    ]
    relational_db_config: PostgreSQLConfig  # Metadata database configuration

    def build(self) -> ParsedContentStorage:
        return ParsedContentStorage(self)
