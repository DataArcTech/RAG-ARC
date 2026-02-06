"""Configuration for MinIO S3-compatible blob storage"""

import os
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.database.file_db.minio import MinIODB
from typing import Literal


class MinIOConfig(AbstractConfig):
    """Configuration for MinIO S3-compatible blob storage

    Credentials loaded from environment variables (MINIO_USERNAME/MINIO_PASSWORD)
    or falls back to defaults.
    """
    type: Literal["minio_blob_store"] = "minio_blob_store"

    endpoint: str = "localhost:9000"
    username: str = Field(default_factory=lambda: os.getenv("MINIO_USERNAME", "ROOTNAME"))
    password: str = Field(default_factory=lambda: os.getenv("MINIO_PASSWORD", "CHANGEME123"))
    bucket_name: str = "test-bucket"
    object_key_prefix: str = ""
    secure: bool = False
    region: str = "us-east-1"

    def build(self) -> MinIODB:
        return MinIODB(self)
