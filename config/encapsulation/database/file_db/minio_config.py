"""Configuration for MinIO S3-compatible blob storage"""

import os
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.database.file_db.minio import MinIODB
from typing import Literal


class MinIOConfig(AbstractConfig):
    """Configuration for MinIO S3-compatible blob storage

    Credentials loaded from environment variables.

    - Prefer official MinIO server env vars: MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
    - Fallback to legacy vars: MINIO_USERNAME / MINIO_PASSWORD
    """
    type: Literal["minio_blob_store"] = "minio_blob_store"

    endpoint: str = Field(default_factory=lambda: str(os.getenv("MINIO_ENDPOINT", "localhost:9000") or "localhost:9000").strip())
    username: str = Field(default_factory=lambda: os.getenv("MINIO_ROOT_USER") or os.getenv("MINIO_USERNAME", "root"))
    password: str = Field(default_factory=lambda: os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("MINIO_PASSWORD", "12345678"))
    bucket_name: str = Field(default_factory=lambda: str(os.getenv("MINIO_BUCKET", "test-bucket") or "test-bucket").strip())
    secure: bool = Field(default_factory=lambda: str(os.getenv("MINIO_SECURE", "false") or "false").strip().lower() in {"1", "true", "yes"})
    region: str = "us-east-1"

    def build(self) -> MinIODB:
        return MinIODB(self)
