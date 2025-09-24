"""Configuration for MinIO S3-compatible blob storage"""

from framework.config import AbstractConfig
from encapsulation.database.file_db.minio import MinIODB
from typing import Literal


class MinIOConfig(AbstractConfig):
    """Configuration for MinIO S3-compatible blob storage"""
    # Discriminator for config type identification
    type: Literal["minio_blob_store"] = "minio_blob_store"

    # MinIO server configuration
    endpoint: str = "localhost:9000"  # MinIO server endpoint
    username: str = "ROOTNAME"  # MinIO access key/username
    password: str = "CHANGEME123"  # MinIO secret key/password
    bucket_name: str = "test-bucket"  # S3 bucket name for storage

    # Connection configuration
    secure: bool = False  # Whether to use HTTPS (True for production)
    region: str = "us-east-1"  # AWS region (required for S3 compatibility)

    def build(self) -> MinIODB:
        return MinIODB(self)