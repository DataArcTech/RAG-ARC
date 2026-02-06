"""Configuration for Alibaba Cloud OSS (S3-compatible blob storage)"""

import os
from pydantic import Field
from framework.config import AbstractConfig
from encapsulation.database.file_db.minio import MinIODB
from typing import Literal


class OSSConfig(AbstractConfig):
    """Configuration for Alibaba Cloud OSS blob storage (S3-compatible API).

    Credentials and bucket info loaded from OSS_* environment variables.
    All chatKB files are stored under object_key_prefix (e.g. chatKB/files, chatKB/parsed_content).
    """
    type: Literal["oss_blob_store"] = "oss_blob_store"

    endpoint: str = Field(default_factory=lambda: os.getenv("OSS_ENDPOINT", "oss-cn-hongkong.aliyuncs.com"))
    username: str = Field(default_factory=lambda: os.getenv("OSS_ACCESS_KEY_ID", ""))
    password: str = Field(default_factory=lambda: os.getenv("OSS_ACCESS_KEY_SECRET", ""))
    bucket_name: str = Field(default_factory=lambda: os.getenv("OSS_BUCKET_NAME", "livingkb-chatkb-test-backet"))
    object_key_prefix: str = Field(default="", description="Prefix for all object keys (e.g. chatKB/files)")
    secure: bool = True
    region: str = Field(default_factory=lambda: os.getenv("OSS_REGION", "cn-hongkong"))

    def build(self) -> MinIODB:
        return MinIODB(self)
