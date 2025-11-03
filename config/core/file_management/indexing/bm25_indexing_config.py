from typing import Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from core.file_management.indexing.bm25_indexing import BM25Indexer


class BM25IndexerConfig(AbstractConfig):
    type: Literal["bm25_indexer"] = "bm25_indexer"
    index_config: BM25BuilderConfig = Field(description="BM25 index configuration")

    # Batch processing configuration
    batch_size: int = Field(
        default=100,
        description="Number of chunks to accumulate before triggering a flush"
    )
    flush_interval: float = Field(
        default=5.0,
        description="Time interval (in seconds) to periodically flush pending chunks"
    )

    def build(self):
        return BM25Indexer(self)