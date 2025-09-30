from typing import Literal
from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from core.file_management.indexing.faiss_indexing import FaissIndexer


class FaissIndexerConfig(AbstractConfig):
    type: Literal["faiss_indexer"] = "faiss_indexer"
    index_config: FaissVectorDBConfig = Field(description="FAISS index configuration")

    def build(self):
        return FaissIndexer(self)