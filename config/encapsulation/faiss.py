from framework.config import AbstractConfig
from config.encapsulation.llm.huggingface_embedding import HuggingFaceEmbeddingConfig
from encapsulation.database.vector_db.faiss import FaissVectorDB
from typing import Literal

class FaissConfig(AbstractConfig):
    """Configuration for FAISS Index"""
    type: Literal["faiss"] = "faiss"
    index_path: str = "./data/unified_faiss_index"
    embedding_config: HuggingFaceEmbeddingConfig
    metric: str = "cosine"
    index_type: str = "flat"
    normalize_L2: bool = True

    def build(self):
        return FaissVectorDB(self)