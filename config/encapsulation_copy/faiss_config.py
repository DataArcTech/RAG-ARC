from framework.config import AbstractConfig
from typing import Literal, List, Annotated
from encapsulation.database.vector_db.faiss import FaissVectorDB
from config.llm.huggingface_config import HuggingFaceEmbedConfig
from pydantic import Field

class FaissIndexConfig(AbstractConfig):
    type: Literal["faiss"] = "faiss"

    index_path: str = None
    index_name: str = "index"

    metric: Literal["cosine", "l2", "ip"]
    index_type: Literal["flat", "ivf", "hnsw"]
    nlist: int = 100
    m: int = 8
    efConstruction: int = 40
    efSearch: int = 16
    train_size: int = 10000
    normalize_L2: bool = True

    embedding_config: Annotated[HuggingFaceEmbedConfig, Field(discriminator="type")]

    def build(self):
        return FaissVectorDB(self)