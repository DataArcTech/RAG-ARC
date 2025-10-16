from framework.config import AbstractConfig
from typing import Literal, List, Annotated, Union
from encapsulation.database.vector_db.faiss import FaissVectorDB
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from pydantic import Field

class FaissVectorDBConfig(AbstractConfig):
    type: Literal["faiss"] = "faiss"

    index_path: str = "./data/faiss_index"
    index_name: str = "index"

    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric")
    index_type: Literal["flat", "ivf", "hnsw"] = Field(default="flat", description="Index type")
    nlist: int = 100
    m: int = 8
    efConstruction: int = 40
    efSearch: int = 16
    train_size: int = 10000
    normalize_L2: bool = True

    embedding_config: Annotated[Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig], Field(discriminator="type")]

    def build(self):
        return FaissVectorDB(self)