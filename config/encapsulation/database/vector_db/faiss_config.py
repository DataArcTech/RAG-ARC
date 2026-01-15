import os
from framework.config import AbstractConfig
from typing import Literal, List, Annotated, Union
from encapsulation.database.vector_db.faiss import FaissVectorDB
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from pydantic import Field

class FaissVectorDBConfig(AbstractConfig):
    type: Literal["faiss"] = "faiss"

    index_path: str = Field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/unified_faiss_index"))
    index_name: str = "index"

    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric")
    index_type: Literal["flat", "ivf", "hnsw"] = Field(default="flat", description="Index type")
    nlist: int = 100
    m: int = 8
    efConstruction: int = 40
    efSearch: int = 16
    train_size: int = 10000
    normalize_L2: bool = True

    embedding_text_prefix_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Optional metadata keys to prefix to the embedded text (in-order), e.g. ['filename']. "
            "This is used to disambiguate similar products across companies by injecting file context into embeddings."
        ),
    )
    embedding_text_filename_root: str | None = Field(
        default=None,
        description=(
            "If set and `filename` is used as a prefix key, trim the filename/path to start from this token "
            "(e.g. 'RAG-ARC')."
        ),
    )
    embedding_text_separator: str = Field(
        default="\n",
        description="Separator used when prefixing metadata fields to the embedding text.",
    )

    embedding_config: Annotated[Union[QwenEmbeddingConfig, OpenAIEmbeddingConfig], Field(discriminator="type")]

    def build(self):
        return FaissVectorDB(self)
