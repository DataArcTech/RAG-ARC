from typing import Literal, Annotated, Dict, Any
from pydantic import Field, ConfigDict
from framework.config import AbstractConfig
from config.encapsulation.database.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.huggingface_config import HuggingFaceEmbedConfig
from core.retrieval.dense import DenseRetriever

class DenseRetrieverConfig(AbstractConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["dense"] = "dense"

    index_path: str = Field(description="Path to the index")
    index_name: str = Field(default="index", description="Name of the index")

    index_config: Annotated[FaissVectorDBConfig, Field(description="Index configuration")]
    # Runtime dependencies (injected by vector database)
    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric from vector database")

    # Search parameters
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "with_score": True,
            "score_threshold": None
        },
        description="Search parameters"
    )

    def build(self):
        return DenseRetriever(self)
