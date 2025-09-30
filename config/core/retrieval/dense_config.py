from typing import Literal, Annotated, Dict, Any
from pydantic import Field, ConfigDict
from framework.config import AbstractConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from core.retrieval.dense import DenseRetriever

class DenseRetrieverConfig(AbstractConfig):
    
    type: Literal["dense"] = "dense"

    index_config: Annotated[FaissVectorDBConfig, Field(description="Index configuration")]
    # Runtime dependencies (injected by vector database)
    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric from vector database")

    search_kwargs: Dict[str, Any] = Field(
        default={
            "k": 5,
            "with_score": True,
            "score_threshold": None
        },
        description="Search parameters"
    )

    def build(self):
        return DenseRetriever(self)
