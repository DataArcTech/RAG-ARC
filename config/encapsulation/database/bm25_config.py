from typing import Literal, Optional, Dict, Any
from pydantic import Field, field_validator
from framework.config import AbstractConfig
from encapsulation.database.bm25_indexer import BM25IndexBuilder


class BM25BuilderConfig(AbstractConfig):
    """bm25 index builder configuration"""
    type: Literal["bm25_builder"] = "bm25_builder"

    index_path: str = Field(description="index storage path")
    bm25_k1: float = Field(default=1.2, description="bm25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="bm25 b parameter")

    preprocess_func_name: Optional[str] = Field(default=None, description="preprocessing function name")
    stopwords_file: Optional[str] = Field(default=None, description="stopwords file path")
    writer_heap_size: Optional[int] = Field(default=None, description="writer heap size")
    batch_size: int = Field(default=50, description="batch size")
    tokenize_batch_size: int = Field(default=200, description="tokenize batch size")
    max_workers: Optional[int] = Field(default=None, description="maximum worker processes")
    progress_interval: int = Field(default=500, description="progress report interval")
    enable_gc: bool = Field(default=True, description="enable garbage collection")
    queue_maxsize: int = Field(default=1000, description="queue max size")

    search_kwargs: Dict[str, Any] = Field(
        default={
            "use_phrase_query": False,
            "k": 5,
            "with_score": True
        },
        description="search parameters"
    )
    k: int = Field(default=5, description="default number of results")
    with_score: bool = Field(default=True, description="return score")

    
    @field_validator("bm25_k1")
    @classmethod
    def validate_bm25_k1(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bm25_k1 must be greater than 0, but got {v}")
        return v
    
    @field_validator("bm25_b")
    @classmethod
    def validate_bm25_b(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError(f"bm25_b must be between 0 and 1, but got {v}")
        return v

    def build(self):
        return BM25IndexBuilder(self)
