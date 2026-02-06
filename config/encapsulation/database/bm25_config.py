import os
from typing import Literal, Optional, Dict, Any, List
from pydantic import Field, field_validator
from framework.config import AbstractConfig
from encapsulation.database.bm25_indexer import BM25IndexBuilder


class BM25BuilderConfig(AbstractConfig):
    """bm25 index builder configuration"""
    type: Literal["bm25_builder"] = "bm25_builder"

    index_path: str = Field(
        default_factory=lambda: os.getenv("BM25_INDEX_PATH", "./data/unified_bm25_index"),
        description="index storage path",
    )

    owner_scoped_enabled: bool = Field(
        default=True,
        description=(
            "When enabled, store/load BM25 index artifacts under an owner-scoped subdirectory "
            "derived from `index_path` (see owner_scoped_dirname/global_owner_name). "
            "This avoids mixing multiple owners into a single on-disk index."
        ),
    )
    owner_scoped_dirname: str = Field(
        default="owners",
        description="Subdirectory name under `index_path` for owner-scoped BM25 indexes.",
    )
    owner_scoped_global_owner_name: str = Field(
        default="__GLOBAL__",
        description="Directory name used for admin/global scope when owner_id is None.",
    )

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

    token_text_prefix_keys: List[str] = Field(
        default_factory=list,
        description=(
            "Optional metadata keys (from stored chunk metadata JSON) to prefix to the BM25 token text, "
            "e.g. ['filename']. This helps disambiguate similar products across companies by injecting file context."
        ),
    )
    token_text_filename_root: str | None = Field(
        default=None,
        description=(
            "If set and `filename` is used as a token text prefix key, trim the filename/path to start "
            "from this token (e.g. 'RAG-ARC')."
        ),
    )
    token_text_separator: str = Field(
        default="\n",
        description="Separator used when prefixing chunk metadata fields to the BM25 token text.",
    )

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
