import os
from framework.config import AbstractConfig
from typing import Literal, List, Annotated, Union
from encapsulation.database.vector_db.faiss import FaissVectorDB
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from pydantic import Field

class FaissVectorDBConfig(AbstractConfig):
    type: Literal["faiss"] = "faiss"

    index_path: str = Field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "io://unified_faiss_index"))
    index_name: str = "index"

    owner_scoped_enabled: bool = Field(
        default=True,
        description=(
            "When enabled, store/load FAISS index artifacts under an owner-scoped subdirectory "
            "derived from `index_path` (see owner_scoped_dirname/global_owner_name). "
            "This avoids mixing multiple owners into a single on-disk index."
        ),
    )
    owner_scoped_dirname: str = Field(
        default="owners",
        description="Subdirectory name under `index_path` for owner-scoped FAISS indexes.",
    )
    owner_scoped_global_owner_name: str = Field(
        default="__GLOBAL__",
        description="Directory name used for admin/global scope when owner_id is None.",
    )
    owner_scoped_admin_max_owner_indexes: int = Field(
        default=50,
        ge=0,
        le=10000,
        description=(
            "When owner_scoped is enabled and an admin/global query is executed, "
            "cap the number of owner indexes scanned/fused to avoid expensive fan-out. "
            "0 disables admin global scan."
        ),
    )

    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric")
    index_type: Literal["flat", "ivf", "hnsw"] = Field(default="flat", description="Index type")
    nlist: int = 100
    m: int = 8
    efConstruction: int = 40
    efSearch: int = 16
    train_size: int = 10000
    min_train_size: int = Field(
        default_factory=lambda: int(os.getenv("FAISS_MIN_TRAIN_SIZE", "100")),
        description=(
            "Minimum number of vectors required to train IVF indexes. "
            "If you use index_type=ivf and have fewer vectors than this (or fewer than nlist), "
            "indexing will fail-fast with a clear error."
        ),
    )
    normalize_L2: bool = True
    two_stage_enabled: bool = Field(
        default_factory=lambda: str(os.getenv("FAISS_TWO_STAGE_ENABLED", "false")).strip().lower()
        in {"1", "true", "yes", "y", "on"},
        description=(
            "Enable two-stage retrieval for HNSW: ANN prefetch then exact rescoring on candidates. "
            "Applies only when index_type=hnsw."
        ),
    )
    two_stage_prefetch_k: int = Field(
        default_factory=lambda: int(os.getenv("FAISS_TWO_STAGE_PREFETCH_K", "200")),
        description=(
            "Candidate pool size for two-stage retrieval (HNSW prefetch). "
            "Must be >= k; if smaller, k is used."
        ),
    )

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
