from typing import Literal, Annotated, Dict, Any, ClassVar
import threading
from pydantic import Field, ConfigDict
from framework.config import AbstractConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from core.retrieval.dense import DenseRetriever
from framework.shared_module_decorator import make_hashable

class DenseRetrieverConfig(AbstractConfig):
    
    type: Literal["dense"] = "dense"

    # Process-level cache toggle (via @shared_module on DenseRetriever).
    # Default True: identical configs share one in-process retriever instance, so index/embedding load once.
    shared_instance: bool = Field(
        default=True,
        description=(
            "Whether to reuse a process-level shared DenseRetriever instance for identical configs. "
            "Disable for strict isolation in tests/multi-tenant workers."
        ),
    )

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

    _process_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    _process_cache: ClassVar[dict[object, DenseRetriever]] = {}

    def build(self):
        if not bool(getattr(self, "shared_instance", True)):
            return DenseRetriever(self)
        key = make_hashable(self.model_dump())
        with self._process_cache_lock:
            cached = self._process_cache.get(key)
            if cached is not None:
                return cached
            inst = DenseRetriever(self)
            self._process_cache[key] = inst
            return inst

    @classmethod
    def clear_process_cache(cls) -> None:
        with cls._process_cache_lock:
            cls._process_cache.clear()
