from typing import Literal, Dict, Any, ClassVar
import threading
from pydantic import Field, ConfigDict
from framework.config import AbstractConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from core.retrieval.tantivy_bm25 import TantivyBM25Retriever
from framework.shared_module_decorator import make_hashable

class TantivyBM25RetrieverConfig(AbstractConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["tantivy_bm25"] = "tantivy_bm25"

    # Process-level cache toggle (via @shared_module on TantivyBM25Retriever).
    # Default True: identical configs share one in-process retriever instance, so index load once.
    shared_instance: bool = Field(
        default=True,
        description=(
            "Whether to reuse a process-level shared TantivyBM25Retriever instance for identical configs. "
            "Disable for strict isolation in tests/multi-tenant workers."
        ),
    )
    
    # Index configuration
    index_config: BM25BuilderConfig = Field(description="BM25 index configuration")

    # Search parameters
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_phrase_query": False,
            "k": 5,
            "with_score": True
        },
        description="""Additional search parameters. Supported parameters:
        - use_phrase_query (bool): Whether to use phrase queries for better relevance (default: False)
        - k (int): Number of chunks to return (default: 5)
        - filters (dict): Dictionary of field names and their values to filter by
        - order_by_field (str): Field to sort by
        - order_desc (bool): Whether to sort in descending order (default: True)
        - with_score (bool): Whether to include score in metadata (default: False)
        """
    )

    def build(self):
        if not bool(getattr(self, "shared_instance", True)):
            return TantivyBM25Retriever(self)
        key = make_hashable(self.model_dump())
        with self._process_cache_lock:
            cached = self._process_cache.get(key)
            if cached is not None:
                return cached
            inst = TantivyBM25Retriever(self)
            self._process_cache[key] = inst
            return inst

    _process_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    _process_cache: ClassVar[dict[object, TantivyBM25Retriever]] = {}

    @classmethod
    def clear_process_cache(cls) -> None:
        with cls._process_cache_lock:
            cls._process_cache.clear()
