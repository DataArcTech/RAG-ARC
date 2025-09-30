from typing import Literal, Dict, Any
from pydantic import Field, ConfigDict
from framework.config import AbstractConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from core.retrieval.tantivy_bm25 import TantivyBM25Retriever

class TantivyBM25RetrieverConfig(AbstractConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["tantivy_bm25"] = "tantivy_bm25"
    
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
        return TantivyBM25Retriever(self)
