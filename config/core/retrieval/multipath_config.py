from typing import Literal, List, Optional, Any, Union, Annotated, Dict
from pydantic import Field, ConfigDict
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from core.retrieval.multipath import MultiPathRetriever
from framework.config import AbstractConfig


class MultiPathRetrieverConfig(AbstractConfig):
    """Configuration for MultiPath Retriever"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["multipath"] = "multipath"

    retrievers: List[Annotated[
        Union["DenseRetrieverConfig", "TantivyBM25RetrieverConfig"],
        Field(discriminator="type")
    ]] = Field(
        default_factory=list,
        description="List of retriever config objects"
    )

    fusion_method: str = Field(default="rrf", description="Fusion method: 'rrf', 'weighted_sum', 'rank_fusion'")
    rrf_k: int = Field(default=60, description="RRF parameter k")
    weights: Optional[List[float]] = Field(default=None, description="Weights for weighted fusion")

    # Search parameters
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "with_score": True
        },
        description="Search parameters"
    )

    # 内部字段
    built_retrievers: Optional[List[Any]] = Field(default=None, exclude=True)
    fusion_instance: Optional[Any] = Field(default=None, exclude=True)
    

    def build(self):
        """Build the MultiPathRetriever instance"""
        built_retrievers = []
        for idx, retriever_config in enumerate(self.retrievers):
            if not hasattr(retriever_config, 'build'):
                raise TypeError(f"Retriever config at position {idx} does not provide a build() method")
            built_retrievers.append(retriever_config.build())

        self.built_retrievers = built_retrievers
        
        return MultiPathRetriever(self)