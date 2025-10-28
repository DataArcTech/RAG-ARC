from typing import Literal, List, Optional, Any, Union, Annotated, Dict
from pydantic import Field, ConfigDict
import logging
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from core.retrieval.multipath import MultiPathRetriever
from framework.config import AbstractConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiPathRetrieverConfig(AbstractConfig):
    """Configuration for MultiPath Retriever"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["multipath"] = "multipath"

    retrievers: List[Annotated[
        Union[
            "DenseRetrieverConfig",
            "TantivyBM25RetrieverConfig",
            "PrunedHippoRAGRetrievalConfig",
            "PrunedHippoRAGNeo4jRetrievalConfig"
        ],
        Field(discriminator="type")
    ]] = Field(
        default_factory=list,
        description="List of retriever config objects (supports dense, tantivy_bm25, pruned_hipporag, pruned_hipporag_neo4j)"
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
            logger.info(f"Building retriever {idx} of type {retriever_config.type}...")
            if not hasattr(retriever_config, 'build'):
                raise TypeError(f"Retriever config at position {idx} does not provide a build() method")
            built_retrievers.append(retriever_config.build())
            logger.info(f"Retriever {idx} built successfully")

        self.built_retrievers = built_retrievers

        return MultiPathRetriever(self)


# Import after class definition to avoid circular imports
from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig
from config.core.retrieval.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jRetrievalConfig

# Rebuild the model to include the forward reference
MultiPathRetrieverConfig.model_rebuild()