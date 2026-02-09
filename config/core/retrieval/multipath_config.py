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

class _DisabledBuiltRetriever:
    """Placeholder retriever used when a MultiPath branch is disabled (e.g. weight=0).

    Why:
    - Avoid building heavy dependencies (e.g. Neo4j + embedding probes) when the backend is disabled by config.
    - Preserve retriever index alignment with `weights` and fusion logic.
    """

    def __init__(self, name: str) -> None:
        self._name = str(name or "disabled")

    def invoke(self, input: str, **kwargs: Any):  # noqa: ANN001
        return []

    def get_name(self) -> str:
        return f"disabled:{self._name}"


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
            # Allow disabling a branch via weight<=0 without building heavyweight retrievers.
            # This is config-driven (no hardcoded backend rules).
            try:
                if self.weights is not None and idx < len(self.weights):
                    if float(self.weights[idx]) <= 0:
                        built_retrievers.append(_DisabledBuiltRetriever(str(retriever_config.type)))
                        logger.info("Retriever %d disabled by weight<=0: type=%s", idx, retriever_config.type)
                        continue
            except Exception:  # noqa: BLE001
                pass
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
