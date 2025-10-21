import logging
from typing import Any, List, TYPE_CHECKING

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.retrieval.multipath_config import MultiPathRetrieverConfig

logger = logging.getLogger(__name__)


class MultiPathRetriever(BaseRetriever):
    """
    MultiPath retriever, uses multiple retrievers and fuses results.

    Supported fusion methods:
    - rrf: Reciprocal Rank Fusion
    - weighted_sum: weighted sum
    - rank_fusion: rank-based fusion
    """
    
    def __init__(self, config: "MultiPathRetrieverConfig"):
        """Initialize MultiPathRetriever"""
        self.config = config
        self._index = None
        self._embedding = None
        self._init_fusion_method()

    def _init_fusion_method(self):
        """Initialize fusion method"""
        if self.config.fusion_method == "rrf":
            from core.utils.fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k)
        elif self.config.fusion_method == "weighted_sum":
            from core.utils.fusion import WeightedSumFusion
            weights = self.config.weights or [1.0] * len(self.config.retrievers)
            self.config.fusion_instance = WeightedSumFusion(weights=weights)
        elif self.config.fusion_method == "rank_fusion":
            from core.utils.fusion import RankFusion
            self.config.fusion_instance = RankFusion()
        else:
            from core.utils.fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k)

    def _get_relevant_chunks(self, query: str, **kwargs: Any) -> List[Chunk]:
        """Search relevant chunks and fuse results"""
        k = kwargs.get('k', self.config.search_kwargs.get('k', 5))

        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        if not query.strip():
            return []

        all_results = []
        subgraph_info = None  # Store subgraph info from graph retriever

        for retriever in self.config.built_retrievers or []:
            try:
                chunks = retriever.invoke(query, **kwargs)

                # Check if this is a graph retriever with subgraph info
                if chunks and hasattr(chunks[0], 'metadata') and chunks[0].metadata:
                    if '_subgraph_info' in chunks[0].metadata:
                        subgraph_info = chunks[0].metadata.pop('_subgraph_info')
                        logger.info(f"Captured subgraph info from {type(retriever).__name__}")

                # Ensure each chunk has a score
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    if 'score' not in chunk.metadata:
                        chunk.metadata['score'] = 1.0
                all_results.append(chunks)
                logger.debug(f"Retriever {type(retriever).__name__} returned {len(chunks)} results")
            except Exception as e:
                logger.error(f"Retriever {type(retriever).__name__} failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                all_results.append([])

        if not all_results or all(len(results) == 0 for results in all_results):
            return []

        fused_chunks = self.config.fusion_instance.fuse(all_results, k)

        # Attach subgraph info to first chunk if available
        if subgraph_info and fused_chunks:
            if fused_chunks[0].metadata is None:
                fused_chunks[0].metadata = {}
            fused_chunks[0].metadata['_subgraph_info'] = subgraph_info
            logger.info("Attached subgraph info to fused results")

        return fused_chunks


    @property
    def retrievers(self):
        """Expose built retrievers for external access"""
        return self.config.built_retrievers or []

    def get_multipath_info(self) -> dict:
        retrievers = self.config.built_retrievers or []
        return {
            "retriever_count": len(retrievers),
            "retriever_types": [type(r).__name__ for r in retrievers],
            "fusion_method": self.config.fusion_method,
            "search_kwargs": self.config.search_kwargs
        }
