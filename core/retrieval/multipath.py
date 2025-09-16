import logging
from typing import Any, List, Optional, Literal, Union, Annotated
from pydantic import ConfigDict, Field, model_validator

from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.Fusion import FusionMethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.retrieval.dense import DenseRetrieverConfig
    from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig

logger = logging.getLogger(__name__)


class MultiPathRetrieverConfig(BaseRetrieverConfig):
    """Configuration for MultiPath Retriever"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["multipath"] = "multipath"

    # 重写index_config为可选，因为MultiPath使用retrievers字段
    index_config: Optional[Any] = None
    # MultiPath检索器不需要embedding配置
    embedding_config: Optional[Any] = None

    retrievers: List[Annotated[Union["DenseRetrieverConfig", "TantivyBM25RetrieverConfig"], Field(discriminator="type")]] = Field(
        default_factory=list,
        description="List of retriever config objects"
    )

    # 融合方法配置
    fusion_method: str = Field(default="rrf", description="Fusion method: 'rrf', 'weighted_sum', 'rank_fusion'")
    rrf_k: int = Field(default=60, description="RRF parameter k")
    weights: Optional[List[float]] = Field(default=None, description="Weights for weighted fusion")

    # 内部字段
    built_retrievers: Optional[List[BaseRetriever]] = Field(default=None, exclude=True)
    fusion_instance: Optional[FusionMethod] = Field(default=None, exclude=True)
    

    @model_validator(mode='after')
    def validate_retrievers(self) -> 'MultiPathRetrieverConfig':
        """Validate retrievers presence"""
        if not self.retrievers:
            raise ValueError("At least one retriever config is required")
        return self

    def build(self) -> "MultiPathRetriever":
        """Build the MultiPathRetriever instance"""
        built_retrievers = []
        for idx, retriever_config in enumerate(self.retrievers):
            if not hasattr(retriever_config, 'build'):
                raise TypeError(f"Retriever config at position {idx} does not provide a build() method")
            built_retrievers.append(retriever_config.build())

        self.built_retrievers = built_retrievers
        return MultiPathRetriever(config=self)


class MultiPathRetriever(BaseRetriever[MultiPathRetrieverConfig]):
    """
    MultiPath检索器，使用多个检索器并融合结果。

    支持的融合方法：
    - rrf: Reciprocal Rank Fusion
    - weighted_sum: 加权求和
    - rank_fusion: 基于排名的融合
    """
    
    def __init__(self, config: MultiPathRetrieverConfig):
        """Initialize MultiPathRetriever"""
        self.config = config
        self._index = None
        self._embedding = None
        self._init_fusion_method()

    def _init_fusion_method(self):
        """初始化融合方法"""
        if self.config.fusion_method == "rrf":
            from core.utils.Fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k)
        elif self.config.fusion_method == "weighted_sum":
            from core.utils.Fusion import WeightedSumFusion
            weights = self.config.weights or [1.0] * len(self.config.retrievers)
            self.config.fusion_instance = WeightedSumFusion(weights=weights)
        elif self.config.fusion_method == "rank_fusion":
            from core.utils.Fusion import RankFusion
            self.config.fusion_instance = RankFusion()
        else:
            from core.utils.Fusion import RRFusion
            self.config.fusion_instance = RRFusion(k=self.config.rrf_k)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """检索相关文档并融合结果"""
        k = kwargs.get('k', self.config.search_kwargs.get('k', 5))

        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        if not query.strip():
            return []

        all_results = []
        for retriever in self.config.built_retrievers or []:
            try:
                documents = retriever.invoke(query, **kwargs)
                # 确保每个文档都有分数
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    if 'score' not in doc.metadata:
                        doc.metadata['score'] = 1.0
                all_results.append(documents)
                logger.debug(f"Retriever {type(retriever).__name__} returned {len(documents)} results")
            except Exception as e:
                logger.error(f"Retriever {type(retriever).__name__} failed: {e}")
                all_results.append([])

        if not all_results or all(len(results) == 0 for results in all_results):
            return []

        return self.config.fusion_instance.fuse(all_results, k)

    def add_retriever(self, retriever: Any) -> None:
        """添加检索器"""
        if not hasattr(retriever, 'invoke'):
            raise ValueError(f"Retriever must implement invoke method")
        if self.config.built_retrievers is None:
            self.config.built_retrievers = []
        self.config.built_retrievers.append(retriever)

    def remove_retriever(self, name: str) -> bool:
        """移除指定名称的检索器"""
        if not self.config.built_retrievers:
            return False
        for i, retriever in enumerate(self.config.built_retrievers):
            if type(retriever).__name__ == name:
                self.config.built_retrievers.pop(i)
                return True
        return False

    def set_fusion_method(self, fusion_method: str) -> None:
        """设置融合方法"""
        self.config.fusion_method = fusion_method
        self._init_fusion_method()

    def get_multipath_info(self) -> dict:
        """获取多路径检索器信息"""
        retrievers = self.config.built_retrievers or []
        return {
            "retriever_count": len(retrievers),
            "retriever_types": [type(r).__name__ for r in retrievers],
            "fusion_method": self.config.fusion_method,
            "search_kwargs": self.config.search_kwargs
        }

    # CRUD方法委托给所有子检索器
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到所有子检索器"""
        all_ids = []
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'add_documents'):
                all_ids.extend(retriever.add_documents(documents))
        return all_ids

    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """从所有子检索器删除文档"""
        results = []
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'delete_documents'):
                results.append(retriever.delete_documents(ids, **kwargs))
        return all(r is not False for r in results) if results else None

    def update_documents(self, documents: List[Document]) -> None:
        """更新所有子检索器的文档"""
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'update_documents'):
                retriever.update_documents(documents)

    def build_index(self, documents: List[Document]) -> None:
        """在所有子检索器中构建索引"""
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'build_index'):
                retriever.build_index(documents)

    def save_index(self, index_path: str, index_name: str = "index") -> None:
        """保存所有子检索器的索引"""
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'save_index'):
                retriever.save_index(index_path, index_name)

    def load_index(self, index_path: Optional[str] = None) -> None:
        """加载所有子检索器的索引"""
        for retriever in self.config.built_retrievers or []:
            if hasattr(retriever, 'load_index'):
                retriever.load_index(index_path)


# 解决Pydantic模型定义问题
try:
    from core.retrieval.dense import DenseRetrieverConfig
    from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig
    MultiPathRetrieverConfig.model_rebuild()
except ImportError:
    # 如果导入失败，稍后会在实际使用时重建
    pass