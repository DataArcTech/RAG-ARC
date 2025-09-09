import asyncio
import logging
from typing import Any, List, Optional, Literal, Annotated
from pydantic import ConfigDict, Field, field_validator, model_validator
import warnings

from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.retrieval.dense import DenseRetrieverConfig
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig
from core.utils.data_model import Document
from core.utils.Fusion import FusionMethod, RRFusion

logger = logging.getLogger(__name__)


class MultiPathRetrieverConfig(BaseRetrieverConfig):
    """Configuration for MultiPath Retriever"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["multipath"] = "multipath"
    
    # Retriever configurations (sub-modules)
    retrievers: List[Annotated[DenseRetrieverConfig | TantivyBM25RetrieverConfig, Field(discriminator="type")]] = Field(
        default_factory=list,
        description="List of retriever configurations, each will be built into a retriever instance"
    )
    
    # Runtime built retrievers (populated after build)
    built_retrievers: Optional[List[BaseRetriever]] = Field(
        default=None,
        exclude=True,
        description="Built retriever instances (internal use only)"
    )
    
    fusion_method: FusionMethod = Field(
        default_factory=RRFusion,
        exclude=True,
        description="Fusion method for merging results from multiple retrievers"
    )
    
    # Retrieval parameters
    top_k_per_retriever: int = Field(
        default=50,
        gt=0,
        description="Number of results returned by each retriever"
    )
    
    @field_validator("retrievers")
    @classmethod
    def validate_retrievers(cls, v: List[Any]) -> List[Any]:
        """Validate that all retriever configs are valid"""
        if len(v) == 0:
            raise ValueError("At least one retriever configuration is required")
        return v
    
    @model_validator(mode='after')
    def validate_k_and_top_k_per_retriever(self) -> 'MultiPathRetrieverConfig':
        """Validate that k is less than or equal to top_k_per_retriever"""
        if self.k > self.top_k_per_retriever:
            raise ValueError(
                f"k ({self.k}) must be less than or equal to top_k_per_retriever ({self.top_k_per_retriever}). "
                f"Each retriever can only return at most {self.top_k_per_retriever} results, "
                f"so the final result cannot exceed this limit."
            )
        return self
    
    def build(self) -> "MultiPathRetriever":
        """Build the MultiPathRetriever instance"""
        if not self.retrievers:
            raise ValueError("MultiPathRetrieverConfig.retrievers cannot be empty. Please provide at least one retriever.")
        
        # Build retriever instances from configurations
        built_retrievers = []
        for retriever_config in self.retrievers:
            retriever_instance = retriever_config.build()
            built_retrievers.append(retriever_instance)
        
        # Store built retrievers in the internal field
        self.built_retrievers = built_retrievers
        
        return MultiPathRetriever(config=self)
    
    def get_built_retrievers(self) -> List[BaseRetriever]:
        """Get the built retriever instances"""
        return self.built_retrievers or []


class MultiPathRetriever(BaseRetriever[MultiPathRetrieverConfig]):
    """
    MultiPathRetriever 是一个多路径文档检索器，可以同时使用多个检索器进行文档检索，并通过指定的融合方法合并和排序多个检索器的结果。

    此类实现多路径检索功能，支持组合不同检索算法（如BM25、向量检索等）的结果，以提高检索准确性和鲁棒性。
    
    主要特性:
    - 支持多个检索器并行运行
    - 支持可配置的融合方法（默认为互惠排名融合）
    - 兼容同步和异步调用
    - 提供动态添加和移除检索器
    - 通过Pydantic验证参数确保配置安全性

    配置参数 (来自 config):
        retrievers (List[Any]): 检索器列表，每个检索器需要实现invoke方法
        fusion_method (FusionMethod): 用于合并多个检索器结果的融合方法
        top_k_per_retriever (int): 每个检索器返回的结果数量
        k (int): 默认返回文档数量
        with_score (bool): 是否默认包含相关性分数
        search_kwargs (dict): 额外搜索参数

    核心方法:
        - invoke: 同步检索的主要入口点
        - _get_relevant_documents: 核心检索实现
        - add_retriever/remove_retriever: 动态管理检索器
        - set_fusion_method: 设置融合方法

    性能考虑:
        - 每个检索器独立运行，整体性能取决于最慢的检索器
        - 融合过程增加额外的计算开销
        - 对于高实时性要求的场景，建议优化单个检索器的性能

    典型用法:
        >>> config = MultiPathRetrieverConfig(
        ...     retrievers=[bm25_config, vector_config],
        ...     fusion_method=RRFusion(),
        ...     top_k_per_retriever=50
        ... )
        >>> multi_retriever = config.build()
        >>> results = multi_retriever.invoke("查询语句")
    """
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        获取与查询相关的文档
        
        此方法将调用所有配置的检索器，获取每个检索器的检索结果，
        然后使用指定的融合方法合并和排序所有结果。
        
        Args:
            query: 查询字符串
            **kwargs: 其他参数，包括k等
            
        Returns:
            融合后的相关文档列表，按相关性排序
            
        Note:
            - 每个检索器返回Document对象列表
            - 融合后的结果返回排序后的Document对象，分数存储在metadata['score']中
        """
        # 使用配置默认值
        top_k = kwargs.get('k', self.config.k)
        top_k_per_retriever = kwargs.get('top_k_per_retriever', self.config.top_k_per_retriever)
        
        # 验证参数
        if top_k <= 0:
            raise ValueError(f"参数 'k' 必须大于 0，得到 {top_k}")
        
        if top_k > top_k_per_retriever:
            raise ValueError(
                f"k ({top_k}) 必须小于等于 top_k_per_retriever ({top_k_per_retriever})。"
                f"每个检索器最多只能返回 {top_k_per_retriever} 个结果，"
                f"因此最终结果不能超过此限制。"
            )
        
        if not query.strip():
            logger.info("空查询，返回空结果")
            return []
        
        all_results = []
        for retriever in self.config.built_retrievers:
            try:
                # 为每个检索器调用时传递正确的参数
                retriever_kwargs = {**kwargs, 'k': top_k_per_retriever}
                documents = retriever.invoke(query, **retriever_kwargs)
                
                # 确保每个文档都有分数在metadata中
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    # 如果没有分数，使用默认分数1.0
                    if 'score' not in doc.metadata:
                        doc.metadata['score'] = 1.0
                
                all_results.append(documents)
                logger.debug(f"检索器 {type(retriever).__name__} 返回 {len(documents)} 个结果")
                
            except Exception as e:
                logger.error(f"检索器 {type(retriever).__name__} 执行失败: {e}")
                warnings.warn(f"检索器 {type(retriever).__name__} 执行失败: {e}", RuntimeWarning)
                all_results.append([])
        
        if not all_results or all(len(results) == 0 for results in all_results):
            logger.warning("所有检索器都没有返回结果")
            return []
        
        fused_documents = self.config.fusion_method.fuse(all_results, top_k)
        logger.info(f"使用 {type(self.config.fusion_method).__name__} 融合了 {len(fused_documents)} 个结果")
        
        return fused_documents

    def add_retriever(self, retriever: Any) -> None:
        """
        向多路径检索器添加新的检索器
        
        Args:
            retriever: 要添加的检索器实例
        """
        if not hasattr(retriever, 'invoke'):
            raise ValueError(f"检索器 {type(retriever).__name__} 必须实现 invoke 方法")
        
        if self.config.built_retrievers is None:
            self.config.built_retrievers = []
        self.config.built_retrievers.append(retriever)
        logger.info(f"已添加检索器 {type(retriever).__name__}")
    
    def remove_retriever(self, name: str) -> bool:
        """
        移除指定的检索器
        
        Args:
            name: 要移除的检索器的类名
            
        Returns:
            是否移除成功
            
        Note:
            此方法通过比较检索器的类名来识别要移除的检索器
        """
        if self.config.built_retrievers is None:
            return False
            
        for i, retriever in enumerate(self.config.built_retrievers):
            if hasattr(retriever, '__class__') and retriever.__class__.__name__ == name:
                removed_retriever = self.config.built_retrievers.pop(i)
                logger.info(f"已移除检索器 {type(removed_retriever).__name__}")
                return True
        logger.warning(f"未找到检索器 {name}")
        return False
    
    def set_fusion_method(self, fusion_method: FusionMethod) -> None:
        """
        设置融合方法
        
        Args:
            fusion_method: 新的融合方法实例
        """
        self.config.fusion_method = fusion_method
        logger.info(f"已设置融合方法为 {type(fusion_method).__name__}")

    def get_multipath_info(self) -> dict:
        """获取多路径检索器信息"""
        retrievers = self.config.built_retrievers or []
        return {
            "retriever_count": len(retrievers),
            "retriever_types": [type(retriever).__name__ for retriever in retrievers],
            "fusion_method": type(self.config.fusion_method).__name__,
            "top_k_per_retriever": self.config.top_k_per_retriever,
            "k": self.config.k,
            "with_score": self.config.with_score,
            "search_kwargs": self.config.search_kwargs
        }
    
    def get_name(self) -> str:
        """获取检索器名称"""
        retrievers = self.config.built_retrievers or []
        retriever_names = [type(r).__name__ for r in retrievers]
        return f"MultiPath[{','.join(retriever_names)}]"