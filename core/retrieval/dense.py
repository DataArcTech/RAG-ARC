from typing import Any, Optional, List, Dict, ClassVar, Collection, Literal, TYPE_CHECKING, Tuple
import logging

from pydantic import ConfigDict, Field, model_validator
from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.retrieval_helper import RetrievalHelper

if TYPE_CHECKING:
    from encapsulation.database.vector_db.base import BaseVectorDB

logger = logging.getLogger(__name__)


class DenseRetrieverConfig(BaseRetrieverConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["dense"] = "dense"
    
    # Runtime dependencies (injected by vector database)
    vectorstore: Optional[Any] = Field(default=None, exclude=True, description="Vector database instance")
    metric: Optional[str] = Field(default=None, exclude=True, description="Distance metric from vector database")
    
    # Retrieval parameters (from vector database configuration)
    search_kwargs: Dict[str, Any] = Field(
        default_factory=dict, 
        exclude=True,
        description="""Additional search parameters. Supported parameters:
        - k (int): Number of documents to return (overrides config default)
        - score_threshold (float): Minimum relevance score threshold for similarity_score_threshold search
        - fetch_k (int): Number of documents to fetch for MMR (default: 20)
        - lambda_mult (float): Diversity parameter for MMR search (0-1, default: 0.5)
        - with_score (bool): Whether to include score in metadata (overrides config default)
        """
    )
    
    def build(self) -> "DenseRetriever":
        """Build the DenseRetriever instance"""
        if self.vectorstore is None:
            raise ValueError("DenseRetrieverConfig.vectorstore must not be None. Please provide a valid BaseVectorDB instance.")
        
        return DenseRetriever(config=self)


class DenseRetriever(BaseRetriever[DenseRetrieverConfig]):
    """
    DenseRetriever 是基于密集向量数据库的高性能文档检索器。
    
    此类实现基于向量相似度的文档检索，支持多种搜索类型和高级功能，
    如最大边际相关性(MMR)搜索以获得多样化的结果集。
    
    主要特性:
    - 多种搜索类型: similarity, similarity_score_threshold, mmr
    - 支持相关性分数阈值过滤
    - 最大边际相关性(MMR)搜索实现结果多样性
    - 异步操作支持
    - 灵活的搜索参数配置
    - 与多种向量数据库兼容
    
    配置参数 (来自 config):
        vectorstore (BaseVectorDB): 向量数据库实例
        metric (str): 距离度量类型 ('cosine', 'l2', 'ip')
        k (int): 默认返回文档数量
        with_score (bool): 是否默认包含相关性分数
        search_kwargs (dict): 额外搜索参数
        
    运行时实例变量:
        _relevance_score_fn: 相关性评分函数
        
    核心方法:
        - invoke: 同步检索的主要入口点
        - _get_relevant_documents: 执行搜索并返回结构化结果
        - _get_docs_with_embeddings_for_mmr: 获取MMR搜索所需的文档和嵌入
        
    性能考虑:
        - 相似性搜索适用于大多数场景
        - MMR搜索提供多样性但计算成本更高
        - 分数阈值过滤可能影响返回文档数量
        
    典型用法:
        >>> # 通过向量数据库创建检索器
        >>> retriever = vector_db.as_retriever()
        >>> results = retriever.invoke("查询文本")
        >>> results = retriever.invoke("查询文本", k=5, search_type="mmr")
    """
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """允许的搜索类型"""
    
    # Runtime instance variables
    _relevance_score_fn = None
    
    def _get_relevance_score_fn(self):
        """获取相关性评分函数（懒加载）"""
        if self._relevance_score_fn is None and self.config.metric:
            try:
                self._relevance_score_fn = RetrievalHelper.select_relevance_score_fn_by_metric(self.config.metric)
            except ValueError:
                logger.warning(f"不支持的度量类型 {self.config.metric}，使用余弦相似度评分函数")
                self._relevance_score_fn = RetrievalHelper.cosine_relevance_score_fn
        return self._relevance_score_fn
    
    def _validate_search_config(self, search_type: str, search_kwargs: Dict[str, Any]) -> None:
        """验证搜索配置
        
        Args:
            search_type: 搜索类型
            search_kwargs: 搜索参数
            
        Raises:
            ValueError: 如果搜索类型不在允许的类型中
            ValueError: 如果使用 similarity_score_threshold 但未指定有效的 score_threshold
        """
        if search_type not in self.allowed_search_types:
            msg = (
                f"search_type '{search_type}' 不被允许。"
                f"有效值为: {self.allowed_search_types}"
            )
            raise ValueError(msg)
        
        if search_type == "similarity_score_threshold":
            score_threshold = search_kwargs.get("score_threshold")
            if (score_threshold is None or 
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "使用 'similarity_score_threshold' 搜索类型时，"
                    "必须在 search_kwargs 中指定有效的 score_threshold (0~1 之间的浮点数)"
                )
                raise ValueError(msg)
    
    
    def _get_relevant_documents(
        self,
        query: str,
        k: Optional[int] = None,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        with_score: Optional[bool] = None,
        **kwargs: Any
    ) -> List[Document]:
        """执行搜索并返回结构化结果
        
        Args:
            query: 查询字符串
            k: 返回文档数量（默认使用配置值）
            search_type: 搜索类型
            score_threshold: 相关性分数阈值
            fetch_k: MMR搜索的候选文档数量
            lambda_mult: MMR多样性参数
            with_score: 是否在元数据中包含分数
            **kwargs: 额外搜索参数
            
        Returns:
            相关文档列表
        """
        # 使用配置默认值
        k = k if k is not None else self.config.k
        with_score = with_score if with_score is not None else self.config.with_score
        
        # 验证参数
        if k <= 0:
            raise ValueError(f"参数 'k' 必须大于 0，得到 {k}")
        
        # 合并搜索参数
        search_params = {**self.config.search_kwargs, **kwargs}
        search_params.update({
            'k': k,
            'score_threshold': score_threshold,
            'fetch_k': fetch_k,
            'lambda_mult': lambda_mult,
            'with_score': with_score
        })
        
        # 验证搜索配置
        self._validate_search_config(search_type, search_params)
        
        if not query.strip():
            logger.info("空查询，返回空结果")
            return []
        
        try:
            if search_type == "similarity":
                docs = self.config.vectorstore.similarity_search(query, k=k, **kwargs)
                
            elif search_type == "similarity_score_threshold":
                # 使用带相关性分数的搜索
                docs_and_similarities = self.config.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k, score_threshold=score_threshold, **kwargs
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif search_type == "mmr":
                # 检查向量数据库是否支持MMR
                if hasattr(self.config.vectorstore, 'max_marginal_relevance_search'):
                    docs = self.config.vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
                    )
                else:
                    # 向量数据库不支持MMR，退回到普通相似性搜索
                    logger.warning(
                        f"向量数据库 {self.config.vectorstore.__class__.__name__} 不支持MMR搜索，"
                        f"退回到普通相似性搜索 (k={k})"
                    )
                    docs = self.config.vectorstore.similarity_search(query, k=k, **kwargs)
                        
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")
            
            # 添加分数到元数据（如果需要且支持）
            if with_score and search_type != "similarity_score_threshold":
                # 对于非阈值搜索，尝试获取分数
                if hasattr(self.config.vectorstore, 'similarity_search_with_score'):
                    try:
                        docs_with_scores = self.config.vectorstore.similarity_search_with_score(
                            query, k=len(docs), **kwargs
                        )
                        score_dict = {doc.id: score for doc, score in docs_with_scores}
                        relevance_score_fn = self._get_relevance_score_fn()
                        
                        for doc in docs:
                            if doc.id in score_dict and relevance_score_fn:
                                relevance_score = relevance_score_fn(score_dict[doc.id])
                                doc.metadata = {**(doc.metadata or {}), "score": relevance_score}
                    except Exception as e:
                        logger.debug(f"无法获取分数: {e}")
            
            logger.debug(f"检索到 {len(docs)} 个文档，搜索类型: {search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"检索文档时发生错误: {e}")
            raise
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """异步获取与查询相关的文档
        
        默认实现使用线程池执行同步版本。子类可以重写以提供真正的异步实现。
        
        Args:
            query: 查询字符串
            **kwargs: 额外的搜索参数
            
        Returns:
            相关文档列表
        """
        try:
            import asyncio
            return await asyncio.to_thread(self._get_relevant_documents, query, **kwargs)
        except AttributeError:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_relevant_documents, query, **kwargs)
    

    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """获取向量数据库信息"""
        info = {
            "vectorstore_class": self.config.vectorstore.__class__.__name__,
            "metric": self.config.metric,
            "k": self.config.k,
            "with_score": self.config.with_score,
            "search_kwargs": self.config.search_kwargs,
            "allowed_search_types": list(self.allowed_search_types),
        }
        
        # 如果向量数据库有嵌入信息，添加到信息中
        if hasattr(self.config.vectorstore, 'embedding') and self.config.vectorstore.embedding:
            info["embedding_class"] = self.config.vectorstore.embedding.__class__.__name__
        
        return info
    
    def get_name(self) -> str:
        """获取检索器名称"""
        return f"{self.config.vectorstore.__class__.__name__}Retriever"