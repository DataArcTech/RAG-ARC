from typing import Any, List, Dict, ClassVar, Collection, Literal, Tuple, Annotated
import logging

from pydantic import ConfigDict, Field
from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.retrieval_helper import RetrievalHelper
from encapsulation.database.vector_db.faiss import FaissIndexConfig
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
from framework.shared_module_decorator import shared_module

logger = logging.getLogger(__name__)


class DenseRetrieverConfig(BaseRetrieverConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["dense"] = "dense"

    index_config: Annotated[FaissIndexConfig, Field(description="Index configuration")]
    embedding_config: Annotated[HuggingFaceEmbedConfig, Field(description="Embedding configuration")]
    # Runtime dependencies (injected by vector database)
    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric from vector database")


    # search_type: Literal["similarity", "similarity_score_threshold", "mmr"] = Field(default="similarity", description="Search type")
    def build(self) -> "DenseRetriever":
        """Build the DenseRetriever instance"""
        return DenseRetriever(self)


@shared_module
class DenseRetriever(BaseRetriever[DenseRetrieverConfig]):
    """
    基于向量数据库的密集检索器

    支持多种搜索类型：相似度搜索、阈值过滤、MMR多样性搜索
    """
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """Allowed search types"""
    
    config: DenseRetrieverConfig
    def __init__(self, config: DenseRetrieverConfig):
        super().__init__(config=config)
        self._ensure_index_initialized()
    
    def get_index(self):
        """Get the vector index"""
        if self._index is None:
            raise ValueError("Index not initialized")
        return self._index

    def _ensure_index_initialized(self) -> None:
        """确保索引已初始化（由IndexManager构建）"""
        # 检查索引是否存在
        if not hasattr(self, '_index') or self._index is None:
            raise RuntimeError("Index not initialized. Please use IndexManager to build the index first.")

        # 检查索引是否包含数据
        if hasattr(self._index, 'index_exists') and not self._index.index_exists():
            raise RuntimeError("Index exists but contains no data. Please use IndexManager to build the index first.")

        logger.debug(f"Index initialized successfully for {self.get_name()}")

    def _validate_search_config(self, search_type: str, search_kwargs: Dict[str, Any]) -> None:
        """验证搜索配置

        Args:
            search_type: 搜索类型
            search_kwargs: 搜索参数

        Raises:
            ValueError: 如果搜索类型不在允许的类型中
            ValueError: 如果使用similarity_score_threshold但没有指定有效的score_threshold
        """
        if search_type not in self.allowed_search_types:
            msg = (
                f"search_type '{search_type}' is not allowed. "
                f"Valid values are: {self.allowed_search_types}"
            )
            raise ValueError(msg)

        if search_type == "similarity_score_threshold":
            score_threshold = search_kwargs.get("score_threshold")
            if (score_threshold is None or
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "When using 'similarity_score_threshold' search type, "
                    "a valid score_threshold (float between 0 and 1) must be specified in search_kwargs"
                )
                raise ValueError(msg)

    def similarity_search(self, query: str, **kwargs: Any) -> List[Document]:
        """相似度搜索"""
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []

        # 嵌入查询
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, **kwargs)
    
    def similarity_search_by_vector(self, embedding: List[float], **kwargs: Any) -> List[Document]:
        """向量相似度搜索"""
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector_with_score(
        self, embedding: List[float], **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """向量搜索并返回分数"""
        index = self.get_index()
        if index is None:
            return []

        # 合并搜索参数
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        search_kwargs["metric"] = self.config.metric

        # 执行FAISS搜索
        return RetrievalHelper.vector_search_with_faiss(index, embedding, search_kwargs)
    
    def similarity_search_with_score(
        self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """搜索并返回分数"""
        index = self.get_index()
        if index is None:
            return []

        # 嵌入查询
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)

        # 执行向量搜索
        return self.similarity_search_by_vector_with_score(query_embedding, **kwargs)
    
    def max_marginal_relevance_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """最大边际相关性搜索（多样性）"""
        index = self.get_index()
        if index is None:
            return []

        # 嵌入查询
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)

        # 合并搜索参数
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        fetch_k = search_kwargs.get("fetch_k", 20)

        # 获取候选文档
        docs_and_scores = self.similarity_search_by_vector_with_score(
            query_embedding, k=fetch_k, **kwargs
        )

        if not docs_and_scores:
            return []

        # 准备MMR搜索参数
        search_kwargs["normalize_for_cosine"] = (
            (hasattr(index.config, 'normalize_L2') and index.config.normalize_L2) or
            (hasattr(index.config, 'metric') and index.config.metric == "cosine")
        )

        # 使用MMR选择文档
        return RetrievalHelper.mmr_search(
            query_embedding, docs_and_scores, embedding_model, search_kwargs
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        search_type: str = "similarity",
        **kwargs: Any
    ) -> List[Document]:
        """执行搜索并返回相关文档"""
        # 合并搜索参数
        search_kwargs = {**self.config.search_kwargs, **kwargs}

        # 获取参数
        k = search_kwargs.get("k", 5)
        with_score = search_kwargs.get("with_score", False)

        # 验证参数
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        # 验证搜索配置
        self._validate_search_config(search_type, search_kwargs)
        
        if not query.strip():
            logger.info("Empty query, returning empty results")
            return []
        
        try:
            if search_type == "similarity":
                docs = self.similarity_search(query, **search_kwargs)

            elif search_type == "similarity_score_threshold":
                # 使用带分数阈值的搜索
                docs_and_scores = self.similarity_search_with_score(query, **search_kwargs)
                docs = [doc for doc, _ in docs_and_scores]

            elif search_type == "mmr":
                docs = self.max_marginal_relevance_search(query, **search_kwargs)

            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # 为文档添加分数到元数据（如果需要且支持）
            if with_score and search_type != "similarity_score_threshold":
                try:
                    score_kwargs = {**search_kwargs, "k": len(docs)}
                    docs_with_scores = self.similarity_search_with_score(query, **score_kwargs)
                    docs = RetrievalHelper.add_scores_to_documents(docs, docs_with_scores)
                except Exception as e:
                    logger.debug(f"Unable to get scores: {e}")
            
            logger.debug(f"Retrieved {len(docs)} documents, search type: {search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"Error occurred while retrieving documents: {e}")
            raise

    def get_vectorstore_info(self) -> Dict[str, Any]:
        """获取向量数据库信息"""
        vectorstore = self.get_index()
        info = {
            "vectorstore_class": vectorstore.__class__.__name__,
            "metric": self.config.metric,
            "k": self.config.search_kwargs.get("k", 5),
            "with_score": self.config.search_kwargs.get("with_score", False),
            "search_kwargs": self.config.search_kwargs,
            "allowed_search_types": list(self.allowed_search_types),
        }

        try:
            embedding_model = self.get_embedding()
            info["embedding_class"] = embedding_model.__class__.__name__
        except ValueError:
            pass

        return info