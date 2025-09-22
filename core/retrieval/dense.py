from typing import Any, List, Dict, ClassVar, Collection
import logging

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.data_model import Document
from core.utils.retrieval_helper import RetrievalHelper
from framework.shared_module_decorator import shared_module

logger = logging.getLogger(__name__)




@shared_module
class DenseRetriever(BaseRetriever):
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
    
    def __init__(self, config):
        self.config = config
        # Pass embedding config to FAISS index config
        if hasattr(config, 'embedding_config') and config.embedding_config is not None:
            config.index_config.embedding_config = config.embedding_config
        self._index = self.config.index_config.build()
        self._embedding = None
        self._load_existing_index()
        self._ensure_index_initialized()

    def _load_existing_index(self) -> None:
        """尝试加载已存在的索引"""
        try:
            if hasattr(self._index, 'load_index'):
                # Check if the index has an index_path in its config
                if hasattr(self._index.config, 'index_path') and self._index.config.index_path:
                    self._index.load_index(self._index.config.index_path)
                else:
                    self._index.load_index()
                logger.info(f"Successfully loaded existing index for {self.get_name()}")
        except Exception as e:
            message = f"Index not found for retriever {self.get_name()}: {e}"
            logger.warning(f"{message}. Index will be empty until documents are added.")
            # Don't raise an error, just continue with an empty index
    
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

    def similarity_search(self, query: str, include_score: bool = False, **kwargs: Any) -> List[Document]:
        """相似度搜索

        Args:
            query: 查询字符串
            include_score: 是否在Document.metadata["score"]中包含相似度分数
            **kwargs: 其他搜索参数

        Returns:
            文档列表，如果include_score=True，则分数存储在metadata["score"]中
        """
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []

        # 嵌入查询
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, include_score=include_score, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], include_score: bool = False, **kwargs: Any) -> List[Document]:
        """向量相似度搜索

        Args:
            embedding: 查询嵌入向量
            include_score: 是否在Document.metadata["score"]中包含相似度分数
            **kwargs: 其他搜索参数

        Returns:
            文档列表，如果include_score=True，则分数存储在metadata["score"]中
        """
        index = self.get_index()
        if index is None:
            return []

        # 合并搜索参数
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        search_kwargs["metric"] = self.config.metric

        # 执行FAISS搜索
        docs_and_scores = RetrievalHelper.vector_search_with_faiss(index, embedding, search_kwargs)

        if include_score:
            # 将分数添加到文档的metadata中
            documents = []
            for doc, score in docs_and_scores:
                # 创建文档副本以避免修改原始文档
                doc_copy = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata={**doc.metadata, "score": score}
                )
                documents.append(doc_copy)
            return documents
        else:
            return [doc for doc, _ in docs_and_scores]
    
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

        # 获取候选文档（使用内部方法获取分数）
        docs_and_scores = RetrievalHelper.vector_search_with_faiss(
            index, query_embedding, {**search_kwargs, "k": fetch_k, "metric": self.config.metric}
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
                docs = self.similarity_search(query, include_score=with_score, **search_kwargs)

            elif search_type == "similarity_score_threshold":
                # 使用带分数阈值的搜索，总是包含分数用于阈值过滤
                docs = self.similarity_search(query, include_score=True, **search_kwargs)

                # 应用分数阈值过滤
                score_threshold = search_kwargs.get("score_threshold")
                if score_threshold is not None:
                    filtered_docs = []
                    for doc in docs:
                        score = doc.metadata.get("score", 0.0)
                        if score >= score_threshold:
                            filtered_docs.append(doc)
                    docs = filtered_docs

                    if len(docs) == 0:
                        logger.warning(f"使用相关性分数阈值 {score_threshold} 没有检索到相关文档")

                # 如果不需要返回分数，则移除分数信息
                if not with_score:
                    for doc in docs:
                        if "score" in doc.metadata:
                            doc.metadata = {k: v for k, v in doc.metadata.items() if k != "score"}

            elif search_type == "mmr":
                docs = self.max_marginal_relevance_search(query, **search_kwargs)

                # 如果需要分数，重新获取分数信息
                if with_score:
                    docs = self.similarity_search(query, include_score=True, k=len(docs), **search_kwargs)

            else:
                raise ValueError(f"Unsupported search type: {search_type}")

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