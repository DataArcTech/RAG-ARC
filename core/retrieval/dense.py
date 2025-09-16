from typing import Any, Optional, List, Dict, ClassVar, Collection, Literal, Tuple, Annotated
import logging
import numpy as np

from pydantic import ConfigDict, Field
from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.retrieval_helper import RetrievalHelper
from encapsulation.database.vector_db.faiss import FaissIndexConfig
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
from framework.shared_module_decorator import shared_module
import os  # New

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
    
    # Runtime instance variables
    _relevance_score_fn = None
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
        """在初始化时，自动加载本地已构建索引（如果存在且开启）"""
        # Get index without embedding check during initialization
        if not hasattr(self, '_index') or self._index is None:
            return
        vs = self._index
        try:
            base_path = getattr(getattr(vs, "config", None), "index_path", None)
            if base_path and isinstance(base_path, str) and os.path.isdir(base_path):
                need_load = True
                if hasattr(vs, "index"):
                    idx = getattr(vs, "index")
                    if idx is not None and hasattr(idx, "ntotal"):
                        need_load = (idx.ntotal == 0)
                    elif idx is not None:
                        need_load = False
                if need_load and hasattr(vs, "load_index"):
                    vs.load_index(base_path)
        except Exception as e:
            logger.debug(f"Auto-loading local index failed: {e}")
    
    # 继承父类的 add_documents 方法，无需重写

    def _get_relevance_score_fn(self):
        """Get relevance scoring function (lazy loading)"""
        if self._relevance_score_fn is None and self.config.metric:
            try:
                self._relevance_score_fn = RetrievalHelper.select_relevance_score_fn_by_metric(self.config.metric)
            except ValueError:
                logger.warning(f"Unsupported metric type {self.config.metric}, using cosine similarity scoring function")
                self._relevance_score_fn = RetrievalHelper.cosine_relevance_score_fn
        return self._relevance_score_fn
    
    def _validate_search_config(self, search_type: str, search_kwargs: Dict[str, Any]) -> None:
        """Validate search configuration
        
        Args:
            search_type: Search type
            search_kwargs: Search parameters
            
        Raises:
            ValueError: If search type is not in allowed types
            ValueError: If using similarity_score_threshold but no valid score_threshold is specified
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

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """相似度搜索"""
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []
        
        # Embed query
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Document]:
        """向量相似度搜索"""
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """向量搜索并返回分数"""
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []
        
        # Prepare query vector
        query_vector = np.array([embedding]).astype(np.float32)
        
        # Normalize vectors if needed
        if hasattr(index, '_normalize_vectors'):
            query_vector = index._normalize_vectors(query_vector)
        elif (hasattr(index.config, 'normalize_L2') and index.config.normalize_L2) or \
             (hasattr(index.config, 'metric') and index.config.metric == "cosine"):
            import faiss
            faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, index.index.ntotal)
        distances, indices = index.index.search(query_vector, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            doc_id = index.index_to_docstore_id[idx]
            doc = index.docstore[doc_id]
            results.append((doc, float(distance)))
        
        return results
    
    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, score_threshold: Optional[float] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """搜索并返回相关性分数 [0, 1]"""
        # Get relevance score function based on metric
        relevance_score_fn = self._get_relevance_score_fn()
        
        # Get documents with distance scores
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        
        # Convert distances to relevance scores
        docs_and_similarities = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]
        
        # Apply score threshold if specified
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
        
        return docs_and_similarities
    
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """搜索并返回距离分数"""
        index = self.get_index()
        if index is None:
            return []
        
        # Embed query
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)
        return self.similarity_search_by_vector_with_score(query_embedding, k, **kwargs)
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """最大边际相关性搜索（多样性）"""
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []
        
        # Embed query
        embedding_model = self.get_embedding()
        query_embedding = embedding_model.embed_query(query)
        
        # Get candidate documents
        docs_and_scores = self.similarity_search_by_vector_with_score(
            query_embedding, fetch_k, **kwargs
        )
        
        if not docs_and_scores:
            return []
        
        # Get document embeddings
        candidate_embeddings = []
        for doc, _ in docs_and_scores:
            # Re-embed document content (in practice, you might want to cache these)
            doc_embedding = embedding_model.embed_query(doc.content)
            candidate_embeddings.append(doc_embedding)
        
        # Normalize embeddings for cosine similarity
        query_emb_norm = np.array(query_embedding)
        candidate_embs_norm = np.array(candidate_embeddings)
        
        if (hasattr(index.config, 'normalize_L2') and index.config.normalize_L2) or \
           (hasattr(index.config, 'metric') and index.config.metric == "cosine"):
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )
        
        # Use MMR selection from retrieval helper
        selected_docs = RetrievalHelper.mmr_select_documents(
            docs_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )
        
        return selected_docs
    
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
        """执行搜索并返回相关文档"""
        # Use configuration defaults
        k = k if k is not None else self.config.search_kwargs.get("k", 5)
        with_score = with_score if with_score is not None else self.config.search_kwargs.get("with_score", False)
        
        # Validate parameters
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")
        
        # Merge search parameters
        search_params = {**self.config.search_kwargs, **kwargs}
        search_params.update({
            'k': k,
            'score_threshold': score_threshold,
            'fetch_k': fetch_k,
            'lambda_mult': lambda_mult,
            'with_score': with_score
        })
        
        # Validate search configuration
        self._validate_search_config(search_type, search_params)
        
        if not query.strip():
            logger.info("Empty query, returning empty results")
            return []
        
        try:
            if search_type == "similarity":
                docs = self.similarity_search(query, k=k, **kwargs)
                
            elif search_type == "similarity_score_threshold":
                # Use search with relevance scores
                docs_and_similarities = self.similarity_search_with_relevance_scores(
                    query, k=k, score_threshold=score_threshold, **kwargs
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif search_type == "mmr":
                docs = self.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
                )
                        
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Add scores to metadata (if needed and supported)
            if with_score and search_type != "similarity_score_threshold":
                try:
                    docs_with_scores = self.similarity_search_with_score(
                        query, k=len(docs), **kwargs
                    )
                    score_dict = {doc.id: score for doc, score in docs_with_scores}
                    relevance_score_fn = self._get_relevance_score_fn()
                    
                    for doc in docs:
                        if doc.id in score_dict and relevance_score_fn:
                            relevance_score = relevance_score_fn(score_dict[doc.id])
                            doc.metadata = {**(doc.metadata or {}), "score": relevance_score}
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