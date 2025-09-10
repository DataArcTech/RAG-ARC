from typing import Any, Optional, List, Dict, ClassVar, Collection, Literal, TYPE_CHECKING, Tuple
import logging

from pydantic import ConfigDict, Field, model_validator
from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.retrieval_helper import RetrievalHelper
from encapsulation.database.vector_db.base import BaseVectorDB

logger = logging.getLogger(__name__)


class DenseRetrieverConfig(BaseRetrieverConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["dense"] = "dense"
    
    # Runtime dependencies (injected by vector database)
    vectorstore: Optional[BaseVectorDB] = Field(default=None, exclude=True, description="Vector database instance")
    metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric from vector database")
    # search_type: Literal["similarity", "similarity_score_threshold", "mmr"] = Field(default="similarity", description="Search type")

    
    def build(self) -> "DenseRetriever":
        """Build the DenseRetriever instance"""
        if self.vectorstore is None:
            raise ValueError("DenseRetrieverConfig.vectorstore must not be None. Please provide a valid BaseVectorDB instance.")
        
        return DenseRetriever(config=self)


class DenseRetriever(BaseRetriever[DenseRetrieverConfig]):
    """
    DenseRetriever is a high-performance document retriever based on dense vector databases.
    
    This class implements document retrieval based on vector similarity, supporting multiple search types
    and advanced features such as Maximum Marginal Relevance (MMR) search for diversified result sets.
    
    Key Features:
    - Multiple search types: similarity, similarity_score_threshold, mmr
    - Support for relevance score threshold filtering
    - Maximum Marginal Relevance (MMR) search for result diversity
    - Async operation support
    - Flexible search parameter configuration
    - Compatible with multiple vector databases
    
    Configuration Parameters (from config):
        vectorstore (BaseVectorDB): Vector database instance
        metric (str): Distance metric type ('cosine', 'l2', 'ip')
        k (int): Default number of documents to return
        with_score (bool): Whether to include relevance scores by default
        search_kwargs (dict): Additional search parameters
        
    Runtime Instance Variables:
        _relevance_score_fn: Relevance scoring function
        
    Core Methods:
        - invoke: Main entry point for synchronous retrieval
        - _get_relevant_documents: Execute search and return structured results
        - _get_docs_with_embeddings_for_mmr: Get documents and embeddings for MMR search
        
    Performance Considerations:
        - Similarity search is suitable for most scenarios
        - MMR search provides diversity but has higher computational cost
        - Score threshold filtering may affect the number of returned documents
        
    Typical Usage:
        >>> # Create retriever through vector database
        >>> retriever = vector_db.as_retriever()
        >>> results = retriever.invoke("query text")
        >>> results = retriever.invoke("query text", k=5, search_type="mmr")
    """
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """Allowed search types"""
    
    # Runtime instance variables
    _relevance_score_fn = None
    
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
        """Execute search and return structured results
        
        Args:
            query: Query string
            k: Number of documents to return (defaults to config value)
            search_type: Search type
            score_threshold: Relevance score threshold
            fetch_k: Number of candidate documents for MMR search
            lambda_mult: MMR diversity parameter
            with_score: Whether to include score in metadata
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents
        """
        # Use configuration defaults
        k = k if k is not None else self.config.k
        with_score = with_score if with_score is not None else self.config.with_score
        
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
                docs = self.config.vectorstore.similarity_search(query, k=k, **kwargs)
                
            elif search_type == "similarity_score_threshold":
                # Use search with relevance scores
                docs_and_similarities = self.config.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k, score_threshold=score_threshold, **kwargs
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif search_type == "mmr":
                # Check if vector database supports MMR
                if hasattr(self.config.vectorstore, 'max_marginal_relevance_search'):
                    docs = self.config.vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
                    )
                else:
                    # Vector database doesn't support MMR, fallback to regular similarity search
                    logger.warning(
                        f"Vector database {self.config.vectorstore.__class__.__name__} does not support MMR search, "
                        f"falling back to regular similarity search (k={k})"
                    )
                    docs = self.config.vectorstore.similarity_search(query, k=k, **kwargs)
                        
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Add scores to metadata (if needed and supported)
            if with_score and search_type != "similarity_score_threshold":
                # For non-threshold searches, try to get scores
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
                        logger.debug(f"Unable to get scores: {e}")
            
            logger.debug(f"Retrieved {len(docs)} documents, search type: {search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"Error occurred while retrieving documents: {e}")
            raise
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Asynchronously get documents relevant to the query
        
        Default implementation uses thread pool to execute synchronous version.
        Subclasses can override to provide true async implementation.
        
        Args:
            query: Query string
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents
        """
        try:
            import asyncio
            return await asyncio.to_thread(self._get_relevant_documents, query, **kwargs)
        except AttributeError:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_relevant_documents, query, **kwargs)
    

    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """Get vector database information"""
        info = {
            "vectorstore_class": self.config.vectorstore.__class__.__name__,
            "metric": self.config.metric,
            "k": self.config.k,
            "with_score": self.config.with_score,
            "search_kwargs": self.config.search_kwargs,
            "allowed_search_types": list(self.allowed_search_types),
        }
        
        # If vector database has embedding information, add it to info
        if hasattr(self.config.vectorstore, 'embedding') and self.config.vectorstore.embedding:
            info["embedding_class"] = self.config.vectorstore.embedding.__class__.__name__
        
        return info
    
    def get_name(self) -> str:
        """Get retriever name"""
        return f"{self.config.vectorstore.__class__.__name__}Retriever"