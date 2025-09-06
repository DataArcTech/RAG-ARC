from typing import Any, Optional, List, Dict, ClassVar, Collection
import logging

from pydantic import ConfigDict, Field, model_validator
from core.retrieval.base import BaseRetriever
from core.utils.data_model import Document
from encapsulation.database.vector_db.VectorStoreBase import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """Vector database retriever
    
    Vector-based document retriever implementation supporting multiple search types:
    - similarity: Similarity search
    - similarity_score_threshold: Similarity search with score threshold
    - mmr: Maximal Marginal Relevance search
    """
    
    vectorstore: 'VectorStore'
    """Vector database instance used for retrieval"""
    
    search_type: str = "similarity"
    """Type of search to perform, defaults to 'similarity'"""
    
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to the search functions"""
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """Allowed search types"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, vectorstore: 'VectorStore', **kwargs):
        """Initialize vector database retriever
        
        Args:
            vectorstore: Vector database instance
            search_type: Search type, defaults to "similarity"
            search_kwargs: Search parameters dictionary
            **kwargs: Other parameters
        """
        self.vectorstore = vectorstore
        self.search_type = kwargs.get("search_type", "similarity")
        self.search_kwargs = kwargs.get("search_kwargs", {})
        
        # Validate search configuration
        self._validate_search_config()
        
        # Call parent initialization
        super().__init__(**kwargs)
    
    def _validate_search_config(self) -> None:
        """Validate search configuration
        
        Raises:
            ValueError: If search type is not in allowed types
            ValueError: If using similarity_score_threshold but no valid score_threshold specified
        """
        if self.search_type not in self.allowed_search_types:
            msg = (
                f"search_type '{self.search_type}' is not allowed. "
                f"Valid values are: {self.allowed_search_types}"
            )
            raise ValueError(msg)
        
        if self.search_type == "similarity_score_threshold":
            score_threshold = self.search_kwargs.get("score_threshold")
            if (score_threshold is None or 
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "When using 'similarity_score_threshold' search type, "
                    "a valid score_threshold (float between 0 and 1) must be specified in search_kwargs"
                )
                raise ValueError(msg)
    
    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search type (Pydantic validator)
        
        Args:
            values: Values to validate
            
        Returns:
            Validated values
            
        Raises:
            ValueError: If search type is invalid
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type '{search_type}' is not allowed. "
                f"Valid values are: {cls.allowed_search_types}"
            )
            raise ValueError(msg)
            
        if search_type == "similarity_score_threshold":
            search_kwargs = values.get("search_kwargs", {})
            score_threshold = search_kwargs.get("score_threshold")
            if (score_threshold is None or 
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "When using 'similarity_score_threshold' search type, "
                    "a valid score_threshold (numeric value between 0 and 1) must be specified in search_kwargs"
                )
                raise ValueError(msg)
        
        return values
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Get documents relevant to the query
        
        Args:
            query: Query string
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents
            
        Raises:
            ValueError: If search type is invalid
        """
        # Merge search parameters
        search_params = {**self.search_kwargs, **kwargs}
        
        # Get number of documents to return, referencing BM25Retriever approach
        k = search_params.get('k', getattr(self, 'k', 4))
        search_params['k'] = k
        
        try:
            if self.search_type == "similarity":
                docs = self.vectorstore.similarity_search(query, **search_params)
                # Ensure only top k documents are returned
                docs = docs[:k]
                
            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    self.vectorstore.similarity_search_with_relevance_scores(
                        query, **search_params
                    )
                )
                docs = [doc for doc, _ in docs_and_similarities]
                # Ensure only top k documents are returned
                docs = docs[:k]
                
            elif self.search_type == "mmr":
                docs = self.vectorstore.max_marginal_relevance_search(
                    query, **search_params
                )
                # Ensure only top k documents are returned
                docs = docs[:k]
                
            else:
                msg = f"Unsupported search type: {self.search_type}"
                raise ValueError(msg)
            
            logger.debug(f"Retrieved {len(docs)} documents, search type: {self.search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Asynchronously get documents relevant to the query
        
        Args:
            query: Query string
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents
            
        Raises:
            ValueError: If search type is invalid
        """
        # Merge search parameters
        search_params = {**self.search_kwargs, **kwargs}
        
        try:
            if self.search_type == "similarity":
                docs = await self.vectorstore.asimilarity_search(query, **search_params)
                
            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    await self.vectorstore.asimilarity_search_with_relevance_scores(
                        query, **search_params
                    )
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif self.search_type == "mmr":
                docs = await self.vectorstore.amax_marginal_relevance_search(
                    query, **search_params
                )
                
            else:
                msg = f"Unsupported search type: {self.search_type}"
                raise ValueError(msg)
            
            logger.debug(f"Asynchronously retrieved {len(docs)} documents, search type: {self.search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"Error asynchronously retrieving documents: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vector database
        
        Args:
            documents: List of documents to add
            **kwargs: Other keyword arguments
            
        Returns:
            List of added document IDs
        """
        try:
            ids = self.vectorstore.add_documents(documents, **kwargs)
            logger.info(f"Successfully added {len(documents)} documents to vector database")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Asynchronously add documents to the vector database
        
        Args:
            documents: List of documents to add
            **kwargs: Other keyword arguments
            
        Returns:
            List of added document IDs
        """
        try:
            ids = await self.vectorstore.aadd_documents(documents, **kwargs)
            logger.info(f"Successfully asynchronously added {len(documents)} documents to vector database")
            return ids
        except Exception as e:
            logger.error(f"Error asynchronously adding documents: {e}")
            raise
    
    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the vector database
        
        Args:
            ids: List of document IDs to delete, if None deletes all documents
            **kwargs: Other keyword arguments
            
        Returns:
            Whether deletion was successful
        """
        try:
            result = self.vectorstore.delete(ids, **kwargs)
            if ids:
                logger.info(f"Deleted {len(ids)} documents")
            else:
                logger.info("Deleted all documents")
            return result
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    async def adelete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Asynchronously delete documents from the vector database
        
        Args:
            ids: List of document IDs to delete, if None deletes all documents
            **kwargs: Other keyword arguments
            
        Returns:
            Whether deletion was successful
        """
        try:
            result = await self.vectorstore.adelete(ids, **kwargs)
            if ids:
                logger.info(f"Asynchronously deleted {len(ids)} documents")
            else:
                logger.info("Asynchronously deleted all documents")
            return result
        except Exception as e:
            logger.error(f"Error asynchronously deleting documents: {e}")
            raise
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents
        """
        try:
            docs = self.vectorstore.get_by_ids(ids)
            logger.debug(f"Retrieved {len(docs)} documents by IDs")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            raise
    
    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
        """Asynchronously get documents by IDs
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents
        """
        try:
            docs = await self.vectorstore.aget_by_ids(ids)
            logger.debug(f"Asynchronously retrieved {len(docs)} documents by IDs")
            return docs
        except Exception as e:
            logger.error(f"Error asynchronously retrieving documents by IDs: {e}")
            raise
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """Get vector database information
        
        Returns:
            Dictionary containing vector database information
        """
        info = {
            "vectorstore_class": self.vectorstore.__class__.__name__,
            "search_type": self.search_type,
            "search_kwargs": self.search_kwargs,
            "allowed_search_types": list(self.allowed_search_types),
        }
        
        # If vector database has embedding information, add it to info
        if hasattr(self.vectorstore, 'embeddings') and self.vectorstore.embeddings:
            info["embedding_class"] = self.vectorstore.embeddings.__class__.__name__
        elif hasattr(self.vectorstore, 'embedding'):
            info["embedding_class"] = self.vectorstore.embedding.__class__.__name__
        
        return info
    
    def get_name(self) -> str:
        """Get retriever name"""
        return f"{self.vectorstore.__class__.__name__}Retriever"
    
    def update_search_params(self, **kwargs: Any) -> None:
        """Update search parameters
        
        Args:
            **kwargs: Search parameters to update
        """
        self.search_kwargs.update(kwargs)
        
        # If search type is updated, re-validate
        if "search_type" in kwargs:
            self.search_type = kwargs["search_type"]
            self._validate_search_config()
        
        logger.debug(f"Updated search parameters: {kwargs}")
    
    def __repr__(self) -> str:
        """Return string representation of the retriever"""
        return (
            f"{self.__class__.__name__}("
            f"vectorstore={self.vectorstore.__class__.__name__}, "
            f"search_type='{self.search_type}', "
            f"search_kwargs={self.search_kwargs})"
        )