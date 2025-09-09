from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Optional,
    TypeVar,
    Sequence,
    List,
    Generic,
    Literal,
)
import asyncio
from pydantic import Field, field_validator

from encapsulation.llm.base import LLMBase
from core.utils.data_model import Document
from framework.config import AbstractConfig
from framework.module import AbstractModule

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="BaseVectorDBConfig")


class BaseVectorDBConfig(AbstractConfig):
    """
    Abstract base class for all vector database configurations.
    - Subclasses must define `type: Literal["xxx"]`
    - Subclasses must implement build() to return the corresponding VectorDB
    """
    type: Literal["base_vector_db"] = "base_vector_db"


    k: int = Field(default=10, description="Default number of documents to return in search", gt=0)
    with_score: bool = Field(default=True, description="Whether to include relevance scores in results")
    search_kwargs: dict = Field(default_factory=dict, description="Additional search parameters for retrieval")
    
    # Vector database specific configuration parameters
    metric: str = Field(default="cosine", description="Distance metric for similarity calculation")
    normalize_L2: bool = Field(default=False, description="Whether to normalize vectors for cosine similarity")
    search_params: dict = Field(default_factory=dict, description="Parameters for search operations")
    
    # embedding model instance
    embedding: Optional[LLMBase] = Field(default=None, description="已加载的嵌入模型实例用于文本向量化", exclude=True)


    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"k must be greater than 0, but got {v}")
        return v

    @abstractmethod
    def build(self) -> "BaseVectorDB":
        """Build the vector database"""
        raise NotImplementedError("Subclasses must implement build() method")


class BaseVectorDB(AbstractModule, Generic[ConfigType], ABC):
    """Base Vector Database Class
    
    A vector database system is defined as a system that can store, index, and search high-dimensional vectors
    efficiently. It provides operations for adding, deleting, and retrieving documents based on vector similarity.
    
    Usage:
    Vector databases follow the standard configuration injection pattern and should be initialized with
    appropriate configuration objects.
    
    Implementation:
    When implementing a custom vector database, the class should implement the abstract methods to define
    the logic for document storage, retrieval, and management operations.
    """
    
    config: ConfigType
    
    def __init__(self, config: ConfigType):
        """Initialize the vector database
        
        Args:
            config: Configuration object containing all parameters
        """
        super().__init__(config=config)
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of IDs for added documents
        """
        pass

    async def aadd_documents(self, documents: List[Document]) -> List[str]:
        """Asynchronously add documents to vector store
        
        Default implementation that wraps the synchronous version in a thread pool.
        Subclasses can override this for a truly asynchronous implementation.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of IDs for added documents
        """
        try:
            return await asyncio.to_thread(self.add_documents, documents)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.add_documents, documents)

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs or other conditions
        
        Args:
            ids: List of IDs to delete. If None, delete all. Default is None
            **kwargs: Additional keyword arguments
            
        Returns:
            Optional[bool]: True if deletion successful, False otherwise, None if not implemented
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by IDs
        
        Args:
            ids: List of IDs to retrieve
            
        Returns:
            List of documents
        """
        pass

    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to query
        
        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of documents most similar to query
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Return documents most similar to query with similarity scores
        
        Args:
            query: Query string
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        pass

    @abstractmethod
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save vector database to local filesystem
        
        Args:
            folder_path: Directory path to save the vector database
            index_name: Base name for saved files (without extension)
        """
        pass

    @abstractmethod
    def load_from_folder(self, folder_path: str) -> None:
        """Initialize this instance by loading from folder(from saved files)
        
        Args:
            folder_path: Directory path containing saved vector database files
        """
        pass

    @abstractmethod
    def initialize_from_documents(self, documents: List[Document]) -> None:
        """Initialize this instance from documents(from scratch)
        
        Args:
            documents: List of Document objects to add to the vector database
        """
        pass

    def as_retriever(self, **kwargs: Any):
        """Return a BaseRetriever from this vector database
        
        Args:
            **kwargs: Additional parameters for retriever configuration
            
        Returns:
            BaseRetriever instance configured with this vector database
        """
        # Delayed import to avoid circular dependency
        try:
            from core.retrieval.base import BaseRetrieverConfig
        except ImportError:
            raise ImportError("BaseRetriever not available. Make sure core.retrieval.base is properly installed.")
        
        # Create retriever configuration
        retriever_config = BaseRetrieverConfig(
            vectorstore=self,
            **kwargs
        )
        
        return retriever_config.build()
    
    def get_name(self) -> str:
        """Get the vector database's unique name from its config 'type' field."""
        return self.config.type

