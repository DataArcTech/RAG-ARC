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
    TYPE_CHECKING,
    Annotated,
    Union,
    Dict,
)
import asyncio
from pydantic import Field, field_validator, model_validator

from encapsulation.llm.base import LLMBase
from core.utils.data_model import Document
from framework.config import AbstractConfig
from framework.module import AbstractModule
from framework.shared_module_decorator import shared_module


from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
from encapsulation.llm.openai import OpenAIConfig
from encapsulation.llm.qwen3 import QwenConfig

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="BaseVectorDBConfig")


class BaseVectorDBConfig(AbstractConfig):
    """
    Abstract base class for all vector database configurations.
    - Subclasses must define `type: Literal["xxx"]`
    - Subclasses must implement build() to return the corresponding VectorDB
    
    嵌入模型配置：
    - embedding_config: 内联配置嵌入模型（支持共享实例）
    
    注意: 使用 @shared_module 装饰器，相同配置的向量数据库实例会被自动共享
    """
    type: Literal["base_vector_db"] = "base_vector_db"

    index_path: Optional[str] = Field(default=None, description="Path to vector database index file")

    # Vector database specific configuration parameters
    metric: str = Field(default="cosine", description="Distance metric for similarity calculation")
    normalize_L2: bool = Field(default=False, description="Whether to normalize vectors for cosine similarity")
    
    # Runtime search configuration
    k: int = Field(default=10, description="Default number of documents to return in search", gt=0, exclude=True)
    with_score: bool = Field(default=True, description="Whether to include relevance scores in results", exclude=True)
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"use_phrase_query": False}, 
        description="Additional search parameters including use_phrase_query",
        exclude=True
    )
    
    embedding_config: HuggingFaceEmbedConfig = Field(
        description="内联配置嵌入模型（支持共享实例）"
    )

    
    @model_validator(mode='after')
    def validate_embedding_config(self) -> 'BaseVectorDBConfig':
        """确保嵌入模型配置已提供"""
        if self.embedding_config is None:
            raise ValueError("embedding_config is required")
        return self

    def _get_embedding(self) -> LLMBase:
        """获取嵌入模型实例

        Returns:
            LLMBase: 嵌入模型实例
            
        Raises:
            ValueError: 当配置无效时
        """
        return self.embedding_config.build()


    @abstractmethod
    def build(self) -> "BaseVectorDB":
        """Build the vector database
        """
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
    
    def __init__(self, config: ConfigType, embedding: LLMBase):
        """Initialize the vector database
        
        Args:
            config: Configuration object containing all parameters
            embedding: 嵌入模型实例（由配置类的build方法传入）
        """
        super().__init__(config=config)
        self.embedding = embedding
    
    def get_default_search_config(self) -> dict:
        """获取默认搜索配置
        
        Returns:
            dict: 默认搜索配置，包含k、with_score、search_kwargs等参数
        """
        return self.config.default_search_config.copy()
    
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
    def save_local(self, index_path: str, index_name: str = "index") -> None:
        """Save vector database to local filesystem
        
        Args:
            index_path: Directory path to save the vector database
            index_name: Base name for saved files (without extension)
        """
        pass

    @abstractmethod
    def load_local(self, index_path: str) -> None:
        """Initialize this instance by loading from local(from saved files)
        
        Args:
            index_path: Directory path containing saved vector database files
        """
        pass

    @abstractmethod
    def from_documents(self, documents: List[Document]) -> None:
        """Initialize this instance from documents(from scratch)
        
        Args:
            documents: List of Document objects to add to the vector database
        """
        pass

    def as_retriever(self, k: Optional[int] = None, with_score: Optional[bool] = None, search_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Return a BaseRetriever from this vector database
        
        Args:
            k: Number of documents to return
            with_score: Whether to include relevance scores in results
            search_kwargs: Additional search parameters
            **kwargs: Additional parameters for retriever configuration

        Returns:
            BaseRetriever instance configured with this vector database
        """
        # Delayed import to avoid circular dependency
        try:
            from core.retrieval.base import BaseRetrieverConfig
        except ImportError:
            raise ImportError("BaseRetriever not available. Make sure core.retrieval.base is properly installed.")
        
        runtime_k = k or self.config.k
        runtime_with_score = with_score or self.config.with_score
        runtime_search_kwargs = search_kwargs or self.config.search_kwargs.copy()

        # Create retriever configuration
        retriever_config = BaseRetrieverConfig(
            vectorstore=self,
            k=runtime_k,
            with_score=runtime_with_score,
            search_kwargs=runtime_search_kwargs,
            **kwargs,
        )
        
        return retriever_config.build()
    
    def get_name(self) -> str:
        """Get the vector database's unique name from its config 'type' field."""
        return self.config.type


# 解决前向引用（字符串）
try:
    BaseVectorDBConfig.model_rebuild()
except Exception:
    pass