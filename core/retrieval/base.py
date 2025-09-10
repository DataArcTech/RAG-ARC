import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, TypeVar, Generic, Tuple, Literal, Dict
from pydantic import Field
from framework.module import AbstractModule
from framework.config import AbstractConfig
from core.utils.data_model import Document


ConfigType = TypeVar("ConfigType", bound="BaseRetrieverConfig")

class  BaseRetrieverConfig(AbstractConfig):
    """
    Abstract base class for all retriever configurations.
    - Subclasses must define `type: Literal["xxx"]`
    - Subclasses must implement build() to return the corresponding Retriever
    """
    type: Literal["base_retriever"] = "base_retriever"

    k: int = Field(default=10, description="Default number of documents to return in search", gt=0, exclude=True)
    with_score: bool = Field(default=True, description="Whether to include relevance scores in results", exclude=True)
    search_kwargs: Dict[str, Any] = Field(
        default_factory={}, 
        description="Additional search parameters including use_phrase_query",
        exclude=True
    )
    
    
    @abstractmethod
    def build(self) -> "BaseRetriever":
        """Build the retriever"""
        raise NotImplementedError("Subclasses must implement build() method")



class BaseRetriever(AbstractModule, Generic[ConfigType], ABC):
    """Base Retriever Class
    
    A retrieval system is defined as a system that can accept a string query and return the most "relevant" documents from a certain source.
    
    Usage:
    Retrievers follow the standard runnable interface and should be used through standard methods such as `invoke`, `ainvoke`, etc.
    
    Implementation:
    When implementing a custom retriever, the class should implement the `_get_relevant_documents` method to define the logic for retrieving documents.
    Optionally, an asynchronous native implementation can be provided by overriding the `_aget_relevant_documents` method.
    """
    
    config: ConfigType
    
    def __init__(self, config: ConfigType):
        """Initialize the retriever
        
        Args:
            **kwargs: Other parameters, such as search_kwargs, tags, metadata, etc.
        """
        super().__init__(config=config)
    
    def get_default_search_config(self) -> dict:
        """获取默认搜索配置
        
        Returns:
            dict: 默认搜索配置，包含k、with_score、search_kwargs等参数
        """
        return self.config.search_kwargs.copy()
    
    def invoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Invoke the retriever to get relevant documents
        
        Main entry point for synchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Runtime parameters (k, with_score, search_kwargs, etc.)
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> retriever.invoke("query")
            >>> retriever.invoke("query", k=5, with_score=True)
        """
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return self._get_relevant_documents(input, **merged_kwargs)
    
    async def ainvoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Asynchronously invoke the retriever to get relevant documents
        
        Main entry point for asynchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Runtime parameters (k, with_score, search_kwargs, etc.)
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> await retriever.ainvoke("query")
            >>> await retriever.ainvoke("query", k=5, with_score=True)
        """
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return await self._aget_relevant_documents(input, **merged_kwargs)
    
    @abstractmethod
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Get documents relevant to the query
        
        Args:
            query: String used to find relevant documents
            **kwargs: Other parameters
            
        Returns:
            List of relevant documents
        """
        pass
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Asynchronously get documents relevant to the query
        
        Default implementation that wraps the synchronous version in a thread pool.
        Subclasses can override this for a truly asynchronous implementation.
        
        Args:
            query: String used to find relevant documents
            **kwargs: Other parameters
            
        Returns:
            List of relevant documents
        """

        try:
            return await asyncio.to_thread(self._get_relevant_documents, query, **kwargs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_relevant_documents, query, **kwargs)
    
    def get_name(self) -> str:
        """Get the retriever's unique name from its config 'type' field."""
        return self.config.type