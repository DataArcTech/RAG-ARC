import asyncio
from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import ConfigDict
from core.utils.data_model import Document

class BaseRetriever(ABC):
    """Base Retriever Class
    
    A retrieval system is defined as a system that can accept a string query and return the most "relevant" documents from a certain source.
    
    Usage:
    Retrievers follow the standard runnable interface and should be used through standard methods such as `invoke`, `ainvoke`, etc.
    
    Implementation:
    When implementing a custom retriever, the class should implement the `_get_relevant_documents` method to define the logic for retrieving documents.
    Optionally, an asynchronous native implementation can be provided by overriding the `_aget_relevant_documents` method.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **kwargs):
        """Initialize the retriever
        
        Args:
            **kwargs: Other parameters, such as search_kwargs, tags, metadata, etc.
        """
        self.search_kwargs = kwargs.get("search_kwargs", {})
        self.tags = kwargs.get("tags")
        self.metadata = kwargs.get("metadata")
    
    def invoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Invoke the retriever to get relevant documents
        
        Main entry point for synchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Other parameters passed to the retriever
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> retriever.invoke("query")
        """
        return self._get_relevant_documents(input, **kwargs)
    
    async def ainvoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Asynchronously invoke the retriever to get relevant documents
        
        Main entry point for asynchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Other parameters passed to the retriever
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> await retriever.ainvoke("query")
        """
        return await self._aget_relevant_documents(input, **kwargs)
    
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
        """Get retriever name"""
        return self.__class__.__name__