from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    Sequence,
    Iterable,
    Tuple,
    List,
)
import asyncio
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from core.utils.data_model import Document
    from framework.config import AbstractConfig

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound="VectorDB")


class VectorDB(ABC):
    """Vector database base class - encapsulation layer for core database operations"""
    
    def __init__(self, config: "AbstractConfig"):
        """Initialize vector store with configuration injection
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add texts to vector store
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of IDs for added documents
        """
        pass

    async def aadd_texts(self,documents: List[Document]) -> List[str]:
        """Asynchronously add documents to vector store"""
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
    def get_by_ids(self, ids: Sequence[str], /) -> List['Document']:
        """Get documents by IDs
        
        Args:
            ids: List of IDs to retrieve
            
        Returns:
            List of documents
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
    def initialize_from_documents(self, documents: List['Document']) -> None:
        """Initialize this instance from documents(from scratch)
        
        Args:
            documents: List of Document objects to add to the vector database
        """
        pass

    
    
