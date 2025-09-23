from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Optional,
    Sequence,
    List,
    Dict,
    Any,
)


from encapsulation.data_model.schema import Document
from framework.module import AbstractModule


class VectorDB(AbstractModule):
    """Vector database base class - encapsulation layer for core database operations"""

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build index from documents

        Args:
            documents: List of Document objects to build index from
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load index from provided folder path

        Args:
            path: Directory path containing saved vector database files
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by IDs

        Args:
            ids: List of IDs to retrieve

        Returns:
            List of documents
        """
        pass

    @abstractmethod
    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete documents by IDs

        Args:
            ids: List of IDs to delete. If None, delete all. Default is None

        Returns:
            Optional[bool]: True if deletion successful, False otherwise, None if not implemented
        """
        pass

    @abstractmethod
    def save_index(self, path: str, name: str = "index") -> None:
        """Save index to filesystem path

        Args:
            path: Directory path to save the vector database
            name: Base name for saved files (without extension)
        """
        pass

    @abstractmethod
    def update_index(self, documents: List[Document]) -> Optional[bool]:
        """Update documents in index

        Args:
            documents: List of Document objects to update

        Returns:
            Optional[bool]: True if update successful, False otherwise, None if not implemented
        """
        pass

    @abstractmethod
    def get_vector_db_info(self) -> Dict[str, Any]:
        """Get vector database information

        Returns:
            Dictionary containing database info (size, dimensions, etc.)
        """
        pass