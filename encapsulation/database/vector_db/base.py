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


from encapsulation.data_model.schema import Chunk
from framework.module import AbstractModule


class VectorDB(AbstractModule):
    """Vector database base class - encapsulation layer for core database operations"""

    @abstractmethod
    def build_index(self, chunks: List[Chunk]) -> None:
        """Build index from chunks

        Args:
            chunks: List of Chunk objects to build index from
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
    def get_by_ids(self, ids: Sequence[str]) -> List[Chunk]:
        """Get chunks by IDs

        Args:
            ids: List of IDs to retrieve

        Returns:
            List of chunks
        """
        pass

    @abstractmethod
    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete chunks by IDs

        Args:
            ids: List of IDs to delete. If None, delete all. Default is None

        Returns:
            Optional[bool]: True if deletion successful, False otherwise, None if not implemented
        """
        pass

    @abstractmethod
    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks. Requires confirmation."""
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
    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update chunks in index

        Args:
            chunks: List of Chunk objects to update

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