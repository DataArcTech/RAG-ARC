from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from encapsulation.data_model.schema import Document
from framework.module import AbstractModule


class GraphStore(AbstractModule):
    """Graph database base class for Document-based graphs"""

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build graph from a list of Documents."""
        raise NotImplementedError

    @abstractmethod
    def update_index(self, documents: List[Document]) -> Optional[bool]:
        """Update existing documents' graphs in the database."""
        raise NotImplementedError

    @abstractmethod
    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete documents and their graphs by IDs. Delete all if ids is None."""
        raise NotImplementedError
    
    @abstractmethod
    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all documents and their graphs. Requires confirmation."""
        raise NotImplementedError


    @abstractmethod
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Retrieve documents (including their graphs) by IDs."""
        raise NotImplementedError

    @abstractmethod
    def save_index(self, path: str, name: str = "index") -> None:
        """Persist the graph database to filesystem."""
        raise NotImplementedError

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load persisted graph database from filesystem."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Run a query on the graph database."""
        raise NotImplementedError
    
    @abstractmethod
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Retrieve documents (including their graphs) by IDs."""
        raise NotImplementedError

    @abstractmethod
    def get_graph_db_info(self) -> Dict[str, Any]:
        """Return statistics or metadata about the graph database."""
        raise NotImplementedError
