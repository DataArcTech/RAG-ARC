from framework.module import AbstractModule
from abc import abstractmethod
from encapsulation.data_model.schema import Chunk
from typing import List

class BaseGraphRetriever(AbstractModule):
    """Base class for graph-based retrievers"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """
        Retrieve relevant chunks for the given query

        Args:
            query: Natural language query
            top_k: Number of top chunks to return

        Returns:
            List of Chunk objects sorted by relevance
        """
        pass

    def invoke(self, query: str, **kwargs) -> List[Chunk]:
        """Standard interface method for compatibility"""
        top_k = kwargs.get('k', kwargs.get('top_k', 10))
        owner_id = kwargs.get('owner_id')
        return_subgraph_info = kwargs.get('return_subgraph_info', False)
        return self.retrieve(query, top_k, return_subgraph_info=return_subgraph_info, owner_id=owner_id)

    def get_name(self) -> str:
        """Get retriever name"""
        return self.config.type