import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from framework.module import AbstractModule
from encapsulation.data_model.schema import Chunk


logger = logging.getLogger(__name__)

class BaseRetriever(AbstractModule, ABC):
    def __init__(self, config):
        self.config = config
        self._index = self.config.index_config.build()
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """尝试加载已存在的索引"""
        try:
            if hasattr(self._index, 'load_index'):
                # Check if the index has an index_path in its config
                if hasattr(self._index.config, 'index_path') and self._index.config.index_path:
                    self._index.load_index(self._index.config.index_path)
                else:
                    self._index.load_index()
                logger.info(f"Successfully loaded existing index for {self.get_name()}")
        except Exception as e:
            message = f"Index not found for retriever {self.get_name()}: {e}"
            logger.warning(f"{message}. Index will be empty until chunks are added.")

    def get_default_search_config(self) -> Dict[str, Any]:
        return self.config.search_kwargs.copy()

    @property
    def index(self) -> Any:
        return self._index

    def invoke(self, input: str, **kwargs: Any) -> List[Chunk]:
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return self._get_relevant_chunks(input, **merged_kwargs)

    async def ainvoke(self, input: str, **kwargs: Any) -> List[Chunk]:
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return await self._aget_relevant_chunks(input, **merged_kwargs)

    @abstractmethod
    def _get_relevant_chunks(self, query: str, **kwargs: Any) -> List[Chunk]:
        pass

    async def _aget_relevant_chunks(self, query: str, **kwargs: Any) -> List[Chunk]:
        return await asyncio.to_thread(self._get_relevant_chunks, query, **kwargs)

    def get_name(self) -> str:
        return self.config.type
