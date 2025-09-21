import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, List, TypeVar, Generic, Tuple, Literal, Dict, Optional, Union
from typing import Annotated
from pydantic import Field
from framework.module import AbstractModule
from framework.config import AbstractConfig
from core.utils.data_model import Document

from encapsulation.database.vector_db.base import BaseIndexConfig
from encapsulation.database.vector_db.faiss import FaissIndexConfig
from encapsulation.llm.base import LLMBaseConfig
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
from encapsulation.llm.openai import OpenAIConfig

logger = logging.getLogger(__name__)

class BaseRetrieverConfig(AbstractConfig):
    type: Literal["retriever"] = "retriever"

    # Index configuration - 支持具体的索引类型
    index_config: Annotated[
        Union[FaissIndexConfig, BaseIndexConfig],
        Field(discriminator="type")
    ]

    # Embedding configuration - 支持具体的嵌入模型类型
    embedding_config: Optional[Annotated[
        Union[HuggingFaceEmbedConfig, OpenAIConfig],
        Field(discriminator="type")
    ]] = None

    # Search parameters, can be modified at runtime
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 5, "with_score": False, "score_threshold": None})

    @abstractmethod
    def build(self) -> "BaseRetriever":
        raise NotImplementedError("Subclasses must implement build() method")
    
    @classmethod
    def model_rebuild_all(cls):
        """Rebuild all retriever config models to resolve forward references"""
        # Import here to avoid circular imports
        try:
            from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig
            cls.model_rebuild()
        except ImportError:
            pass

ConfigType = TypeVar("ConfigType", bound="BaseRetrieverConfig")



class BaseRetriever(AbstractModule, Generic[ConfigType], ABC):
    config: ConfigType

    def __init__(self, config: ConfigType):
        super().__init__(config=config)
        self._index = config.index_config.build()
        self._embedding = None
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """尝试加载已存在的索引"""
        try:
            if hasattr(self._index, 'load_index'):
                self._index.load_index()
                logger.info(f"Successfully loaded existing index for {self.get_name()}")
        except Exception as e:
            message = f"Index not found for retriever {self.get_name()}: {e}"
            logger.warning(f"{message}. Please use IndexManager to build the index first.")
            raise RuntimeError(message)

    def get_default_search_config(self) -> Dict[str, Any]:
        return self.config.search_kwargs.copy()

    @property
    def index(self) -> Any:
        return self._index

    def get_embedding(self) -> Any:
        if self._embedding is None:
            if hasattr(self.config, "embedding_config") and self.config.embedding_config is not None:
                self._embedding = self.config.embedding_config.build()
            else:
                raise ValueError("This retriever does not have an embedding_config")
        return self._embedding

    def invoke(self, input: str, **kwargs: Any) -> List[Document]:
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return self._get_relevant_documents(input, **merged_kwargs)

    async def ainvoke(self, input: str, **kwargs: Any) -> List[Document]:
        default_config = self.get_default_search_config()
        merged_kwargs = {**default_config, **kwargs}
        return await self._aget_relevant_documents(input, **merged_kwargs)

    @abstractmethod
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        pass

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, **kwargs)

    def get_name(self) -> str:
        return self.config.type

    def build_index(self, documents: List[Document]) -> None:
        """        
        Args:
            documents: 用于构建索引的文档列表

        Raises:
            NotImplementedError: 如果子类没有实现此方法
            RuntimeError: 如果索引已存在
        """
        raise NotImplementedError(f"Retriever {self.__class__.__name__} does not support build_index operation")