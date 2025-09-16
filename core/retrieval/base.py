import asyncio
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

class BaseRetrieverConfig(AbstractConfig):
    type: Literal["retriever"] = "retriever"

    # Index configuration - 支持具体的索引类型
    index_config: Annotated[
        Union[FaissIndexConfig],
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
        self.config = config
        self._index = self.config.index_config.build()
        self._embedding = None

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

    # CRUD 操作代理
    def add_documents(self, documents: List[Document]) -> List[str]:
        # 计算嵌入向量
        if hasattr(self.config, "embedding_config") and self.config.embedding_config is not None:
            embedding_model = self.get_embedding()
            texts = [doc.content for doc in documents]
            embeddings = embedding_model.embed_documents(texts)
            return self._index.add(documents, embeddings)
        else:
            # 对于不需要embedding的索引（如BM25），直接调用add方法
            return self._index.add(documents)

    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return self._index.delete(ids, **kwargs)

    def update_documents(self, documents: List[Document]) -> None:
        # 计算嵌入向量
        if hasattr(self.config, "embedding_config") and self.config.embedding_config is not None:
            embedding_model = self.get_embedding()
            texts = [doc.content for doc in documents]
            embeddings = embedding_model.embed_documents(texts)
            return self._index.update(documents, embeddings)
        else:
            # 对于不需要embedding的索引（如BM25），直接调用update方法
            return self._index.update(documents)

    def build_index(self, documents: List[Document]) -> None:
        """构建索引（仅在索引不存在时使用）

        Args:
            documents: 用于构建索引的文档列表

        Raises:
            RuntimeError: 如果索引已存在
        """
        # 检查索引是否已存在
        if self._index.index_exists():
            raise RuntimeError(
                "Index already exists. Use add_documents() to add documents to existing index, "
                "or delete the existing index first if you want to rebuild it."
            )

        # 计算嵌入向量
        if hasattr(self.config, "embedding_config") and self.config.embedding_config is not None:
            embedding_model = self.get_embedding()
            texts = [doc.content for doc in documents]
            embeddings = embedding_model.embed_documents(texts)
            return self._index.build_index(documents, embeddings)
        else:
            # 对于不需要embedding的索引（如BM25），直接调用build_index方法
            return self._index.build_index(documents)

    def save_index(self, index_path: str, index_name: str = "index") -> None:
        return self._index.save_index(index_path, index_name)

    def load_index(self, index_path: Optional[str] = None) -> None:
        return self._index.load_index(index_path)
