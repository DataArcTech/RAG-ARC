from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Optional,
    TypeVar,
    Sequence,
    List,
    Generic,
    Literal,
)
from core.utils.data_model import Document
from framework.config import AbstractConfig
from framework.module import AbstractModule


logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="BaseIndexConfig")


class BaseIndexConfig(AbstractConfig):
    """索引配置基类"""
    type: Literal["index"] = "index"
    index_name: str = "default_index"
    index_path: Optional[str] = None

    def build(self) -> "BaseIndex":
        raise NotImplementedError("Subclasses must implement build() method")

class BaseIndex(AbstractModule, Generic[ConfigType], ABC):
    """索引基类，定义索引的基本操作接口"""
    
    config: ConfigType
    
    @abstractmethod
    def add(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """添加文档到索引（会根据ID去重，重复ID的文档不会被添加）

        Args:
            documents: 要添加的文档列表
            embeddings: 预计算的文档嵌入向量

        Returns:
            成功添加的文档ID列表
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """从索引中删除文档

        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他删除条件

        Returns:
            删除是否成功
        """
        pass

    @abstractmethod
    def update(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """更新索引中的文档

        Args:
            documents: 要更新的文档列表（需要包含文档ID）
            embeddings: 预计算的文档嵌入向量
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """根据ID获取文档
        
        Args:
            ids: 文档ID列表
            
        Returns:
            对应的文档列表
        """
        pass

    @abstractmethod
    def save_index(self, index_path: str, index_name: str = "index") -> None:
        """保存索引到磁盘
        
        Args:
            index_path: 保存路径
            index_name: 索引名称
        """
        pass

    @abstractmethod
    def load_index(self, index_path: Optional[str] = None) -> None:
        """从磁盘加载索引
        
        Args:
            index_path: 索引路径，如果为None则使用配置中的路径
        """
        pass

    @abstractmethod
    def build_index(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """构建索引（仅在索引不存在时使用，如果索引已存在会抛出异常）

        Args:
            documents: 用于构建索引的文档列表
            embeddings: 预计算的文档嵌入向量

        Raises:
            RuntimeError: 如果索引已存在
        """
        pass

    @abstractmethod
    def index_exists(self) -> bool:
        """检查索引是否存在

        Returns:
            bool: 索引是否存在
        """
        pass
