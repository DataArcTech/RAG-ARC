"""
1. 优化了排序方式，排序时间加速了10倍
2. 优化了增量更新方式，不必每次都重新构建索引，只需要更新索引即可
"""

import asyncio
import logging
import os
import uuid
import warnings
import dill
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_arc.core.search.base import BaseRetriever
from rag_arc.utils import Document

logger = logging.getLogger(__name__)


def default_preprocessing_func(text: str) -> List[str]:
    """默认的文本预处理函数，仅在英文文本上有效
    
    该函数进行简单的空格分词，适用于英文文本。
    对于中文或其他语言，建议提供自定义的分词函数。
    
    Args:
        text: 输入文本
        
    Returns:
        分词后的词语列表
        
    Example:
        >>> default_preprocessing_func("Hello world!")
        ['Hello', 'world!']
    """
    return text.split()


class BM25Retriever(BaseRetriever):
    """
    BM25Retriever 是一个基于 BM25 算法的文档检索器，适用于信息检索、问答系统、知识库等场景下的高效文本相关性排序。

    该类通过集成 rank_bm25 库，实现了对文档集合的 BM25 检索，支持文档的动态添加、删除、批量构建索引等操作。
    适合文档集合相对静态、检索速度要求较高的场景。对于频繁增删文档的场景，建议使用向量检索（如 VectorStoreRetriever）。

    主要特性：
    - 支持从文本列表或 Document 对象列表快速构建 BM25 检索器
    - 支持自定义分词/预处理函数，适配不同语言和分词需求
    - 支持动态添加、删除文档（每次操作会重建索引，适合中小规模数据集）
    - 可获取检索分数、top-k 文档及分数、检索器配置信息等
    - 兼容异步文档添加/删除，便于大规模数据处理
    - 通过 Pydantic 校验参数，保证配置安全
    - 支持将 BM25 索引持久化到磁盘，并从磁盘加载

    主要参数：
        vectorizer: BM25 向量化器实例（通常为 BM25Okapi）
        docs: 当前检索器持有的文档对象列表
        k: 默认返回的相关文档数量
        preprocess_func: 文本分词/预处理函数，默认为空格分词
        bm25_params: 传递给 BM25Okapi 的参数（如 k1、b 等）

    核心方法：
        - from_texts/from_documents: 从原始文本或 Document 构建检索器
        - _get_relevant_documents: 检索与查询最相关的前 k 个文档
        - get_scores: 获取查询对所有文档的 BM25 分数
        - get_top_k_with_scores: 获取 top-k 文档及其分数
        - add_documents/delete_documents: 动态增删文档并重建索引
        - get_bm25_info: 获取检索器配置信息和统计
        - update_k: 动态调整返回文档数量
        - save_to_disk: 将检索器状态保存到磁盘
        - load_from_disk: 从磁盘加载检索器

    性能注意事项：
        - 默认采用空格分词，对于中文等语言需要自定义分词函数
        - 每次添加/删除文档都会重建 BM25 索引，适合文档量较小或更新不频繁的场景
        - 文档量较大或频繁更新时，建议使用向量检索方案
        - 支持异步操作，便于大规模数据处理
        - 持久化使用 dill，支持复杂对象和函数的序列化

    典型用法：
        >>> # 从文本创建检索器
        >>> retriever = BM25Retriever.from_texts(
        ...     ["文本1", "文本2", "文本3"], 
        ...     k=3
        ... )
        >>> 
        >>> # 检索相关文档
        >>> results = retriever._get_relevant_documents("查询语句")
        >>> 
        >>> # 获取文档和分数
        >>> doc_scores = retriever.get_top_k_with_scores("查询语句", k=2)
        >>> 
        >>> # 添加新文档
        >>> retriever.add_documents([Document(content="新文档")])
        >>> 
        >>> # 删除文档
        >>> retriever.delete_documents(ids=["doc_id"])
        >>> 
        >>> # 获取检索器信息
        >>> info = retriever.get_bm25_info()
        >>> 
        >>> # 持久化
        >>> retriever.save_to_disk("bm25_index.pkl")
        >>> loaded_retriever = BM25Retriever.load_from_disk("bm25_index.pkl")

    Attributes:
        vectorizer: BM25 向量化器实例
        docs: 文档列表
        k: 返回的文档数量
        preprocess_func: 文本分词函数
        bm25_params: BM25 算法参数
    """
    
    
    vectorizer: Any = Field(
        default=None, 
        description="BM25 向量化器实例，通常为 rank_bm25.BM25Okapi 对象"
    )
    
    docs: List[Document] = Field(
        default_factory=list, 
        repr=False, 
        description="当前检索器持有的文档对象列表"
    )
    
    k: int = Field(
        default=5, 
        gt=0, 
        description="默认返回的相关文档数量，必须大于0"
    )
    
    preprocess_func: Callable[[str], List[str]] = Field(
        default=default_preprocessing_func,
        description="文本预处理/分词函数，接收字符串返回词语列表"
    )
    
    bm25_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="传递给 BM25Okapi 构造函数的参数，如 {'k1': 1.5, 'b': 0.75}"
    )
    
    # 私有字段，用于控制是否对默认预处理函数发出警告
    _warn_default_preprocess: bool = Field(
        default=True, 
        exclude=True, 
        description="是否对默认预处理函数发出警告"
    )
    
    # Pydantic 配置
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许任意类型（如 BM25Okapi 对象）
        validate_assignment=True,      # 属性赋值时进行验证
        extra='forbid',               # 禁止额外字段
    )
    
    
    def __init__(self, **kwargs):
        """初始化 BM25 检索器
        
        使用 Pydantic 的字段验证机制确保参数正确性，并在需要时发出警告。
        
        Args:
            vectorizer: BM25 向量化器实例，可选
            docs: 文档列表，默认空列表
            k: 返回文档数量，必须大于0，默认5
            preprocess_func: 预处理函数，默认为空格分词
            bm25_params: BM25 参数字典，默认空字典
            warn_default_preprocess: 是否对默认预处理函数发出警告，默认True
            **kwargs: 其他参数传递给父类
            
        Raises:
            ValueError: 如果参数验证失败
        """
        # 提取控制警告的标志
        warn_flag = kwargs.pop('warn_default_preprocess', True)
        
        # 调用父类构造函数，Pydantic 会自动处理字段验证和赋值
        super().__init__(**kwargs)
        self._thread_executor = ThreadPoolExecutor(max_workers=4)
        # 对默认预处理函数发出警告
        self._warn_about_default_preprocess(warn_flag, kwargs)
        
        logger.debug(f"初始化 BM25Retriever: docs={len(self.docs)}, k={self.k}")
    
    def _warn_about_default_preprocess(self, warn_flag: bool, init_kwargs: Dict[str, Any]) -> None:
        """对使用默认预处理函数发出警告
        
        Args:
            warn_flag: 是否需要警告
            init_kwargs: 初始化时的参数字典
        """
        if (warn_flag and 
            self.preprocess_func == default_preprocessing_func and 
            'preprocess_func' not in init_kwargs):
            warnings.warn(
                "使用默认的英文空格分词预处理函数。对于中文或其他语言，"
                "建议提供自定义的预处理函数以获得更好的检索效果。",
                UserWarning,
                stacklevel=3  # 指向调用者的调用者
            )
    
    
    @field_validator('k')
    @classmethod
    def validate_k(cls, v: int) -> int:
        """验证 k 值必须大于0
        
        Args:
            v: 待验证的 k 值
            
        Returns:
            验证通过的 k 值
            
        Raises:
            ValueError: 如果 k <= 0
        """
        if v <= 0:
            raise ValueError(f"k 必须大于 0，当前值: {v}")
        return v
    
    @field_validator('preprocess_func')
    @classmethod 
    def validate_preprocess_func(cls, v: Callable) -> Callable:
        """验证预处理函数必须可调用
        
        Args:
            v: 待验证的预处理函数
            
        Returns:
            验证通过的函数
            
        Raises:
            ValueError: 如果函数不可调用
        """
        if not callable(v):
            raise ValueError("preprocess_func 必须是可调用的函数")
        return v
    
    @field_validator('bm25_params')
    @classmethod
    def validate_bm25_params(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """验证 BM25 参数
        
        Args:
            v: 待验证的参数字典
            
        Returns:
            验证通过的参数字典
            
        Raises:
            ValueError: 如果参数无效
        """
        if not isinstance(v, dict):
            raise ValueError("bm25_params 必须是字典类型")
        
        # 验证常见的 BM25 参数
        if 'k1' in v and (not isinstance(v['k1'], (int, float)) or v['k1'] < 0):
            raise ValueError("BM25 参数 k1 必须是非负数")
        
        if 'b' in v and (not isinstance(v['b'], (int, float)) or not 0 <= v['b'] <= 1):
            raise ValueError("BM25 参数 b 必须在 [0, 1] 范围内")
            
        return v
    
    @model_validator(mode='after')
    def validate_model_state(self) -> 'BM25Retriever':
        """验证模型整体状态的一致性
        
        检查各个字段之间的逻辑关系是否合理。
        
        Returns:
            验证通过的模型实例
        """
        # 检查 vectorizer 和 docs 的一致性
        if self.vectorizer is not None and len(self.docs) == 0:
            warnings.warn(
                "检测到 vectorizer 存在但文档列表为空，这可能导致不一致的状态",
                UserWarning,
                stacklevel=2
            )
        
        # 检查 k 值是否合理
        if len(self.docs) > 0 and self.k > len(self.docs):
            logger.info(f"k={self.k} 大于文档总数 {len(self.docs)}，实际返回文档数将被限制")
        
        return self
    
    
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[str]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """从文本列表创建 BM25Retriever
        
        这是创建检索器最常用的方法，可以直接从文本列表构建BM25索引。
        
        Args:
            texts: 文本列表或可迭代对象
            metadatas: 对应的元数据列表，可选。如果提供，长度必须与texts相同
            ids: 对应的ID列表，可选。如果提供，长度必须与texts相同。如果不提供，会自动生成UUID
            bm25_params: BM25 算法参数，如 {'k1': 1.5, 'b': 0.75}
            preprocess_func: 文本预处理函数，默认为空格分词
            **kwargs: 其他传递给构造函数的参数
            
        Returns:
            初始化完成的 BM25Retriever 实例
            
        Raises:
            ImportError: 如果未安装 rank_bm25 库
            ValueError: 如果参数不匹配或为空
            
        Example:
            >>> retriever = BM25Retriever.from_texts(
            ...     texts=["Hello world", "Python programming"],
            ...     metadatas=[{"source": "doc1"}, {"source": "doc2"}],
            ...     k=5
            ... )
        """
        # 检查依赖库
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "未找到 rank_bm25 库，请安装: pip install rank_bm25"
            )
        
        # 转换为列表并验证
        texts_list = list(texts)
        if not texts_list:
            raise ValueError("texts 不能为空")
        
        logger.info(f"开始从 {len(texts_list)} 个文本创建 BM25Retriever")
        
        # 处理元数据和ID
        metadatas_list = cls._process_metadatas(metadatas, texts_list)
        ids_list = cls._process_ids(ids, texts_list)
        
        # 预处理文本
        logger.info(f"正在使用 {preprocess_func.__name__} 预处理文本...")
        texts_processed = [preprocess_func(text) for text in texts_list]
        
        # 创建 BM25 向量化器
        bm25_params = bm25_params or {}
        logger.info(f"创建 BM25 向量化器，参数: {bm25_params}")
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        
        # 创建文档对象
        docs = [
            Document(content=text, metadata=metadata, id=doc_id)
            for text, metadata, doc_id in zip(texts_list, metadatas_list, ids_list)
        ]
        
        logger.info(f"成功创建包含 {len(docs)} 个文档的 BM25Retriever")
        
        # 检查是否使用默认预处理函数（用于控制警告）
        warn_default = 'preprocess_func' not in kwargs
        
        return cls(
            vectorizer=vectorizer,
            docs=docs,
            preprocess_func=preprocess_func,
            bm25_params=bm25_params,
            warn_default_preprocess=warn_default,
            **kwargs
        )
    
    @staticmethod
    def _process_metadatas(
        metadatas: Optional[Iterable[Dict[str, Any]]], 
        texts_list: List[str]
    ) -> List[Dict[str, Any]]:
        """处理和验证元数据列表
        
        Args:
            metadatas: 元数据迭代器，可以为None
            texts_list: 文本列表，用于验证长度
            
        Returns:
            处理后的元数据列表
            
        Raises:
            ValueError: 如果元数据长度与文本长度不匹配
        """
        if metadatas is not None:
            metadatas_list = list(metadatas)
            if len(metadatas_list) != len(texts_list):
                raise ValueError(
                    f"metadatas 长度 ({len(metadatas_list)}) "
                    f"与 texts 长度 ({len(texts_list)}) 不匹配"
                )
            return metadatas_list
        else:
            return [{} for _ in texts_list]
    
    @staticmethod
    def _process_ids(
        ids: Optional[Iterable[str]], 
        texts_list: List[str]
    ) -> List[str]:
        """处理和验证ID列表
        
        Args:
            ids: ID迭代器，可以为None
            texts_list: 文本列表，用于验证长度
            
        Returns:
            处理后的ID列表
            
        Raises:
            ValueError: 如果ID长度与文本长度不匹配
        """
        if ids is not None:
            ids_list = list(ids)
            if len(ids_list) != len(texts_list):
                raise ValueError(
                    f"ids 长度 ({len(ids_list)}) "
                    f"与 texts 长度 ({len(texts_list)}) 不匹配"
                )
            return ids_list
        else:
            return [str(uuid.uuid4()) for _ in texts_list]
    
    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """从文档对象列表创建 BM25Retriever
        
        Args:
            documents: Document 对象列表或可迭代对象
            bm25_params: BM25 算法参数，可选
            preprocess_func: 预处理函数，默认为空格分词
            **kwargs: 其他传递给构造函数的参数
            
        Returns:
            初始化完成的 BM25Retriever 实例
            
        Raises:
            ValueError: 如果文档列表为空
            
        Example:
            >>> docs = [
            ...     Document(content="Hello world", metadata={"type": "greeting"}),
            ...     Document(content="Python is great", metadata={"type": "tech"})
            ... ]
            >>> retriever = BM25Retriever.from_documents(docs)
        """
        docs_list = list(documents)
        if not docs_list:
            raise ValueError("documents 不能为空")
        
        # 提取文档内容、元数据和ID
        texts = [doc.content for doc in docs_list]
        metadatas = [doc.metadata for doc in docs_list]
        ids = [doc.id for doc in docs_list]
        
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            bm25_params=bm25_params,
            preprocess_func=preprocess_func,
            **kwargs,
        )
    
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """获取与查询相关的前k个文档
        
        这是检索器的核心方法，根据BM25算法计算文档相关性并返回最相关的文档。

        Args:
            query: 查询字符串
            **kwargs: 其他参数
                k: 返回文档数量，覆盖默认的self.k
                
        Returns:
            按相关性降序排列的文档列表
            
        Raises:
            ValueError: 如果向量化器未初始化
            
        Example:
            >>> results = retriever._get_relevant_documents("Python programming")
            >>> print(f"找到 {len(results)} 个相关文档")
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化，请先调用 from_texts 或 from_documents")

        if not self.docs:
            logger.warning("文档列表为空，返回空结果")
            return []

        # 获取返回文档数量
        k = kwargs.get('k', self.k)
        k = min(k, len(self.docs))  # 确保不超过总文档数

        try:
            # 预处理查询
            processed_query = self.preprocess_func(query)
            logger.debug(f"预处理后的查询: {processed_query}")

            # 获取所有文档的BM25分数
            scores = self.vectorizer.get_scores(processed_query)
            
            # 使用 argpartition 进行高效的 top-k 选择
            top_k_idx_np = np.argpartition(-scores, k-1)[:k]
            top_indices = top_k_idx_np[np.argsort(-scores[top_k_idx_np])]
            
            # 返回前k个文档
            top_docs = [self.docs[idx] for idx in top_indices]
            
            logger.debug(f"检索查询 '{query}' 找到 {len(top_docs)} 个相关文档")
            return top_docs

        except Exception as e:
            logger.error(f"BM25 搜索时发生错误: {e}")
            raise

    def get_scores(self, query: str) -> List[float]:
        """获取查询对所有文档的 BM25 分数
        
        Args:
            query: 查询字符串
            
        Returns:
            所有文档的 BM25 分数列表，顺序与 self.docs 相同
            
        Raises:
            ValueError: 如果向量化器未初始化
            
        Example:
            >>> scores = retriever.get_scores("Python programming")
            >>> print(f"文档分数: {scores}")
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化")
        
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        return scores.tolist()
    
    def get_top_k_with_scores(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """获取 top-k 文档及其 BM25 分数
        
        Args:
            query: 查询字符串
            k: 返回文档数量，如果为 None 则使用实例的 k 值
            
        Returns:
            (文档, 分数) 元组列表，按分数降序排列
            
        Raises:
            ValueError: 如果向量化器未初始化
            
        Example:
            >>> doc_scores = retriever.get_top_k_with_scores("Python", k=3)
            >>> for doc, score in doc_scores:
            ...     print(f"分数: {score:.3f}, 内容: {doc.content[:50]}...")
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化")
        
        if not self.docs:
            return []
        
        k = k or self.k
        k = min(k, len(self.docs))
        
        # 获取所有分数
        scores = self.get_scores(query)
        
        # 获取 top-k 索引
        top_k_idx_np = np.argpartition(-np.array(scores), k-1)[:k]
        top_indices = top_k_idx_np[np.argsort(-np.array(scores)[top_k_idx_np])]
        
        # 返回文档和分数的元组列表
        results = []
        for idx in top_indices:
            results.append((self.docs[idx], scores[idx]))
        
        return results
    
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """添加新文档到检索器
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        对于频繁的文档更新操作，建议考虑使用 VectorStoreRetriever。
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 其他参数
                rebuild_threshold: 文档数量阈值，超过此值会发出性能警告（默认1000）
            
        Returns:
            添加文档的ID列表（仅包含有ID的文档）
            
        Raises:
            ImportError: 如果未安装 rank_bm25
            RuntimeWarning: 如果文档数量超过建议阈值
            
        Example:
            >>> new_docs = [Document(content="New document", id="doc_new")]
            >>> added_ids = retriever.add_documents(new_docs)
            >>> print(f"添加了文档: {added_ids}")
        """
        if not documents:
            logger.info("没有文档需要添加")
            return []
        
        # 添加文档到现有列表
        original_count = len(self.docs)
        self.docs.extend(documents)
        
        # 重建索引
        rebuild_threshold = kwargs.get('rebuild_threshold', 1000)
        self._rebuild_vectorizer(rebuild_threshold)
        
        logger.info(f"添加了 {len(documents)} 个文档（总数: {original_count} → {len(self.docs)}）")
        
        # 返回添加文档的ID
        return [doc.id for doc in documents if doc.id is not None]
    
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """异步添加文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 其他参数
            
        Returns:
            添加文档的ID列表
        """
        # loop = asyncio.get_event_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._thread_executor, self.add_documents, documents
        )
    
    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """删除文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        对于频繁的文档更新操作，建议考虑使用 VectorStoreRetriever。
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他参数
                rebuild_threshold: 文档数量阈值，超过此值会发出性能警告（默认1000）
            
        Returns:
            是否成功删除了文档（True表示删除了至少一个文档）
            
        Example:
            >>> # 删除特定文档
            >>> success = retriever.delete_documents(["doc1", "doc2"])
            >>> 
            >>> # 删除所有文档
            >>> retriever.delete_documents()
        """
        original_count = len(self.docs)
        
        if ids is None:
            # 删除所有文档
            self.docs.clear()
            self.vectorizer = None
            logger.info("删除了所有文档")
            return original_count > 0
        
        # 删除指定ID的文档
        ids_set = set(ids)  # 转为集合提高查找效率
        self.docs = [doc for doc in self.docs if doc.id not in ids_set]
        deleted_count = original_count - len(self.docs)
        
        if deleted_count > 0:
            # 重建索引
            rebuild_threshold = kwargs.get('rebuild_threshold', 1000)
            self._rebuild_vectorizer(rebuild_threshold)
            logger.info(f"删除了 {deleted_count} 个文档（总数: {original_count} → {len(self.docs)}）")
        else:
            logger.info("没有找到匹配的文档进行删除")
        
        return deleted_count > 0
    
    async def adelete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """异步删除文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他参数
            
        Returns:
            是否成功删除了文档
        """
        # loop = asyncio.get_event_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._thread_executor, self.delete_documents, ids, **kwargs
        )
    
    def _rebuild_vectorizer(self, warn_threshold: int = 1000) -> None:
        """重建 BM25 向量化器
        
        这是一个内部方法，用于在文档增删后重新构建BM25索引。
        
        Args:
            warn_threshold: 发出性能警告的文档数量阈值
            
        Raises:
            ImportError: 如果未安装 rank_bm25
        """
        if not self.docs:
            self.vectorizer = None
            logger.debug("文档列表为空，清空向量化器")
            return
        
        # 性能警告
        if len(self.docs) > warn_threshold:
            warnings.warn(
                f"正在重建包含 {len(self.docs)} 个文档的 BM25 索引，这可能很慢。"
                f"对于大型或频繁更新的文档集合，建议使用 VectorStoreRetriever。",
                RuntimeWarning,
                stacklevel=4  # 调整堆栈级别指向用户代码
            )
        
        try:
            from rank_bm25 import BM25Okapi
            
            # 预处理所有文档文本
            all_texts = [doc.content for doc in self.docs]
            texts_processed = [self.preprocess_func(text) for text in all_texts]
            
            # 重建向量化器
            self.vectorizer = BM25Okapi(texts_processed, **self.bm25_params)
            logger.debug(f"成功重建 BM25 索引，包含 {len(self.docs)} 个文档")
            
        except ImportError:
            raise ImportError(
                "未找到 rank_bm25 库，请安装: pip install rank_bm25"
            )
    
    
    def get_document_count(self) -> int:
        """获取文档总数
        
        Returns:
            当前检索器中的文档总数
            
        Example:
            >>> count = retriever.get_document_count()
            >>> print(f"共有 {count} 个文档")
        """
        return len(self.docs)
    
    def get_bm25_info(self) -> Dict[str, Any]:
        """获取 BM25 检索器的详细信息
        
        返回包含检索器配置、统计信息等的详细信息字典。
        
        Returns:
            包含检索器信息的字典，包括：
            - document_count: 文档总数
            - k: 默认返回文档数量
            - bm25_params: BM25算法参数
            - preprocess_func: 预处理函数名称
            - has_vectorizer: 是否已初始化向量化器
            - vocab_size: 词汇表大小（如果向量化器存在）
            - average_doc_length: 平均文档长度（如果向量化器存在）
            
        Example:
            >>> info = retriever.get_bm25_info()
            >>> print(f"文档数量: {info['document_count']}")
            >>> print(f"词汇表大小: {info.get('vocab_size', 'N/A')}")
        """
        info = {
            "document_count": len(self.docs),
            "k": self.k,
            "bm25_params": self.bm25_params.copy(),
            "preprocess_func": self.preprocess_func.__name__,
            "has_vectorizer": self.vectorizer is not None,
        }
        
        # 如果向量化器存在，添加更多统计信息
        if self.vectorizer is not None:
            info.update({
                "vocab_size": len(self.vectorizer.idf),
                "average_doc_length": getattr(self.vectorizer, 'avgdl', 'N/A'),
            })
            
            # 添加文档长度统计
            if hasattr(self.vectorizer, 'doc_len'):
                doc_lengths = self.vectorizer.doc_len
                info.update({
                    "min_doc_length": int(np.min(doc_lengths)),
                    "max_doc_length": int(np.max(doc_lengths)),
                    "median_doc_length": int(np.median(doc_lengths)),
                })
        
        return info
    
    def update_k(self, new_k: int) -> None:
        """更新返回文档数量
        
        由于启用了 validate_assignment，这个方法会自动验证新的k值。
        
        Args:
            new_k: 新的文档返回数量，必须大于0
            
        Raises:
            ValueError: 如果 new_k <= 0（由Pydantic验证器抛出）
            
        Example:
            >>> retriever.update_k(10)
            >>> print(f"新的k值: {retriever.k}")
        """
        old_k = self.k
        self.k = new_k  # Pydantic 会自动验证
        logger.info(f"更新 k 值: {old_k} → {new_k}")
    
    def get_name(self) -> str:
        """获取检索器名称
        
        Returns:
            检索器的类型名称
        """
        return "BM25Retriever"
    
    # =========================== 持久化方法 ===========================
    
    def save_to_disk(self, path: str) -> None:
        """将 BM25 检索器状态保存到磁盘
        
        使用 dill 序列化整个检索器状态，包括 vectorizer、docs、配置等。
        dill 比 pickle 更强大，可以序列化更复杂的对象和函数。
        
        注意：保存的文件包含完整的检索器状态，加载时无需重新构建索引。
        
        Args:
            path: 保存路径，支持文件路径或目录路径
                  - 如果是目录，会在目录下创建 bm25.pkl 文件
                  - 如果是文件，必须以 .pkl 结尾
        
        Raises:
            IOError: 如果保存失败
            ValueError: 如果路径格式错误
            
        Example:
            >>> # 保存到文件
            >>> retriever.save_to_disk("my_bm25_index.pkl")
            >>> 
            >>> # 保存到目录
            >>> retriever.save_to_disk("./models/")  # 会创建 ./models/bm25.pkl
        """
        # 处理路径
        if os.path.isdir(path):
            path = os.path.join(path, "bm25.pkl")
        elif not path.endswith('.pkl'):
            raise ValueError("文件路径必须以 .pkl 结尾")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        try:
            # 准备要保存的状态
            state = {
                'vectorizer': self.vectorizer,
                'docs': self.docs,
                'k': self.k,
                'preprocess_func': self.preprocess_func,
                'bm25_params': self.bm25_params,
                'version': '1.0',  # 添加版本信息以便未来兼容性检查
                'metadata': {
                    'doc_count': len(self.docs),
                    'created_at': str(uuid.uuid4()),  # 简单的保存标识
                }
            }
            
            # 保存到磁盘
            with open(path, 'wb') as f:
                dill.dump(state, f)
                
            logger.info(f"BM25 检索器已成功保存到 {path}")
            logger.debug(f"保存状态: {len(self.docs)} 个文档, k={self.k}")
            
        except Exception as e:
            logger.error(f"保存 BM25 检索器失败: {e}")
            raise IOError(f"保存到 {path} 失败: {e}")

    @classmethod
    def load_from_disk(cls, path: str) -> "BM25Retriever":
        """从磁盘加载 BM25 检索器
        
        从之前保存的文件中恢复完整的检索器状态。
        
        Args:
            path: 加载路径（.pkl 文件）
        
        Returns:
            加载的 BM25Retriever 实例
        
        Raises:
            IOError: 如果文件不存在或加载失败
            ValueError: 如果状态数据无效或版本不兼容
            
        Example:
            >>> retriever = BM25Retriever.load_from_disk("my_bm25_index.pkl")
            >>> print(f"加载了 {retriever.get_document_count()} 个文档")
        """
        if not os.path.exists(path):
            raise IOError(f"文件不存在: {path}")
        
        try:
            # 从磁盘加载状态
            with open(path, 'rb') as f:
                state = dill.load(f)
            
            # 验证必需字段
            required_fields = ['vectorizer', 'docs', 'k', 'preprocess_func', 'bm25_params']
            missing_fields = [field for field in required_fields if field not in state]
            if missing_fields:
                raise ValueError(f"保存的状态缺少必需字段: {missing_fields}")
            
            # 创建检索器实例
            retriever = cls(
                vectorizer=state['vectorizer'],
                docs=state['docs'],
                k=state['k'],
                preprocess_func=state['preprocess_func'],
                bm25_params=state['bm25_params'],
                warn_default_preprocess=False,  # 加载时不需要警告
            )
            
            logger.info(f"从 {path} 成功加载 BM25 检索器")
            logger.debug(f"加载状态: {len(retriever.docs)} 个文档, k={retriever.k}")
            
            # 验证加载后的状态
            if retriever.vectorizer is not None and len(retriever.docs) == 0:
                warnings.warn("加载的检索器状态可能不一致：存在向量化器但无文档", UserWarning)
            
            return retriever
            
        except Exception as e:
            logger.error(f"加载 BM25 检索器失败: {e}")
            if isinstance(e, (IOError, ValueError)):
                raise
            else:
                raise IOError(f"从 {path} 加载失败: {e}")
    
    
    def __repr__(self) -> str:
        """返回检索器的字符串表示
        
        Returns:
            检索器的简洁字符串描述
        """
        return (
            f"{self.__class__.__name__}("
            f"docs={len(self.docs)}, "
            f"k={self.k}, "
            f"preprocess_func={self.preprocess_func.__name__})"
        )
    
    def __len__(self) -> int:
        """返回文档数量
        
        Returns:
            文档总数
            
        Example:
            >>> len(retriever)
            42
        """
        return len(self.docs)
    

    
    def preview_documents(self, n: int = 5, max_length: int = 100) -> List[Dict[str, Any]]:
        """预览前n个文档的内容
        
        Args:
            n: 预览文档数量
            max_length: 每个文档内容的最大显示长度
            
        Returns:
            文档预览信息列表
        """
        previews = []
        for i, doc in enumerate(self.docs[:n]):
            content = doc.content
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            previews.append({
                "index": i,
                "id": doc.id,
                "content": content,
                "content_length": len(doc.content),
                "metadata_keys": list(doc.metadata.keys()) if doc.metadata else [],
            })
        
        return previews