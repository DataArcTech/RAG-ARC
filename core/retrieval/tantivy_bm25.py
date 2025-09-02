import os
import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
import threading
from enum import Enum

import jieba
from pydantic import field_validator

try:
    from .base import BaseRetriever
    from ..utils.data_model import Document
    print(1)
except ImportError:
    try:
        from core.retrieval.base import BaseRetriever
        from core.utils.data_model import Document
        print(2)
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from core.retrieval.base import BaseRetriever
        from core.utils.data_model import Document

try:
    import tantivy
    from tantivy import (
        Index, SchemaBuilder, Document as TantivyDocument,
        Tokenizer, TextAnalyzerBuilder, Filter, Query
    )
except ImportError:
    raise ImportError(
        "The 'tantivy-py' library was not found. Please install it using: pip install tantivy"
    )

logger = logging.getLogger(__name__)

jieba.enable_parallel(4)  # 启用并行分词，使用4个进程


class SearchMode(Enum):
    """搜索模式枚举"""
    ALL = "all"  # 所有词都必须匹配
    ANY = "any"  # 至少一个词匹配
    PHRASE = "phrase"  # 短语匹配


class TokenCache:
    """线程安全的分词缓存"""
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get(self, text: str) -> Optional[List[str]]:
        with self._lock:
            result = self._cache.get(text)
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
            return result
    
    def put(self, text: str, tokens: List[str]):
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 简单的 LRU: 删除最早的 10% 条目
                remove_count = self.max_size // 10
                for key in list(self._cache.keys())[:remove_count]:
                    del self._cache[key]
            self._cache[text] = tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0
            }


# 全局分词缓存
_token_cache = TokenCache()


def cached_preprocessing_func(text: str, cache: Optional[TokenCache] = None) -> List[str]:
    """带缓存的默认文本预处理函数"""
    if cache is None:
        cache = _token_cache
    
    # 尝试从缓存获取
    cached_tokens = cache.get(text)
    if cached_tokens is not None:
        return cached_tokens
    
    # 执行分词并缓存结果
    tokens = list(jieba.cut(text))
    cache.put(text, tokens)
    return tokens


class TantivyBM25Retriever(BaseRetriever):
    """
    高性能的基于 Tantivy 搜索引擎的 BM25 检索器。
    
    性能优化特性：
    - 批量文档处理
    - 分词结果缓存
    - 异步操作支持
    - 连接池和 searcher 复用
    - 复杂查询支持（元数据过滤、布尔查询等）
    """
    
    def __init__(
        self, 
        docs: Optional[List[Document]] = None, 
        index_path: str = None, 
        k: int = 5,
        preprocess_func: Optional[Callable[[str], List[str]]] = None,
        bm25_k1: float = 1.2, 
        bm25_b: float = 0.75,
        stopwords: Optional[List[str]] = None, 
        writer_heap_size: int = 256 * 1024 * 1024,  # 增加默认堆大小
        num_threads: int = 0,  # 0 表示自动检测
        batch_size: int = 100,  # 批量处理大小
        enable_cache: bool = False,  # 启用分词缓存
        cache_size: int = 10000,  # 缓存大小
        **kwargs
    ):
        super().__init__(**kwargs)
        self.index_path = index_path or f"./tantivy_index_{uuid.uuid4().hex[:8]}"
        self.k = k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.stopwords = stopwords if stopwords is not None else [
            "的", "是", "在", "和", "与", "或", "了", "等", "就", "也", 
            "一", "个", "有", "这", "那", "不", "但", "对", "为", "很"
        ]
        self._index = None
        self._schema = None
        self._writer_heap_size = writer_heap_size
        self._num_threads = num_threads
        self._batch_size = batch_size
        self._searcher = None  # 缓存 searcher 对象
        self._searcher_lock = threading.RLock()
        
        # 设置分词函数和缓存
        if enable_cache:
            self._token_cache = TokenCache(max_size=cache_size)
            self.preprocess_func = lambda text: cached_preprocessing_func(text, self._token_cache)
        else:
            self.preprocess_func = preprocess_func or (lambda text: list(jieba.cut(text)))
        
        # 线程池用于异步操作
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        self._initialize_tantivy()
        
        if docs:
            self.add_documents(docs)

    def _initialize_tantivy(self):
        """初始化 Tantivy 组件，包括schema、分词器和索引"""
        if self._index is not None:
            return

        # 1. 定义schema - 优化字段配置
        schema_builder = SchemaBuilder()
        
        # ID 字段：使用 FAST 字段加速查找
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        
        # 内容字段：原始内容存储
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        
        # 分词字段：用于搜索，不存储原始内容以节省空间
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=False)
        
        # 元数据字段 - 为常用元数据添加独立字段以支持高效过滤
        schema_builder.add_json_field("metadata", stored=True)
        
        # 添加常用元数据字段用于过滤
        schema_builder.add_text_field("meta_category", tokenizer_name="raw", stored=False, fast=True)
        schema_builder.add_text_field("meta_source", tokenizer_name="raw", stored=False, fast=True)
        schema_builder.add_text_field("meta_tags", tokenizer_name="custom", stored=False)
        
        self._schema = schema_builder.build()

        # 2. 创建或打开索引
        if os.path.exists(self.index_path) and any(os.scandir(self.index_path)):
            logger.info(f"Loading existing index from: {self.index_path}")
            self._index = Index.open(self.index_path)
        else:
            logger.info(f"Creating new index at: {self.index_path}")
            os.makedirs(self.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.index_path)
        
        # 3. 注册自定义分词器
        custom_analyzer = (
            TextAnalyzerBuilder(Tokenizer.whitespace())
            .filter(Filter.lowercase())
            .filter(Filter.custom_stopword(self.stopwords))
            .build()
        )
        self._index.register_tokenizer("custom", custom_analyzer)
        
        raw_analyzer = TextAnalyzerBuilder(Tokenizer.raw()).build()
        self._index.register_tokenizer("raw", raw_analyzer)
        
        logger.info("Tantivy initialized successfully.")

    def _get_searcher(self):
        """获取或创建缓存的 searcher 对象"""
        with self._searcher_lock:
            if self._searcher is None:
                self._index.reload()
                self._searcher = self._index.searcher()
            return self._searcher

    def _invalidate_searcher(self):
        """使缓存的 searcher 失效"""
        with self._searcher_lock:
            self._searcher = None

    @classmethod
    def from_documents(
        cls, documents: Iterable[Document], **kwargs: Any
    ) -> "TantivyBM25Retriever":
        """从文档创建检索器"""
        docs_list = list(documents)
        if not docs_list:
            raise ValueError("The 'documents' list cannot be empty.")
        return cls(docs=docs_list, **kwargs)

    @classmethod
    def load(cls, index_path: str, **kwargs: Any) -> "TantivyBM25Retriever":
        """加载现有索引"""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
        
        retriever = cls(index_path=index_path, docs=None, **kwargs)
        try:
            retriever._index.reload()
            logger.info(f"Successfully loaded index with {retriever.get_document_count()} documents.")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
        return retriever

    def _build_metadata_query(self, metadata_filters: Dict[str, Any]) -> Optional[Query]:
        """构建元数据过滤查询"""
        if not metadata_filters:
            return None
        
        queries = []
        query_parser = self._index.query_parser()
        
        for key, value in metadata_filters.items():
            if key == "category" and value:
                queries.append(query_parser.parse_query(f'meta_category:"{value}"'))
            elif key == "source" and value:
                queries.append(query_parser.parse_query(f'meta_source:"{value}"'))
            elif key == "tags" and value:
                # 标签匹配
                if isinstance(value, list):
                    tag_query = " OR ".join([f'meta_tags:"{tag}"' for tag in value])
                else:
                    tag_query = f'meta_tags:"{value}"'
                queries.append(query_parser.parse_query(tag_query))
        
        if len(queries) == 1:
            return queries[0]
        elif len(queries) > 1:
            # 组合多个查询条件（AND 逻辑）
            return Query.boolean_query([(Query.Occur.Must, q) for q in queries])
        
        return None

    def _execute_search(
        self, 
        query: Query, 
        k: int,
        metadata_query: Optional[Query] = None
    ) -> List[Tuple[Document, float]]:
        """执行搜索查询"""
        try:
            searcher = self._get_searcher()
            
            if searcher.num_docs == 0:
                logger.warning("Search executed on an empty index.")
                return []
            
            # 如果有元数据过滤，组合查询
            if metadata_query:
                final_query = Query.boolean_query([
                    (Query.Occur.Must, query),
                    (Query.Occur.Must, metadata_query)
                ])
            else:
                final_query = query
            
            # 执行搜索
            search_result = searcher.search(final_query, k)
            top_docs = search_result.hits
            
            results = []
            for score, doc_address in top_docs:
                tantivy_doc = searcher.doc(doc_address)
                
                # 获取字段值
                doc_id = tantivy_doc.get_first("id") or ""
                content = tantivy_doc.get_first("content") or ""
                metadata = tantivy_doc.get_first("metadata") or {}
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata
                )
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def search(
        self, 
        query: str, 
        k: Optional[int] = None,
        mode: SearchMode = SearchMode.ANY,
        metadata_filters: Optional[Dict[str, Any]] = None,
        boost_fields: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        高级搜索接口
        
        Args:
            query: 搜索查询字符串
            k: 返回结果数量
            mode: 搜索模式 (ALL, ANY, PHRASE)
            metadata_filters: 元数据过滤条件
            boost_fields: 字段权重设置
            
        Returns:
            文档和分数的列表
        """
        k = k or self.k
        
        # 分词处理
        query_tokens = self.preprocess_func(query)
        
        # 根据搜索模式构建查询
        if mode == SearchMode.ALL:
            # 所有词都必须匹配
            query_str = " AND ".join(query_tokens)
        elif mode == SearchMode.PHRASE:
            # 短语匹配
            query_str = f'"{" ".join(query_tokens)}"'
        else:  # SearchMode.ANY
            # 至少一个词匹配
            query_str = " ".join(query_tokens)
        
        # 解析查询
        parsed_query = self._index.parse_query(query_str, ["content_tokens"])
        
        # 构建元数据查询
        metadata_query = self._build_metadata_query(metadata_filters) if metadata_filters else None
        
        # 执行搜索
        return self._execute_search(parsed_query, k, metadata_query)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """获取与查询最相关的前 k 个文档"""
        k = kwargs.get('k', self.k)
        metadata_filters = kwargs.get('metadata_filters')
        mode = kwargs.get('mode', SearchMode.ANY)
        
        # 使用高级搜索接口
        search_results = self.search(
            query=query,
            k=k,
            mode=mode,
            metadata_filters=metadata_filters
        )
        
        return [doc for doc, score in search_results]

    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """异步获取相关文档"""
        loop = asyncio.get_running_loop()  # 使用 get_running_loop
        return await loop.run_in_executor(
            self._executor, 
            self._get_relevant_documents, 
            query, 
            kwargs
        )

    def search_by_similarity(
        self, 
        reference_doc_id: str, 
        k: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        基于相似文档的搜索
        
        Args:
            reference_doc_id: 参考文档ID
            k: 返回结果数量
            metadata_filters: 元数据过滤条件
            
        Returns:
            相似文档列表
        """
        k = k or self.k
        
        # 获取参考文档
        searcher = self._get_searcher()
        query = self._index.parse_query(f'id:"{reference_doc_id}"', ["id"])
        results = searcher.search(query, 1)
        
        if not results.hits:
            logger.warning(f"Reference document {reference_doc_id} not found")
            return []
        
        _, doc_address = results.hits[0]
        ref_doc = searcher.doc(doc_address)
        content = ref_doc.get_first("content") or ""
        
        # 使用参考文档的内容进行搜索
        return self.search(
            query=content[:500],  # 限制长度
            k=k + 1,  # 多获取一个以排除自己
            metadata_filters=metadata_filters
        )[1:]  # 排除第一个（自己）

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """批量添加文档到索引"""
        if not documents:
            return []
        
        # 使失效缓存的 searcher
        self._invalidate_searcher()
        
        # 创建写入器
        writer = self._index.writer(
            heap_size=self._writer_heap_size,
            num_threads=self._num_threads or 0
        )
        
        added_ids = []
        batch_docs = []
        
        try:
            for doc in documents:
                doc_id = str(doc.id) if doc.id is not None else str(uuid.uuid4())
                content = doc.content or ""
                metadata = doc.metadata or {}
                
                # 预处理分词
                content_tokens = " ".join(self.preprocess_func(content))
                
                # 创建 Tantivy 文档
                tantivy_doc = TantivyDocument()
                tantivy_doc.add_text("id", doc_id)
                tantivy_doc.add_text("content", content)
                tantivy_doc.add_text("content_tokens", content_tokens)
                tantivy_doc.add_json("metadata", metadata)
                
                # 添加元数据索引字段
                if "category" in metadata:
                    tantivy_doc.add_text("meta_category", str(metadata["category"]))
                if "source" in metadata:
                    tantivy_doc.add_text("meta_source", str(metadata["source"]))
                if "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list):
                        tags_str = " ".join(tags)
                    else:
                        tags_str = str(tags)
                    tantivy_doc.add_text("meta_tags", tags_str)
                
                batch_docs.append(tantivy_doc)
                added_ids.append(doc_id)
                
                # 批量写入
                if len(batch_docs) >= self._batch_size:
                    for td in batch_docs:
                        writer.add_document(td)
                    batch_docs = []
            
            # 写入剩余文档
            for td in batch_docs:
                writer.add_document(td)
            
            # 提交更改
            writer.commit()
            logger.info(f"Successfully added {len(documents)} documents to the index.")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            writer.rollback()
            raise
        
        return added_ids

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """异步添加文档"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.add_documents, 
            documents, 
            kwargs
        )

    def delete_documents(self, ids: List[str], **kwargs: Any) -> bool:
        """批量删除文档"""
        if not ids:
            return False
        
        self._invalidate_searcher()
        
        writer = self._index.writer(self._writer_heap_size)
        deleted_count = 0
        
        try:
            for doc_id in ids:
                deleted_count += writer.delete_documents(field_name="id", field_value=str(doc_id))
            
            writer.commit()
            logger.info(f"Deleted {deleted_count} documents.")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            writer.rollback()
            raise

    async def adelete_documents(self, ids: List[str], **kwargs: Any) -> bool:
        """异步删除文档"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.delete_documents, 
            ids, 
            kwargs
        )

    def update_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """更新文档（删除后重新添加）"""
        ids_to_delete = [str(doc.id) for doc in documents if doc.id is not None]
        
        if ids_to_delete:
            self.delete_documents(ids_to_delete)
        
        return self.add_documents(documents)

    def optimize_index(self):
        """优化索引以提高查询性能"""
        writer = self._index.writer(self._writer_heap_size)
        try:
            writer.wait_merging_threads()  # 等待合并线程完成
            writer.commit()
            logger.info("Index optimization completed.")
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            writer.rollback()
            raise
        finally:
            self._invalidate_searcher()

    def get_document_count(self) -> int:
        """获取索引中的文档总数"""
        searcher = self._get_searcher()
        return searcher.num_docs

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """根据ID获取单个文档"""
        searcher = self._get_searcher()
        query = self._index.parse_query(f'id:"{doc_id}"', ["id"])
        results = searcher.search(query, 1)
        
        if results.hits:
            _, doc_address = results.hits[0]
            tantivy_doc = searcher.doc(doc_address)
            return Document(
                id=tantivy_doc.get_first("id") or "",
                content=tantivy_doc.get_first("content") or "",
                metadata=tantivy_doc.get_first("metadata") or {}
            )
        return None

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        searcher = self._get_searcher()
        cache_stats = self._token_cache.get_stats() if hasattr(self, '_token_cache') else {}
        
        return {
            "num_docs": searcher.num_docs,
            "index_path": self.index_path,
            "cache_stats": cache_stats,
            "batch_size": self._batch_size,
            "writer_heap_size_mb": self._writer_heap_size / (1024 * 1024),
        }

    def clear_cache(self):
        """清空分词缓存"""
        if hasattr(self, '_token_cache'):
            with self._token_cache._lock:
                self._token_cache._cache.clear()
                self._token_cache.hits = 0
                self._token_cache.misses = 0
            logger.info("Token cache cleared.")

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"docs={self.get_document_count()}, "
            f"k={self.k}, "
            f"index_path='{self.index_path}')"
        )

    @field_validator('k')
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"k must be greater than 0, but got {v}")
        return v
    
    @field_validator('bm25_k1')
    @classmethod
    def validate_bm25_k1(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bm25_k1 must be greater than 0, but got {v}")
        return v
    
    @field_validator('bm25_b')
    @classmethod
    def validate_bm25_b(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"bm25_b must be between 0 and 1, but got {v}")
        return v