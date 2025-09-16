import os
import logging
import uuid
import time
import json
import queue
import threading
import psutil
import multiprocessing
import gc
from typing import Any, Callable, Dict, List, Optional, Literal, Union
from concurrent.futures import ProcessPoolExecutor
from pydantic import Field, field_validator
import jieba

from core.utils.data_model import Document
from encapsulation.database.vector_db.base import BaseIndex, BaseIndexConfig
from encapsulation.database.utils.TokenizerManager import TokenizerManager

try:
    from tantivy import (
        Index, SchemaBuilder, Document as TantivyDocument,
        Tokenizer, TextAnalyzerBuilder, Filter
    )
except ImportError:
    raise ImportError("Please install tantivy: pip install tantivy")

logger = logging.getLogger(__name__)

# Fields to exclude from dynamic schema creation in metadata
EXCLUDED_METADATA_FIELDS = {
    "score",           # Relevance scores should not be indexed as searchable fields
    "_score",          # Alternative score field name
    "rank",            # Ranking information
    "_rank",           # Alternative rank field name
    "distance",        # Distance/similarity measures
    "_distance",       # Alternative distance field name
    "similarity",      # Similarity scores
    "_similarity",     # Alternative similarity field name
}


class BM25IndexBuilderConfig(BaseIndexConfig):
    """BM25索引构建器配置"""
    type: Literal["bm25_indexer"] = "bm25_indexer"

    # 核心配置
    index_path: str = Field(description="索引存储路径")
    bm25_k1: float = Field(default=1.2, description="BM25 k1参数")
    bm25_b: float = Field(default=0.75, description="BM25 b参数")

    # 可选配置
    preprocess_func_name: Optional[str] = Field(default=None, description="预处理函数名")
    stopwords_file: Optional[str] = Field(default=None, description="停用词文件路径")
    writer_heap_size: Optional[int] = Field(default=None, description="写入器堆大小")
    batch_size: int = Field(default=50, description="批处理大小")
    tokenize_batch_size: int = Field(default=200, description="分词批处理大小")
    max_workers: Optional[int] = Field(default=None, description="最大工作进程数")
    progress_interval: int = Field(default=500, description="进度报告间隔")
    enable_gc: bool = Field(default=True, description="是否启用垃圾回收")
    queue_maxsize: int = Field(default=1000, description="队列最大大小")

    # 运行时字段
    preprocess_func: Optional[Callable[[str], List[str]]] = Field(default=None, exclude=True)
    progress_callback: Optional[Callable] = Field(default=None, exclude=True)
    
    @field_validator("bm25_k1")
    @classmethod
    def validate_bm25_k1(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bm25_k1 must be greater than 0, but got {v}")
        return v
    
    @field_validator("bm25_b")
    @classmethod
    def validate_bm25_b(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError(f"bm25_b must be between 0 and 1, but got {v}")
        return v
    

    def build(self) -> "BM25IndexBuilder":
        """Build the BM25IndexBuilder instance"""
        return BM25IndexBuilder(config=self)


def init_jieba_worker():
    """Initialize jieba in the worker process to reduce initialization overhead"""
    return jieba


class BM25IndexBuilder(BaseIndex):
    """
    基于Tantivy的BM25索引构建器

    主要功能：
    - 文档索引构建和管理
    - 支持中文分词和多语言
    - 流式处理和批量操作
    - 多进程优化
    """

    # Runtime instance variables (initialized when needed)
    _index: Optional[Index] = None
    _schema = None
    _writer_heap_size: int = 0
    _tokenizers_registered: bool = False
    _executor: Optional[ProcessPoolExecutor] = None
    _executor_closed: bool = False
    _queue: queue.Queue = None
    _writer_thread: Optional[threading.Thread] = None
    _stop_event: threading.Event = None
    _tokenizer_manager: TokenizerManager = None

    @property
    def tokenizer_manager(self) -> TokenizerManager:
        """Lazy-initialized tokenizer manager"""
        if self._tokenizer_manager is None:
            self._tokenizer_manager = TokenizerManager(
                custom_preprocess_func=self.config.preprocess_func,
                custom_stopwords_file=self.config.stopwords_file
            )
        return self._tokenizer_manager
    
    @property
    def writer_heap_size(self) -> int:
        """Lazy-calculated writer heap size"""
        if self._writer_heap_size == 0:
            if self.config.writer_heap_size is None:
                total_mem = psutil.virtual_memory().total
                self._writer_heap_size = min(int(total_mem * 0.2), 1024 * 1024 * 1024)
            else:
                self._writer_heap_size = self.config.writer_heap_size
        return self._writer_heap_size
    
    @property
    def processing_queue(self) -> queue.Queue:
        """Lazy-initialized processing queue"""
        if self._queue is None:
            self._queue = queue.Queue(maxsize=self.config.queue_maxsize)
        return self._queue
    
    @property
    def stop_event(self) -> threading.Event:
        """Lazy-initialized stop event"""
        if self._stop_event is None:
            self._stop_event = threading.Event()
        return self._stop_event
    
    @property
    def index(self) -> Optional[Index]:
        """Get the Tantivy index instance"""
        return self._index
    
    
    def _set_tokenizer(self, documents: List[Document]):
        """Set tokenizer (proxied to TokenizerManager)"""
        self.tokenizer_manager.set_tokenizer_by_detection(documents)

    def _ensure_index_loaded(self) -> None:
        """Ensure index is loaded before operations
        
        Raises:
            RuntimeError: If index is not loaded
        """
        if self._index is None:
            raise RuntimeError(
                "Index is not loaded. Call load_local() to load existing index "
                "or from_documents() to create new index."
            )

    def _tokenize_batch_sequential(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts sequentially (single process)
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized texts
        """
        return self.tokenizer_manager.batch_tokenize(texts)

    def _tokenize_batch_parallel(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts in parallel (multiprocessing)
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized texts
        """
        executor = self._get_executor()
        if not executor or len(texts) <= self.config.tokenize_batch_size:
            return self._tokenize_batch_sequential(texts)

        # Split texts into batches
        batches = [texts[i:i + self.config.tokenize_batch_size] for i in range(0, len(texts), self.config.tokenize_batch_size)]
        results = []
        
        # Create serializable tokenization tasks
        futures = []
        for batch in batches:
            # Since TokenizerManager may contain non-serializable custom functions,
            # we directly use the current instance's tokenization method
            future = executor.submit(self._tokenize_batch_sequential, batch)
            futures.append(future)
            
        # Collect results
        for future in futures:
            try:
                results.extend(future.result(timeout=60))
            except Exception as e:
                logger.warning(f"Parallel tokenization failed, fallback to sequential: {e}")
                return self._tokenize_batch_sequential(texts)
        return results

    def _extract_string_fields_from_documents(self, documents: List[Document]) -> set[str]:
        """Extract string fields from document metadata for dynamic schema creation
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Set of field names that contain string values
        """
        string_fields = set()
        
        for doc in documents:
            if not doc.metadata:
                continue
                
            for key, value in doc.metadata.items():
                # Only include string fields for filtering, exclude system/score fields
                if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                    string_fields.add(key)
        
        logger.debug(f"Extracted dynamic string fields: {string_fields}")
        return string_fields

    def _initialize_index(self, documents: Optional[List[Document]] = None) -> None:
        """Initialize the Tantivy index with dynamic schema based on document metadata
        
        Args:
            documents: Optional list of documents to analyze for dynamic field creation
        """
        if self._index is not None:
            return
            
        # Build schema with core fields
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=True)
        schema_builder.add_json_field("metadata", stored=True)
        
        # Add dynamic fields based on document metadata
        if documents:
            dynamic_fields = self._extract_string_fields_from_documents(documents)
            for field_name in dynamic_fields:
                schema_builder.add_text_field(field_name, tokenizer_name="raw", stored=False, fast=True)
                logger.debug(f"Added dynamic field: {field_name}")
        
        self._schema = schema_builder.build()

        # Load existing index or create new one
        if os.path.exists(self.config.index_path) and any(os.scandir(self.config.index_path)):
            logger.info(f"Loading existing index from: {self.config.index_path}")
            self._index = Index.open(self.config.index_path)
        else:
            logger.info(f"Creating new index at: {self.config.index_path}")
            os.makedirs(self.config.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.config.index_path)
        
        # Always register tokenizers when index is loaded/created
        if not self._tokenizers_registered:
            self._register_tokenizers()

        logger.info("Tantivy index initialized successfully")

    def _register_tokenizers(self) -> None:
        """Register tokenizers to avoid duplicate registration"""
        if self._tokenizers_registered or self._index is None:
            return
            
        try:
            custom_analyzer = (
                TextAnalyzerBuilder(Tokenizer.whitespace())
                .filter(Filter.lowercase())
                .filter(Filter.custom_stopword(self.tokenizer_manager.get_stopwords()))
                .build()
            )
            self._index.register_tokenizer("custom", custom_analyzer)
            
            raw_analyzer = TextAnalyzerBuilder(Tokenizer.raw()).build()
            self._index.register_tokenizer("raw", raw_analyzer)
            
            self._tokenizers_registered = True
            logger.debug("Tokenizers registered successfully")
                
        except Exception as e:
            logger.error(f"Failed to register tokenizers: {e}")
            raise

    def _writer_worker(self, writer) -> None:
        """Consumer thread: index writing worker
        
        Args:
            writer: Tantivy index writer
        """
        batch_docs = []
        while not self.stop_event.is_set() or not self.processing_queue.empty():
            try:
                doc = self.processing_queue.get(timeout=1)
                if doc is None:
                    break
                batch_docs.append(doc)
                if len(batch_docs) >= self.config.batch_size:
                    self._batch_write_documents(batch_docs, writer)
                    batch_docs.clear()
                    # Trigger garbage collection if enabled
                    if self.config.enable_gc:
                        gc.collect()
            except queue.Empty:
                continue
        
        if batch_docs:
            self._batch_write_documents(batch_docs, writer)

    def _batch_write_documents(self, docs: List[TantivyDocument], writer) -> None:
        """Write a batch of documents to the index
        
        Args:
            docs: List of Tantivy documents to write
            writer: Tantivy index writer
        """
        try:
            writer.add_documents(docs)
        except AttributeError:
            for d in docs:
                writer.add_document(d)
        except Exception as e:
            logger.error(f"Error writing batch of documents: {e}")
            raise

    def _delete_documents_by_ids(self, doc_ids: List[str]) -> int:
        """Delete documents by their IDs
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents actually deleted
        """
        if not doc_ids:
            return 0
        
        self._ensure_index_loaded()
            
        try:
            # First, check which documents actually exist
            searcher = self._index.searcher()
            existing_ids = []
            
            for doc_id in doc_ids:
                query = self._index.parse_query(f'id:"{doc_id}"', ["id"])
                results = searcher.search(query, 1)
                logger.info(f"Checking document {doc_id}: found {len(results.hits)} hits")
                if results.hits:
                    existing_ids.append(doc_id)
            
            if not existing_ids:
                logger.info("No documents found to delete")
                return 0
            
            # Delete only existing documents
            writer = self._index.writer(heap_size=self._writer_heap_size)
            deleted_count = 0
            
            for doc_id in existing_ids:
                delete_result = writer.delete_documents("id", doc_id)
                logger.info(f"Deleting document {doc_id}: {delete_result} documents deleted")
                deleted_count += delete_result
            
            logger.info(f"Committing deletion of {deleted_count} documents")
            writer.commit()
            logger.info("Reloading index after deletion")
            self._index.reload()
            logger.info(f"Successfully deleted {deleted_count} documents from index (requested: {len(doc_ids)})")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise


    def _build_index(self, documents: List[Document]) -> List[str]:
        """Build index using producer-consumer pattern
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            List of document IDs that were added to the index
            
        Raises:
            RuntimeError: If there's an error during index building
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return []
        
        if self.tokenizer_manager.custom_preprocess_func is None:
            self._set_tokenizer(documents)
        
        # For new indices, reinitialize with dynamic fields based on documents
        index_exists = os.path.exists(self.config.index_path) and any(os.scandir(self.config.index_path)) if os.path.exists(self.config.index_path) else False
        if not index_exists:
            self._index = None  # Reset to force reinitialization with dynamic fields
            self._initialize_index(documents)
        
        if self._index is None:
            raise RuntimeError("Index has not been initialized")
            
        total_docs = len(documents)
        added_ids, processed_count = [], 0
        
        # Ensure any previous writer thread is stopped
        if self._writer_thread and self._writer_thread.is_alive():
            self.stop_event.set()
            try:
                self._writer_thread.join(timeout=5.0)
            except:
                pass
        
        # Clear the queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        writer = self._index.writer(heap_size=self.writer_heap_size)
        
        # Reset stop event for this build operation
        self.stop_event.clear()
        
        self._writer_thread = threading.Thread(target=self._writer_worker, args=(writer,))
        self._writer_thread.start()

        start_time = time.time()
        
        try:
            for doc in documents:
                content_tokens = self.tokenizer_manager.get_current_tokenizer()(doc.content or "")
                doc_id = str(doc.id) if doc.id else str(uuid.uuid4())
                
                tantivy_doc = TantivyDocument()
                tantivy_doc.add_text("id", doc_id)
                tantivy_doc.add_text("content", doc.content or "")
                tantivy_doc.add_text("content_tokens", " ".join(content_tokens))
                
                metadata = doc.metadata or {}
                tantivy_doc.add_json("metadata", metadata)
                
                # Dynamically add all string fields from metadata for filtering
                for key, value in metadata.items():
                    if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                        try:
                            tantivy_doc.add_text(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to add field '{key}' to document: {e}")
                
                self.processing_queue.put(tantivy_doc)

                added_ids.append(doc_id)
                processed_count += 1
                
                if processed_count % self.config.progress_interval == 0:
                    elapsed = time.time() - start_time
                    stats = {
                        "processed": processed_count,
                        "total": total_docs,
                        "elapsed_sec": round(elapsed, 2),
                        "throughput_docs_sec": round(processed_count / elapsed, 2)
                    }
                    logger.info(f"[IndexProgress] {json.dumps(stats, ensure_ascii=False)}")
                    if self.config.progress_callback:
                        self.config.progress_callback(processed_count, total_docs, stats)

            # Final progress callback if not already called at the end
            if processed_count % self.config.progress_interval != 0 and self.config.progress_callback:
                elapsed = time.time() - start_time
                stats = {
                    "processed": processed_count,
                    "total": total_docs,
                    "elapsed_sec": round(elapsed, 2),
                    "throughput_docs_sec": round(processed_count / elapsed, 2)
                }
                logger.info(f"[IndexProgress] Final: {json.dumps(stats, ensure_ascii=False)}")
                self.config.progress_callback(processed_count, total_docs, stats)

            self.processing_queue.put(None)
            self._writer_thread.join()
            writer.commit()
            self._index.reload()
            
            tokenizer_info = self.tokenizer_manager.get_tokenizer_info()
            logger.info(f"Successfully built index with {len(added_ids)} documents using {tokenizer_info} tokenizer")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            try:
                writer.rollback()
            except:
                pass
            raise
        finally:
            self.stop_event.set()
            if self.config.enable_gc:
                gc.collect()
        
        return added_ids

    def add(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Add documents to the existing index (with ID deduplication)

        Args:
            documents: List of Document objects to add
            embeddings: 预计算的嵌入向量（BM25不需要，忽略此参数，保持接口一致）

        Returns:
            List of document IDs that were successfully added to the index
        """
        if not documents:
            logger.warning("No documents provided for adding")
            return []

        # BM25 索引不需要 embeddings，忽略此参数
        if embeddings is not None:
            logger.debug("BM25 index does not use embeddings, ignoring embeddings parameter")

        # 检查重复ID并过滤
        unique_documents = []
        duplicate_ids = []
        existing_ids = set()

        # 获取现有文档ID
        try:
            if self._index is not None:
                # 这里可以实现获取现有ID的逻辑，暂时简化处理
                pass
        except Exception:
            pass

        # 检查文档列表中的重复ID
        seen_ids = set()
        for doc in documents:
            if doc.id in seen_ids:
                duplicate_ids.append(doc.id)
                logger.warning(f"Duplicate document ID found: {doc.id}. Use update_documents() to update existing documents.")
            else:
                seen_ids.add(doc.id)
                unique_documents.append(doc)

        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate document IDs: {duplicate_ids}")

        if not unique_documents:
            logger.warning("No unique documents to add after deduplication")
            return []

        return self._build_index(unique_documents)

    def save_index(self, index_path: str, index_name: str = "index") -> None:
        """保存索引状态（Tantivy自动持久化）"""
        self._ensure_index_loaded()
        logger.info(f"Index saved at: {self.config.index_path}")
        if index_path != self.config.index_path:
            logger.debug(f"Using configured path {self.config.index_path}, ignoring {index_path}")

    def build_index(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Build index from documents (alias for from_documents)

        Args:
            documents: List of Document objects to index
            embeddings: 预计算的嵌入向量（BM25不需要，忽略此参数）
        """
        # BM25 索引不需要 embeddings，忽略此参数
        if embeddings is not None:
            logger.debug("BM25 index does not use embeddings, ignoring embeddings parameter")

        self.from_documents(documents)

    def load_index(self, index_path: Optional[str] = None) -> None:
        """Load index from storage (alias for load_local)

        Args:
            index_path: 索引路径（BM25使用配置中的路径，忽略此参数）

        Raises:
            FileNotFoundError: If index path does not exist
            RuntimeError: If index is already loaded
        """
        if index_path is not None and index_path != self.config.index_path:
            logger.debug(f"BM25 index uses configured path {self.config.index_path}, ignoring provided path {index_path}")
        self.load_local()

    def update(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Update documents in the index by first deleting then adding

        Args:
            documents: List of Document objects to update
            embeddings: 预计算的嵌入向量（BM25不需要，忽略此参数）
        """
        if not documents:
            return

        # BM25 索引不需要 embeddings，忽略此参数
        if embeddings is not None:
            logger.debug("BM25 index does not use embeddings, ignoring embeddings parameter")

        # 先删除现有文档，再添加更新的文档
        doc_ids = [str(doc.id) for doc in documents if doc.id is not None]
        if doc_ids:
            logger.info(f"Update mode: attempting to delete {len(doc_ids)} existing documents")
            deleted_count = self._delete_documents_by_ids(doc_ids)
            logger.info(f"Update mode: successfully deleted {deleted_count} documents")

            # Ensure index is reloaded after deletion for consistency
            if self._index is not None:
                self._index.reload()

        # 添加更新的文档
        self._build_index(documents)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """删除指定ID的文档"""
        if ids is None:
            # 删除所有文档 - 重新创建空索引
            try:
                if self._index is not None:
                    # 关闭当前索引
                    self._index = None
                    self._tokenizers_registered = False

                # 重新创建空索引
                self._initialize_index()
                logger.info("Successfully deleted all documents by recreating index")
                return True
            except Exception as e:
                logger.error(f"Error deleting all documents: {e}")
                return False

        if not ids:
            return True

        unique_doc_ids = list(set(ids))

        try:
            deleted_count = self._delete_documents_by_ids(unique_doc_ids)
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    # TODO Modify to batch retrieval
    def get_by_ids(self, doc_ids: List[str]) -> List[Document]:
        """Retrieve documents by their IDs
        
        Args:
            doc_ids: List of document IDs to retrieve
            
        Returns:
            List of Document objects found
            
        Raises:
            RuntimeError: If index is not initialized
        """
        if not doc_ids:
            return []
            
        self._ensure_index_loaded()
        documents = []

        try:
            searcher = self._index.searcher()
            
            for doc_id in doc_ids:
                query = self._index.parse_query(f'id:"{doc_id}"', ["id"])
                results = searcher.search(query, 1)

                if results.hits:
                    _, doc_address = results.hits[0]
                    tantivy_doc = searcher.doc(doc_address)
                    
                    doc_id_field = tantivy_doc.get_first("id") or ""
                    content_field = tantivy_doc.get_first("content") or ""
                    metadata_field = tantivy_doc.get_first("metadata") or {}

                    if isinstance(metadata_field, str):
                        try:
                            metadata_field = json.loads(metadata_field)
                        except json.JSONDecodeError:
                            metadata_field = {}

                    documents.append(Document(
                        id=doc_id_field,
                        content=content_field,
                        metadata=metadata_field
                    ))
            
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents by IDs {doc_ids}: {e}")
            return []

    def load_local(self) -> "BM25IndexBuilder":
        """Load existing index from local path specified in config
        
        Returns:
            Self (BM25IndexBuilder instance)
            
        Raises:
            FileNotFoundError: If index path does not exist
            RuntimeError: If index is already loaded
            Exception: If there's an error during index loading
        """
        if self._index is not None:
            logger.warning("Index is already loaded")
            return self
            
        if not os.path.exists(self.config.index_path):
            raise FileNotFoundError(f"Index path does not exist: {self.config.index_path}")
        
        if not any(os.scandir(self.config.index_path)):
            raise FileNotFoundError(f"Index directory is empty: {self.config.index_path}")
        
        try:
            # Load existing index without dynamic fields (they're already in the schema)
            self._initialize_index()
            logger.info(f"Successfully loaded existing index from: {self.config.index_path}")
            return self
        except Exception as e:
            logger.error(f"Failed to load index from {self.config.index_path}: {e}")
            self.close()
            raise

    def from_documents(self, documents: List[Document]) -> "BM25IndexBuilder":
        """Build index from document list (only for initial creation)
        
        This method is intended for creating a new index from scratch.
        If you want to add documents to an existing index, use add_documents() instead.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Self (BM25IndexBuilder instance)
            
        Raises:
            ValueError: If documents list is empty
            RuntimeError: If index is already loaded (use add_documents instead)
            Exception: If there's an error during index building
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if self._index is not None:
            raise RuntimeError(
                "Index is already loaded. from_documents() is only for initial index creation. "
                "To add documents to existing index, use: builder.add_documents(documents)"
            )
        
        try:
            self._build_index(documents)
            return self
        except Exception:
            self.close()
            raise



    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        order_by_field: Optional[str] = None,
        order_desc: bool = True,
        with_score: Optional[bool] = None,
        use_phrase_query: Optional[bool] = None,
        **kwargs: Any
    ) -> List[Document]:
        """执行搜索并返回文档列表"""
        from tantivy import Query, Occur, Order
        
        # Use config defaults if parameters not provided
        k = k if k is not None else self.config.k
        filters = filters or {}
        with_score = with_score if with_score is not None else self.config.with_score
        use_phrase_query = use_phrase_query if use_phrase_query is not None else self.config.search_kwargs.get("use_phrase_query", False)
        
        # Validate k parameter
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")
        
        if not query.strip():
            logger.info("Empty query received, returning empty results.")
            return []

        self._ensure_index_loaded()
        
        # 1. Preprocess query
        try:
            query_tokens = self.tokenizer_manager.get_current_tokenizer()(query)
            logger.debug(f"Query tokens: {query_tokens}")
        except Exception as e:
            logger.error(f"Error during query preprocessing: {e}")
            return []

        # 2. Build main query
        if use_phrase_query and len(query_tokens) > 1:
            # Use phrase query for better relevance
            phrase_query = ' '.join(query_tokens)
            main_query = self._index.parse_query(f'content_tokens:"{phrase_query}"', ["content_tokens"])
        else:
            # Use standard BM25 query on content_tokens field for proper tokenization
            query_str = ' '.join(query_tokens)
            main_query = self._index.parse_query(query_str, ["content_tokens"])
        
        # 3. Build filter queries
        filter_subqueries = []
        for field_name, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            if not values:
                continue
            try:
                q = Query.term_set_query(self._index.schema, field_name, values)
                filter_subqueries.append((Occur.Must, q))
            except Exception as e:
                logger.warning(f"Skipping invalid filter field '{field_name}': {e}")
        
        # 4. Combine queries
        final_query = (
            Query.boolean_query([(Occur.Must, main_query)] + filter_subqueries)
            if filter_subqueries else main_query
        )

        # 5. Calculate actual search k (expand search range in filter mode)
        search_k = k * 3 if filter_subqueries else k

        # 6. Execute search
        try:
            searcher = self._index.searcher()
            order = Order.Desc if order_desc else Order.Asc
            search_result = searcher.search(
                final_query,
                limit=search_k,
                order_by_field=order_by_field,
                order=order
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

        # 7. Assemble results
        results = []
        for score, doc_address in search_result.hits[:k]:  # Truncate to k
            try:
                tantivy_doc = searcher.doc(doc_address)
                metadata_field = tantivy_doc.get_first("metadata") or {}
                
                if isinstance(metadata_field, str):
                    try:
                        metadata_field = json.loads(metadata_field)
                    except json.JSONDecodeError:
                        metadata_field = {}
                
                # Add score to metadata if with_score is True
                if with_score:
                    metadata_field = {**metadata_field, "score": float(score)}
                else:
                    # Ensure score is not included when with_score is False
                    metadata_field = {k: v for k, v in metadata_field.items() if k != "score"}
                
                document = Document(
                    id=tantivy_doc.get_first("id") or "",
                    content=tantivy_doc.get_first("content") or "",
                    metadata=metadata_field
                )

                results.append(document)
            except Exception as e:
                logger.warning(f"Failed to parse document from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results



    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            if self._index is not None:
                searcher = self._index.searcher()
                num_docs = searcher.num_docs
            else:
                num_docs = 0
        except Exception:
            num_docs = 0
            
        return {
            "num_docs": num_docs,
            "index_path": self.config.index_path,
            "batch_size": self.config.batch_size,
            "tokenize_batch_size": self.config.tokenize_batch_size,
            "max_workers": self.config.max_workers,
            "writer_heap_size_mb": self.writer_heap_size / (1024 * 1024),
            "enable_gc": self.config.enable_gc,
            "tokenizers_registered": self._tokenizers_registered,
            "use_jieba": self.tokenizer_manager._use_jieba,
            "use_custom_preprocess": self.tokenizer_manager.custom_preprocess_func is not None,
            "executor_active": self._executor is not None and not self._executor_closed
        }

    def get_tokenizer_stats(self) -> dict:
        """Get tokenizer statistics
        
        Returns:
            Dictionary containing tokenizer statistics
        """
        return self.tokenizer_manager.get_stats()

    def __repr__(self) -> str:
        """String representation of the BM25IndexBuilder instance"""
        try:
            if self._index is not None:
                searcher = self._index.searcher()
                num_docs = searcher.num_docs
            else:
                num_docs = 0
        except:
            num_docs = 0
        
        tokenizer = self.tokenizer_manager.get_tokenizer_info()
        
        return (
            f"{self.__class__.__name__}("
            f"docs={num_docs}, "
            f"index_path='{self.config.index_path}', "
            f"workers={self.config.max_workers}, "
            f"tokenizer={tokenizer})"
        )


    def __enter__(self) -> "BM25IndexBuilder":
        """Context manager entry point
        
        Returns:
            BM25IndexBuilder instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出，清理资源"""
        self.close()
        if exc_type is not None:
            logger.error(f"Exception in BM25IndexBuilder context: {exc_type.__name__}: {exc_val}")

    def _get_executor(self) -> Optional[ProcessPoolExecutor]:
        """Lazy load process pool executor with initializer
        
        Returns:
            ProcessPoolExecutor instance or None if not available
        """
        max_workers = self.config.max_workers or min(4, multiprocessing.cpu_count() - 1)
        if max_workers > 1 and self._executor is None and not self._executor_closed:
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=init_jieba_worker  # Initialize jieba in each worker process
                )
                logger.debug(f"Process pool executor created with {max_workers} workers")
            except Exception as e:
                logger.error(f"Failed to create process pool executor: {e}")
                self._executor_closed = True
        return self._executor

    def close(self) -> None:
        """Close the process pool executor manually"""
        if self._executor and not self._executor_closed:
            try:
                self._executor.shutdown(wait=True)
                logger.info("Process pool executor closed successfully")
            except Exception as e:
                logger.error(f"Error closing process pool executor: {e}")
            finally:
                self._executor = None
                self._executor_closed = True

    def __del__(self) -> None:
        """Destructor to close process pool"""
        try:
            self.close()
        except Exception as e:
            try:
                logger.error(f"Error in __del__: {e}")
            except:
                pass