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
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor

from core.utils.data_model import Document
from core.retrieval.tantivy_bm25_new import TantivyBM25Retriever
from encapsulation.database.utils.TokenizerManager import TokenizerManager

try:
    from tantivy import (
        Index, SchemaBuilder, Document as TantivyDocument,
        Tokenizer, TextAnalyzerBuilder, Filter
    )
except ImportError:
    raise ImportError(
        "The 'tantivy-py' library was not found. Please install it using: pip install tantivy"
    )

logger = logging.getLogger(__name__)



def init_jieba_worker():
    """在工作进程中初始化jieba，减少重复初始化开销"""
    try:
        import jieba
        return jieba
    except ImportError:
        raise ImportError("The 'jieba' library was not found. Please install it using: pip install jieba")



class BM25IndexBuilder:
    """
    基于 Tantivy 搜索引擎的高性能索引构建器

    优化特性：
    - 流式处理减少内存占用
    - 智能批处理
    - 内存管理和垃圾回收
    - 异常恢复机制
    - 优化的多进程处理
    - 增量更新和文档去重
    - 上下文管理器支持
    - 自动语言检测和分词器选择 (仅在未提供自定义 preprocess_func 时)
    """

    def __init__(self,
                 index_path: str = None,
                 preprocess_func: Optional[Callable[[str], List[str]]] = None,
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.75,
                 stopwords: Optional[List[str]] = None,
                 writer_heap_size: Optional[int] = None,
                 batch_size: int = 50,
                 tokenize_batch_size: int = 200,
                 max_workers: Optional[int] = None,
                 progress_interval: int = 500,
                 enable_gc: bool = True,
                 progress_callback: Optional[Callable] = None,
                 **kwargs):
        if bm25_k1 <= 0:
            raise ValueError(f"bm25_k1 must be greater than 0, but got {bm25_k1}")
        if not (0 <= bm25_b <= 1):
            raise ValueError(f"bm25_b must be between 0 and 1, but got {bm25_b}")

        self.index_path = index_path or f"./tantivy_index_{uuid.uuid4().hex[:8]}"
        self.bm25_k1, self.bm25_b = bm25_k1, bm25_b
        self.stopwords = stopwords or ["的", "是", "在", "和", "与", "或", "了", "等", "就", "也",
                                       "一", "个", "有", "这", "那", "不", "但", "对", "为", "很"]
        self._index: Optional[Index] = None
        self._schema = None
        self._batch_size = batch_size
        self._tokenize_batch_size = tokenize_batch_size
        self._max_workers = max_workers or min(4, multiprocessing.cpu_count() - 1)
        self._progress_interval = progress_interval
        self._enable_gc = enable_gc
        self._tokenizers_registered = False
        
        # 使用 TokenizerManager 管理分词器
        self.tokenizer_manager = TokenizerManager(preprocess_func)
        # 同步停用词
        self.tokenizer_manager.update_stopwords(self.stopwords)

        # 动态 heap_size (默认取系统内存 20%，最大 1GB)
        if writer_heap_size is None:
            total_mem = psutil.virtual_memory().total
            # TODO 比例和最大值需要调整
            self._writer_heap_size = min(int(total_mem * 0.2), 1024 * 1024 * 1024)
        else:
            self._writer_heap_size = writer_heap_size

        # 并发 & 队列
        self._executor: Optional[ProcessPoolExecutor] = None
        self._executor_closed = False
        #TODO  maxsize 暂时还不确定
        self._queue = queue.Queue(maxsize=1000)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.progress_callback = progress_callback

        self._initialize_index()



    
    def _set_tokenizer(self, documents: List[Document]):
        """设置分词器（代理到 TokenizerManager）"""
        self.tokenizer_manager.set_tokenizer_by_detection(documents)

    def _tokenize_batch_sequential(self, texts: List[str]) -> List[List[str]]:
        """顺序批量分词（单进程）"""
        return self.tokenizer_manager.batch_tokenize(texts)

    def _tokenize_batch_parallel(self, texts: List[str]) -> List[List[str]]:
        """并行批量分词（多进程）"""
        executor = self._get_executor()
        if not executor or len(texts) <= self._tokenize_batch_size:
            return self._tokenize_batch_sequential(texts)

        batches = [texts[i:i + self._tokenize_batch_size] for i in range(0, len(texts), self._tokenize_batch_size)]
        results = []
        
        # 创建一个可序列化的分词任务
        futures = []
        for batch in batches:
            # 由于 TokenizerManager 可能包含不可序列化的自定义函数，
            # 我们直接使用当前实例的分词方法
            future = executor.submit(self._tokenize_batch_sequential, batch)
            futures.append(future)
            
        for future in futures:
            try:
                results.extend(future.result(timeout=60))
            except Exception as e:
                logger.warning(f"Parallel tokenization failed, fallback to sequential: {e}")
                return self._tokenize_batch_sequential(texts)
        return results

    # TODO 支持任意字段过滤
    def _initialize_index(self):
        if self._index is not None:
            return
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=False)
        schema_builder.add_json_field("metadata", stored=True)
        schema_builder.add_text_field("source", tokenizer_name="raw", stored=False, fast=True)
        schema_builder.add_text_field("tag", tokenizer_name="raw", stored=False, fast=True)
        self._schema = schema_builder.build()

        is_new_index = True
        if os.path.exists(self.index_path) and any(os.scandir(self.index_path)):
            logger.info(f"Loading existing index from: {self.index_path}")
            self._index = Index.open(self.index_path)
            is_new_index = False
        else:
            logger.info(f"Creating new index at: {self.index_path}")
            os.makedirs(self.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.index_path)
        
        # 只在新索引或明确需要时注册tokenizer
        if is_new_index or not self._tokenizers_registered:
            self._register_tokenizers()

        logger.info("Tantivy index initialized successfully")

    def _register_tokenizers(self):
        """注册tokenizer，避免重复注册"""
        if self._tokenizers_registered:
            return
            
        try:
            custom_analyzer = (
                TextAnalyzerBuilder(Tokenizer.whitespace())
                .filter(Filter.lowercase())
                .filter(Filter.custom_stopword(self.stopwords))
                .build()
            )
            self._index.register_tokenizer("custom", custom_analyzer)
            
            raw_analyzer = TextAnalyzerBuilder(Tokenizer.raw()).build()
            self._index.register_tokenizer("raw", raw_analyzer)
            
            self._tokenizers_registered = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Tokenizers registered successfully")
                
        except Exception as e:
            logger.error(f"Failed to register tokenizers: {e}")
            raise

    def _writer_worker(self, writer):
        """消费者线程：索引写入"""
        batch_docs = []
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                doc = self._queue.get(timeout=1)
                if doc is None:
                    break
                batch_docs.append(doc)
                if len(batch_docs) >= self._batch_size:
                    self._batch_write_documents(batch_docs, writer)
                    batch_docs.clear()
            except queue.Empty:
                continue
        if batch_docs:
            self._batch_write_documents(batch_docs, writer)

    def _batch_write_documents(self, docs: List[TantivyDocument], writer):
        try:
            writer.add_documents(docs)
        except AttributeError:
            for d in docs:
                writer.add_document(d)

    def _delete_documents_by_ids(self, doc_ids: List[str]):
        """根据ID删除文档"""
        if not doc_ids:
            return
            
        try:
            writer = self._index.writer(heap_size=self._writer_heap_size)
            
            for doc_id in doc_ids:
                writer.delete_documents("id", doc_id)
            
            writer.commit()
            # 重新加载索引以反映删除操作
            self._index.reload()
            logger.info(f"Deleted {len(doc_ids)} documents from index")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise


    def build_index(self, documents: List[Document]) -> List[str]:
        """使用生产者-消费者模式的索引构建方法"""
        if not documents:
            logger.warning("No documents provided for indexing")
            return []
        
        # 只有在没有使用自定义 preprocess_func 时才检测语言和设置分词器
        if self.tokenizer_manager.custom_preprocess_func is None:
            self._set_tokenizer(documents)
        
        total_docs = len(documents)
        added_ids, processed_count = [], 0
        
        # 创建写入器
        writer = self._index.writer(heap_size=self._writer_heap_size)
        
        # 启动写入线程
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
                
                # 添加元数据
                metadata = doc.metadata or {}
                tantivy_doc.add_json("metadata", metadata)
                
                # 添加元数据索引字段
                if "source" in metadata:
                    tantivy_doc.add_text("source", str(metadata["source"]))
                if "tag" in metadata:
                    tantivy_doc.add_text("tag", str(metadata["tag"]))
                
                # 放入队列
                self._queue.put(tantivy_doc)

                added_ids.append(doc_id)
                processed_count += 1
                
                # 进度报告
                if processed_count % self._progress_interval == 0:
                    elapsed = time.time() - start_time
                    stats = {
                        "processed": processed_count,
                        "total": total_docs,
                        "elapsed_sec": round(elapsed, 2),
                        "throughput_docs_sec": round(processed_count / elapsed, 2)
                    }
                    logger.info(f"[IndexProgress] {json.dumps(stats, ensure_ascii=False)}")
                    if self.progress_callback:
                        self.progress_callback(processed_count, total_docs, stats)

            # 结束信号
            self._queue.put(None)
            self._writer_thread.join()
            writer.commit()
            self._index.reload()
            
            # 报告使用的分词器
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
            self._stop_event.set()
            if self._enable_gc:
                gc.collect()
        
        return added_ids

    def add_documents(self, documents: List[Document], overwrite: bool = False) -> List[str]:
        """向现有索引添加文档，支持去重"""
        if not documents:
            logger.warning("No documents provided for adding")
            return []
        
        if overwrite:
            # 提取要添加的文档ID
            doc_ids = [str(doc.id) for doc in documents if doc.id is not None]
            if doc_ids:
                logger.info(f"Overwrite mode: deleting {len(doc_ids)} existing documents")
                self._delete_documents_by_ids(doc_ids)
        
        return self.build_index(documents)

    def update_documents(self, documents: List[Document]) -> List[str]:
        """更新文档，先删除后添加"""
        return self.add_documents(documents, overwrite=True)

    def delete_documents(self, doc_ids: List[str]) -> int:
        """删除指定ID的文档"""
        if not doc_ids:
            return 0
        
        # 去重处理
        unique_doc_ids = list(set(doc_ids))
        
        self._delete_documents_by_ids(unique_doc_ids)
        return len(unique_doc_ids)

    # TODO 修改为batch获取
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """根据ID获取单个文档"""
        if self._index is None:
            raise RuntimeError("Index has not been initialized. Call build_index() or load_local() first.")

        try:
            searcher = self._index.searcher()
            # 构建精确匹配 id 字段的查询
            query = self._index.parse_query(f'id:"{doc_id}"', ["id"])
            results = searcher.search(query, 1)

            if results.hits:
                _, doc_address = results.hits[0]
                tantivy_doc = searcher.doc(doc_address)
                
                # 提取字段
                doc_id_field = tantivy_doc.get_first("id") or ""
                content_field = tantivy_doc.get_first("content") or ""
                metadata_field = tantivy_doc.get_first("metadata") or {}

                # 尝试解析 metadata（如果是字符串）
                if isinstance(metadata_field, str):
                    try:
                        metadata_field = json.loads(metadata_field)
                    except json.JSONDecodeError:
                        metadata_field = {}

                return Document(
                    id=doc_id_field,
                    content=content_field,
                    metadata=metadata_field
                )
            return None

        except Exception as e:
            logger.error(f"Error retrieving document by ID '{doc_id}': {e}")
            return None


    @classmethod
    def from_documents(cls, documents: List[Document], **kwargs: Any) -> "BM25IndexBuilder":
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        builder = cls(**kwargs)
        try:
            builder.build_index(documents)
            return builder
        except Exception:
            builder.close()
            raise

    @classmethod
    def load_local(cls, index_path: str, **kwargs: Any) -> "BM25IndexBuilder":
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
        
        builder = cls(index_path=index_path, **kwargs)
        try:
            builder._index.reload()
            logger.info("Successfully loaded existing index")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            builder.close()
            raise
        return builder


    def as_retriever(self, **kwargs: Any) -> "TantivyBM25Retriever":
        if self._index is None:
            raise RuntimeError(
                "Index has not been initialized. "
                "Call build_index() or load_local() first."
            )

        # 确保索引是最新的
        self._index.reload()
        
        retriever = TantivyBM25Retriever(
            index=self._index,
            preprocess_func=self.tokenizer_manager.get_current_tokenizer(), # 传递当前使用的分词器
            stopwords=self.stopwords,
            **kwargs
        )
        
        # 重新加载搜索器以获取最新数据
        retriever.reload_searcher()
        
        return retriever

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            searcher = self._index.searcher()
            num_docs = searcher.num_docs
        except Exception:
            num_docs = 0
            
        return {
            "num_docs": num_docs,
            "index_path": self.index_path,
            "batch_size": self._batch_size,
            "tokenize_batch_size": self._tokenize_batch_size,
            "max_workers": self._max_workers,
            "writer_heap_size_mb": self._writer_heap_size / (1024 * 1024),
            "enable_gc": self._enable_gc,
            "tokenizers_registered": self._tokenizers_registered,
            "use_jieba": self.tokenizer_manager._use_jieba,
            "use_custom_preprocess": self.tokenizer_manager.custom_preprocess_func is not None,
            "executor_active": self._executor is not None and not self._executor_closed
        }

    def get_tokenizer_stats(self) -> dict:
        """获取分词器统计信息"""
        return self.tokenizer_manager.get_stats()

    def __repr__(self) -> str:
        num_docs = 0
        try:
            searcher = self._index.searcher()
            num_docs = searcher.num_docs
        except:
            pass
        
        tokenizer = self.tokenizer_manager.get_tokenizer_info()
        
        return (
            f"{self.__class__.__name__}("
            f"docs={num_docs}, "
            f"index_path='{self.index_path}', "
            f"workers={self._max_workers}, "
            f"tokenizer={tokenizer})"
        )


    def __enter__(self) -> "BM25IndexBuilder":
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源清理"""
        self.close()
        if exc_type is not None:
            logger.error(f"Exception in BM25IndexBuilder context: {exc_type.__name__}: {exc_val}")

    def _get_executor(self) -> Optional[ProcessPoolExecutor]:
        """懒加载进程池，带初始化器"""
        if self._max_workers > 1 and self._executor is None and not self._executor_closed:
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=init_jieba_worker  # 在每个工作进程中初始化jieba
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Process pool executor created with {self._max_workers} workers")
            except Exception as e:
                logger.error(f"Failed to create process pool executor: {e}")
                self._executor_closed = True
        return self._executor


    def close(self):
        """手动关闭进程池"""
        if self._executor and not self._executor_closed:
            try:
                self._executor.shutdown(wait=True)
                if logger.isEnabledFor(logging.INFO):
                    logger.info("Process pool executor closed successfully")
            except Exception as e:
                logger.error(f"Error closing process pool executor: {e}")
            finally:
                self._executor = None
                self._executor_closed = True

    def __del__(self):
        """析构时关闭进程池"""
        try:
            self.close()
        except Exception as e:
            # 记录析构时的异常，避免掩盖问题
            try:
                logger.error(f"Error in __del__: {e}")
            except:
                # 如果连日志都无法记录，则静默处理
                pass

