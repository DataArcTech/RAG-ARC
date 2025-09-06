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
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import jieba
from core.utils.data_model import Document
from core.retrieval.tantivy_bm25 import TantivyBM25Retriever
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
    """Initialize jieba in the worker process to reduce initialization overhead"""
    return jieba


class BM25IndexBuilder:
    """
    BM25IndexBuilder is a high-performance index builder based on the Tantivy search engine.
    
    This class implements efficient indexing for document collections by leveraging Tantivy's
    capabilities, supporting stream processing to reduce memory usage, intelligent batching,
    memory management, exception recovery mechanisms, optimized multiprocessing, incremental
    updates, and document deduplication.
    
    Key features:
    - Stream processing to reduce memory footprint
    - Intelligent batch processing
    - Memory management and garbage collection
    - Exception recovery mechanisms
    - Optimized multiprocessing
    - Incremental updates and document deduplication
    - Context manager support
    - Automatic language detection and tokenizer selection (only when no custom preprocess_func is provided)
    
    Main parameters:
        index_path (str): Path to store the index
        preprocess_func (Callable): Custom text preprocessing function
        bm25_k1 (float): BM25 k1 parameter
        bm25_b (float): BM25 b parameter
        stopwords (List[str]): List of stopwords to filter out
        writer_heap_size (int): Heap size for the index writer
        batch_size (int): Number of documents to process in each batch
        tokenize_batch_size (int): Number of texts to tokenize in each batch
        max_workers (int): Maximum number of worker processes
        progress_interval (int): Interval for progress reporting
        enable_gc (bool): Whether to enable garbage collection
        progress_callback (Callable): Callback function for progress reporting
    
    Core methods:
        - build_index: Build index using producer-consumer pattern
        - add_documents: Add documents to existing index
        - update_documents: Update documents in index
        - delete_documents: Delete documents from index
        - get_document_by_id: Retrieve document by ID
        - as_retriever: Create retriever from current index
        - get_index_stats: Get index statistics
        - close: Close process pool executor
    
    Performance considerations:
        - Stream processing reduces memory usage
        - Intelligent batching optimizes performance
        - Memory management and garbage collection reduce memory footprint
        - Multiprocessing improves tokenization performance
        - Context manager ensures proper resource cleanup
    
    Typical usage:
        >>> with BM25IndexBuilder(index_path="./my_index") as builder:
        >>>     builder.build_index(documents)
        >>>     retriever = builder.as_retriever()
        
        >>> builder = BM25IndexBuilder.from_documents(documents)
        >>> retriever = builder.as_retriever()
        
        >>> builder = BM25IndexBuilder.load_local("./my_index")
        >>> retriever = builder.as_retriever()
    
    Attributes:
        index_path: Path to store the index
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter
        stopwords: List of stopwords to filter out
        tokenizer_manager: TokenizerManager instance
        _index: Tantivy index instance
        _schema: Tantivy schema
        _batch_size: Number of documents to process in each batch
        _tokenize_batch_size: Number of texts to tokenize in each batch
        _max_workers: Maximum number of worker processes
        _progress_interval: Interval for progress reporting
        _enable_gc: Whether to enable garbage collection
        _writer_heap_size: Heap size for the index writer
        _tokenizers_registered: Whether tokenizers are registered
        progress_callback: Callback function for progress reporting
    """

    def __init__(self,
                 index_path: Optional[str] = None,
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
        """Initialize BM25 Index Builder
        
        Args:
            index_path: Path to store the index, defaults to a generated path
            preprocess_func: Custom text preprocessing function
            bm25_k1: BM25 k1 parameter, must be greater than 0
            bm25_b: BM25 b parameter, must be between 0 and 1
            stopwords: List of stopwords to filter out
            writer_heap_size: Heap size for the index writer
            batch_size: Number of documents to process in each batch
            tokenize_batch_size: Number of texts to tokenize in each batch
            max_workers: Maximum number of worker processes
            progress_interval: Interval for progress reporting
            enable_gc: Whether to enable garbage collection
            progress_callback: Callback function for progress reporting
            **kwargs: Additional parameters
        """
        # Validate parameters
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
        
        # Use TokenizerManager to manage tokenizers
        self.tokenizer_manager = TokenizerManager(preprocess_func)
        # Sync stopwords
        self.tokenizer_manager.update_stopwords(self.stopwords)

        # Dynamic heap_size (default to 20% of system memory, max 1GB)
        if writer_heap_size is None:
            total_mem = psutil.virtual_memory().total
            self._writer_heap_size = min(int(total_mem * 0.2), 1024 * 1024 * 1024)
        else:
            self._writer_heap_size = writer_heap_size

        self._executor: Optional[ProcessPoolExecutor] = None
        self._executor_closed = False
        self._queue = queue.Queue(maxsize=1000)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.progress_callback = progress_callback

        self._initialize_index()
    
    def _set_tokenizer(self, documents: List[Document]):
        """Set tokenizer (proxied to TokenizerManager)"""
        self.tokenizer_manager.set_tokenizer_by_detection(documents)

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
        if not executor or len(texts) <= self._tokenize_batch_size:
            return self._tokenize_batch_sequential(texts)

        # Split texts into batches
        batches = [texts[i:i + self._tokenize_batch_size] for i in range(0, len(texts), self._tokenize_batch_size)]
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

    # TODO Support arbitrary field filtering
    def _initialize_index(self) -> None:
        """Initialize the Tantivy index"""
        if self._index is not None:
            return
            
        # Build schema
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=False)
        schema_builder.add_json_field("metadata", stored=True)
        schema_builder.add_text_field("source", tokenizer_name="raw", stored=False, fast=True)
        schema_builder.add_text_field("tag", tokenizer_name="raw", stored=False, fast=True)
        self._schema = schema_builder.build()

        # Load existing index or create new one
        is_new_index = True
        if os.path.exists(self.index_path) and any(os.scandir(self.index_path)):
            logger.info(f"Loading existing index from: {self.index_path}")
            self._index = Index.open(self.index_path)
            is_new_index = False
        else:
            logger.info(f"Creating new index at: {self.index_path}")
            os.makedirs(self.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.index_path)
        
        # Register tokenizers only for new index or when not yet registered
        if is_new_index or not self._tokenizers_registered:
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
                .filter(Filter.custom_stopword(self.stopwords))
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
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                doc = self._queue.get(timeout=1)
                if doc is None:
                    break
                batch_docs.append(doc)
                if len(batch_docs) >= self._batch_size:
                    self._batch_write_documents(batch_docs, writer)
                    batch_docs.clear()
                    # Trigger garbage collection if enabled
                    if self._enable_gc:
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

    def _delete_documents_by_ids(self, doc_ids: List[str]) -> None:
        """Delete documents by their IDs
        
        Args:
            doc_ids: List of document IDs to delete
        """
        if not doc_ids or self._index is None:
            return
            
        try:
            writer = self._index.writer(heap_size=self._writer_heap_size)
            
            for doc_id in doc_ids:
                writer.delete_documents("id", doc_id)
            
            writer.commit()
            self._index.reload()
            logger.info(f"Deleted {len(doc_ids)} documents from index")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise


    def build_index(self, documents: List[Document]) -> List[str]:
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
        
        if self._index is None:
            raise RuntimeError("Index has not been initialized")
            
        total_docs = len(documents)
        added_ids, processed_count = [], 0
        
        writer = self._index.writer(heap_size=self._writer_heap_size)
        
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
                
                if "source" in metadata:
                    tantivy_doc.add_text("source", str(metadata["source"]))
                if "tag" in metadata:
                    tantivy_doc.add_text("tag", str(metadata["tag"]))
                
                self._queue.put(tantivy_doc)

                added_ids.append(doc_id)
                processed_count += 1
                
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

            self._queue.put(None)
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
            self._stop_event.set()
            if self._enable_gc:
                gc.collect()
        
        return added_ids

    def add_documents(self, documents: List[Document], overwrite: bool = False) -> List[str]:
        """Add documents to the existing index, supporting deduplication
        
        Args:
            documents: List of Document objects to add
            overwrite: Whether to overwrite existing documents with the same IDs
            
        Returns:
            List of document IDs that were added to the index
        """
        if not documents:
            logger.warning("No documents provided for adding")
            return []
        
        if overwrite:
            doc_ids = [str(doc.id) for doc in documents if doc.id is not None]
            if doc_ids:
                logger.info(f"Overwrite mode: deleting {len(doc_ids)} existing documents")
                self._delete_documents_by_ids(doc_ids)
        
        return self.build_index(documents)

    def update_documents(self, documents: List[Document]) -> List[str]:
        """Update documents in the index by first deleting then adding
        
        Args:
            documents: List of Document objects to update
            
        Returns:
            List of document IDs that were updated in the index
        """
        return self.add_documents(documents, overwrite=True)

    def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents with specified IDs
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not doc_ids:
            return 0
        
        unique_doc_ids = list(set(doc_ids))
        
        self._delete_documents_by_ids(unique_doc_ids)
        return len(unique_doc_ids)

    # TODO Modify to batch retrieval
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by its ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document object if found, None otherwise
            
        Raises:
            RuntimeError: If index is not initialized
        """
        if self._index is None:
            raise RuntimeError("Index has not been initialized. Call build_index() or load_local() first.")

        try:
            searcher = self._index.searcher()
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
        """Create BM25IndexBuilder from document list
        
        Args:
            documents: List of Document objects to index
            **kwargs: Additional parameters for BM25IndexBuilder
            
        Returns:
            BM25IndexBuilder instance
            
        Raises:
            ValueError: If documents list is empty
            Exception: If there's an error during index building
        """
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
        """Load existing index from local path
        
        Args:
            index_path: Path to the existing index
            **kwargs: Additional parameters for BM25IndexBuilder
            
        Returns:
            BM25IndexBuilder instance
            
        Raises:
            FileNotFoundError: If index path does not exist
            Exception: If there's an error during index loading
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
        
        builder = cls(index_path=index_path, **kwargs)
        try:
            if builder._index is not None:
                builder._index.reload()
            logger.info("Successfully loaded existing index")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            builder.close()
            raise
        return builder


    def as_retriever(self, **kwargs: Any) -> "TantivyBM25Retriever":
        """Create a retriever from the current index
        
        Args:
            **kwargs: Additional parameters for the retriever
            
        Returns:
            TantivyBM25Retriever instance
            
        Raises:
            RuntimeError: If index is not initialized
        """
        if self._index is None:
            raise RuntimeError(
                "Index has not been initialized. "
                "Call build_index() or load_local() first."
            )

        # Ensure index is up to date
        self._index.reload()
        
        # Create retriever with current tokenizer
        retriever = TantivyBM25Retriever(
            index=self._index,
            preprocess_func=self.tokenizer_manager.get_current_tokenizer(),
            stopwords=self.stopwords,
            **kwargs
        )
        
        # Reload searcher to get latest data
        retriever.reload_searcher()
        
        return retriever

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
            f"index_path='{self.index_path}', "
            f"workers={self._max_workers}, "
            f"tokenizer={tokenizer})"
        )


    def __enter__(self) -> "BM25IndexBuilder":
        """Context manager entry point
        
        Returns:
            BM25IndexBuilder instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point, ensures resource cleanup
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.close()
        if exc_type is not None:
            logger.error(f"Exception in BM25IndexBuilder context: {exc_type.__name__}: {exc_val}")

    def _get_executor(self) -> Optional[ProcessPoolExecutor]:
        """Lazy load process pool executor with initializer
        
        Returns:
            ProcessPoolExecutor instance or None if not available
        """
        if self._max_workers > 1 and self._executor is None and not self._executor_closed:
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=init_jieba_worker  # Initialize jieba in each worker process
                )
                logger.debug(f"Process pool executor created with {self._max_workers} workers")
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