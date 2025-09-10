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

from framework.config import AbstractConfig
from framework.module import AbstractModule
from core.utils.data_model import Document
from core.retrieval.tantivy_bm25 import TantivyBM25Retriever, TantivyBM25RetrieverConfig
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


class BM25IndexBuilderConfig(AbstractConfig):
    """
    Configuration for BM25IndexBuilder.
    Contains all configuration parameters for index building and retrieval.
    Must be used to instantiate BM25IndexBuilder via build().
    """
    type: Literal["bm25_indexer"] = "bm25_indexer"
    
    # Index building configuration
    index_path: str = Field(description="Path to store the index (required)")
    preprocess_func_name: Optional[str] = Field(default=None, description="Name of the preprocessing function to use")
    bm25_k1: float = Field(default=1.2, description="BM25 k1 parameter, must be greater than 0")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter, must be between 0 and 1")
    stopwords_file: Optional[str] = Field(default=None, description="Path to custom stopwords file (txt format, one word per line). If not provided, stopwords will be selected automatically based on tokenizer.")
    writer_heap_size: Optional[int] = Field(default=None, description="Heap size for the index writer")
    batch_size: int = Field(default=50, description="Number of documents to process in each batch")
    tokenize_batch_size: int = Field(default=200, description="Number of texts to tokenize in each batch")
    max_workers: Optional[int] = Field(default=None, description="Maximum number of worker processes")
    progress_interval: int = Field(default=500, description="Interval for progress reporting")
    enable_gc: bool = Field(default=True, description="Whether to enable garbage collection")
    queue_maxsize: int = Field(default=1000, description="Maximum size of the processing queue")
    

    # Runtime-only
    preprocess_func: Optional[Callable[[str], List[str]]] = Field(default=None, exclude=True)
    progress_callback: Optional[Callable] = Field(default=None, exclude=True)

    k: int = Field(default=10, description="Default number of documents to return in search", gt=0, exclude=True)
    with_score: bool = Field(default=True, description="Whether to include relevance scores in results", exclude=True)
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"use_phrase_query": False}, 
        description="Additional search parameters including use_phrase_query",
        exclude=True
    )
    
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


class BM25IndexBuilder(AbstractModule):
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
    
    Configuration parameters (from config):
        index_path (str): Path to store the index
        preprocess_func (Callable): Custom text preprocessing function
        bm25_k1 (float): BM25 k1 parameter
        bm25_b (float): BM25 b parameter
        stopwords_file (str): Path to custom stopwords file
        writer_heap_size (int): Heap size for the index writer
        batch_size (int): Number of documents to process in each batch
        tokenize_batch_size (int): Number of texts to tokenize in each batch
        max_workers (int): Maximum number of worker processes
        progress_interval (int): Interval for progress reporting
        enable_gc (bool): Whether to enable garbage collection
        progress_callback (Callable): Callback function for progress reporting
    
    Runtime instance variables:
        tokenizer_manager: TokenizerManager instance
        _index: Tantivy index instance
        _schema: Tantivy schema
        _writer_heap_size: Calculated heap size for the index writer
        _tokenizers_registered: Whether tokenizers are registered
        _executor: ProcessPoolExecutor instance
        _executor_closed: Whether executor is closed
        _queue: Queue for producer-consumer pattern
        _writer_thread: Writer thread instance
        _stop_event: Threading event for stopping operations
    
    Core methods:
        - load_local: Load existing index from local path
        - from_documents: Build new index from document list (initial creation only)
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
        >>> # Create new index (initial creation)
        >>> config = BM25IndexBuilderConfig(index_path="./my_index")
        >>> builder = config.build()
        >>> builder.from_documents(initial_documents)  # Only for initial creation
        >>> retriever = builder.as_retriever()
        
        >>> # Add more documents to existing index
        >>> builder.add_documents(additional_documents)  # Use add_documents for more data
        
        >>> # Load existing index
        >>> config2 = BM25IndexBuilderConfig(index_path="./existing_index")
        >>> builder2 = config2.build()
        >>> builder2.load_local()
        >>> builder2.add_documents(new_documents)  # Add documents to loaded index
        >>> retriever = builder2.as_retriever()
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
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=False)
        schema_builder.add_json_field("metadata", stored=True)
        
        # Add dynamic fields based on document metadata
        if documents:
            dynamic_fields = self._extract_string_fields_from_documents(documents)
            for field_name in dynamic_fields:
                schema_builder.add_text_field(field_name, tokenizer_name="raw", stored=False, fast=True)
                logger.debug(f"Added dynamic field: {field_name}")
        
        self._schema = schema_builder.build()

        # Load existing index or create new one
        is_new_index = True
        if os.path.exists(self.config.index_path) and any(os.scandir(self.config.index_path)):
            logger.info(f"Loading existing index from: {self.config.index_path}")
            self._index = Index.open(self.config.index_path)
            is_new_index = False
        else:
            logger.info(f"Creating new index at: {self.config.index_path}")
            os.makedirs(self.config.index_path, exist_ok=True)
            self._index = Index(self._schema, path=self.config.index_path)
        
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
                logger.info(f"Overwrite mode: attempting to delete {len(doc_ids)} existing documents")
                deleted_count = self._delete_documents_by_ids(doc_ids)
                logger.info(f"Overwrite mode: successfully deleted {deleted_count} documents")
                
                # Ensure index is reloaded after deletion for consistency
                if self._index is not None:
                    self._index.reload()
        
        return self._build_index(documents)

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
        
        deleted_count = self._delete_documents_by_ids(unique_doc_ids)
        return deleted_count

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
        self._ensure_index_loaded()

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



    def as_retriever(self, k: Optional[int] = None, with_score: Optional[bool] = None, search_kwargs: Optional[Dict[str, Any]] = None) -> "TantivyBM25Retriever":
        """Create a retriever from the current index
        
        The retriever uses the retrieval parameters configured in the index.
        All retrieval-related configurations are defined in BM25IndexBuilderConfig.
        
        Returns:
            TantivyBM25Retriever instance
            
        Raises:
            RuntimeError: If index is not initialized
            
        Examples:
            # Create retriever - uses configuration from the index
            retriever = builder.as_retriever()
            
            # Runtime can override default configuration from the index
            results = retriever.invoke(
                "query text",
                k=5,                    # Override default k value from index
                use_phrase_query=True,  # Override default setting from index
                filters={"category": "tech"}
            )
        """
        self._ensure_index_loaded()
        self._index.reload()
        
        runtime_k = k or self.config.k
        runtime_with_score = with_score or self.config.with_score
        runtime_search_kwargs = search_kwargs or self.config.search_kwargs.copy()
        # Create simplified retriever configuration using parameters from index configuration
        retriever_config = TantivyBM25RetrieverConfig(
            # Inject runtime dependencies
            index=self._index,
            preprocess_func=self.tokenizer_manager.get_current_tokenizer(),
            stopwords=self.tokenizer_manager.get_stopwords(),
            
            # Get retrieval parameters from index configuration
            k=runtime_k,
            with_score=runtime_with_score,
            search_kwargs=runtime_search_kwargs
        )
        
        # Create and return retriever
        retriever = retriever_config.build()
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