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
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import jieba

from encapsulation.data_model.schema import Chunk
from encapsulation.database.utils.TokenizerManager import TokenizerManager
from framework.shared_module_decorator import shared_module

try:
    from tantivy import (
        Index, SchemaBuilder, Document as TantivyDocument,
        Tokenizer, TextAnalyzerBuilder, Filter
    )
except ImportError:
    raise ImportError("Please install dependencies with: uv sync")

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



def init_jieba_worker():
    """Initialize jieba in the worker process to reduce initialization overhead"""
    return jieba


@shared_module
class BM25IndexBuilder():
    """
    Based on Tantivy's BM25 implementation, this class provides a convenient

    main features include:
    - Indexing and management of chunks
    - Chinese tokenization and multi-language support
    - Streaming and batch operations
    - Multi-process optimization
    """

    def __init__(self, config):
        """Initialize BM25IndexBuilder with configuration

        Args:
            config: BM25IndexBuilderConfig instance
        """
        self.config = config

        # Runtime instance variables (initialized when needed)
        self._index: Optional[Index] = None
        self._schema = None
        self._writer_heap_size: int = 0
        self._tokenizers_registered: bool = False
        self._executor: Optional[ProcessPoolExecutor] = None
        self._executor_closed: bool = False
        self._queue: queue.Queue = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = None
        self._tokenizer_manager: TokenizerManager = None

    @property
    def tokenizer_manager(self) -> TokenizerManager:
        """Lazy-initialized tokenizer manager"""
        if self._tokenizer_manager is None:
            self._tokenizer_manager = TokenizerManager(
                custom_preprocess_func=None,  # Runtime injection no longer supported
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


    
    
    def _set_tokenizer(self, chunks: List[Chunk]):
        """Set tokenizer (proxied to TokenizerManager)"""
        self.tokenizer_manager.set_tokenizer_by_detection(chunks)

    def _set_tokenizer_from_existing_index(self):
        """Set tokenizer based on existing index"""
        try:
            if self._index is None:
                logger.warning("Index not loaded, cannot set tokenizer from existing index")
                return

            # Check if index has chunks
            searcher = self._index.searcher()
            num_docs = searcher.num_docs

            if num_docs == 0:
                logger.warning("Index is empty, using default whitespace tokenizer")
                return

            # For existing Chinese indices, set tokenizer to jieba
            logger.info(f"Loading existing index with {num_docs} chunks, setting tokenizer to jieba for Chinese content")
            self.tokenizer_manager._use_jieba = True
            self.tokenizer_manager._load_stopwords()

        except Exception as e:
            logger.warning(f"Failed to set tokenizer from existing index: {e}, using default whitespace tokenizer")

    def _ensure_index_loaded(self) -> None:
        """Ensure index is loaded before operations
        
        Raises:
            RuntimeError: If index is not loaded
        """
        if self._index is None:
            raise RuntimeError(
                "Index is not loaded. Call load_local() to load existing index "
                "or from_chunks() to create new index."
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

    def _extract_string_fields_from_chunks(self, chunks: List[Chunk]) -> set[str]:
        """Extract string fields from chunk metadata for dynamic schema creation

        Args:
            chunks: List of chunks to analyze

        Returns:
            Set of field names that contain string values
        """
        string_fields = set()

        for chunk in chunks:
            if not chunk.metadata:
                continue

            for key, value in chunk.metadata.items():
                # Only include string fields for filtering, exclude system/score fields
                if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                    string_fields.add(key)

        logger.debug(f"Extracted dynamic string fields: {string_fields}")
        return string_fields

    def _initialize_index(self, chunks: Optional[List[Chunk]] = None) -> None:
        """Initialize the Tantivy index with dynamic schema based on chunk metadata
        
        Args:
            chunks: Optional list of chunks to analyze for dynamic field creation
        """
        if self._index is not None:
            return
            
        # Build schema with core fields
        schema_builder = SchemaBuilder()
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw", fast=True)
        schema_builder.add_text_field("content", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_tokens", tokenizer_name="custom", stored=True)
        schema_builder.add_json_field("metadata", stored=True)

        # Add owner_id field for user isolation (stored for retrieval)
        schema_builder.add_text_field("owner_id", stored=True, tokenizer_name="raw", fast=True)
        logger.debug("Added owner_id field for user isolation")

        # Add dynamic fields based on chunk metadata
        if chunks:
            dynamic_fields = self._extract_string_fields_from_chunks(chunks)
            for field_name in dynamic_fields:
                if field_name == "owner_id":  # Skip owner_id as it's already added
                    continue
                schema_builder.add_text_field(field_name, tokenizer_name="raw", stored=False, fast=True)
                logger.debug(f"Added dynamic field: {field_name}")
        
        self._schema = schema_builder.build()

        # Load existing index or create new one
        if os.path.exists(self.config.index_path):
            with os.scandir(self.config.index_path) as entries:
                has_files = any(entries)
            if has_files:
                logger.info(f"Loading existing index from: {self.config.index_path}")
                try:
                    self._index = Index.open(self.config.index_path)
                    # Get schema from the loaded index
                    self._schema = self._index.schema
                except Exception as e:
                    logger.error(f"Failed to load existing index at {self.config.index_path}: {str(e)}")
                    logger.error("Please check index intergerity or delete manually")
                    raise RuntimeError("Index corrupted or incompatible - manual intervention required")
            else:
                logger.info(f"Creating new index at existing empty directory: {self.config.index_path}")
                self._index = Index(self._schema, path=self.config.index_path)
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
        """Write a batch of tantivy documents to the index

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
            logger.error(f"Error writing batch of tantivy documents: {e}")
            raise

    def _delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        """Delete chunks by their IDs with retry mechanism for lock conflicts

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks actually deleted
        """
        if not chunk_ids:
            return 0

        # Try to load index if not loaded
        if self._index is None:
            try:
                logger.info("Index not loaded, attempting to load from disk for deletion")
                self.load_local()
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"Cannot load index for deletion: {e}")
                # Index doesn't exist, nothing to delete
                return 0

        # Retry mechanism for lock conflicts
        max_retries = 3
        retry_delay = 0.1  # seconds

        for attempt in range(max_retries):
            try:
                # First, check which chunks actually exist
                searcher = self._index.searcher()
                existing_ids = []

                for chunk_id in chunk_ids:
                    query = self._index.parse_query(f'id:"{chunk_id}"', ["id"])
                    results = searcher.search(query, 1)
                    logger.info(f"Checking chunk {chunk_id}: found {len(results.hits)} hits")
                    if results.hits:
                        existing_ids.append(chunk_id)

                if not existing_ids:
                    logger.info("No chunks found to delete")
                    return 0

                # Delete only existing chunks
                writer = self._index.writer(heap_size=self.writer_heap_size)
                deleted_count = 0

                for chunk_id in existing_ids:
                    # Use delete_documents_by_term for exact matching (id field has tokenizer="raw")
                    writer.delete_documents_by_term("id", chunk_id)
                    logger.info(f"Deleted chunk {chunk_id}")
                    deleted_count += 1

                logger.info(f"Committing deletion of {deleted_count} chunks")
                writer.commit()
                logger.info("Reloading index after deletion")
                self._index.reload()
                logger.info(f"Successfully deleted {deleted_count} chunks from index (requested: {len(chunk_ids)})")
                return deleted_count

            except Exception as e:
                error_msg = str(e)
                # Check if it's a lock conflict
                if "LockBusy" in error_msg or "Failed to acquire" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"Lock conflict on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Failed to acquire lock after {max_retries} attempts")
                        raise
                else:
                    # Other errors, don't retry
                    logger.error(f"Error deleting chunks: {e}")
                    raise

        return 0


    def _build_index(self, chunks: List[Chunk]) -> List[str]:
        """Build index using producer-consumer pattern
        
        Args:
            chunks: List of Chunk objects to index
            
        Returns:
            List of chunk IDs that were added to the index
            
        Raises:
            RuntimeError: If there's an error during index building
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return []
        
        if self.tokenizer_manager.custom_preprocess_func is None:
            self._set_tokenizer(chunks)
        
        # For new indices, reinitialize with dynamic fields based on chunks
        index_exists = False
        if os.path.exists(self.config.index_path):
            try:
                with os.scandir(self.config.index_path) as entries:
                    index_exists = any(entries)
            except OSError:
                index_exists = False

        if not index_exists:
            self._index = None  # Reset to force reinitialization with dynamic fields
            self._initialize_index(chunks)
        
        if self._index is None:
            raise RuntimeError("Index has not been initialized")
            
        total_docs = len(chunks)
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
            for chunk in chunks:
                content_tokens = self.tokenizer_manager.get_current_tokenizer()(chunk.content or "")
                chunk_id = str(chunk.id) if chunk.id else str(uuid.uuid4())

                tantivy_doc = TantivyDocument()
                tantivy_doc.add_text("id", chunk_id)
                tantivy_doc.add_text("content", chunk.content or "")
                tantivy_doc.add_text("content_tokens", " ".join(content_tokens))

                # Add owner_id for user isolation
                owner_id = chunk.owner_id if hasattr(chunk, 'owner_id') and chunk.owner_id else ""
                tantivy_doc.add_text("owner_id", owner_id)

                metadata = chunk.metadata or {}
                tantivy_doc.add_json("metadata", metadata)

                # Dynamically add all string fields from metadata for filtering
                for key, value in metadata.items():
                    if isinstance(value, str) and key not in EXCLUDED_METADATA_FIELDS:
                        try:
                            tantivy_doc.add_text(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to add field '{key}' to tantivy document: {e}")
                
                self.processing_queue.put(tantivy_doc)

                added_ids.append(chunk_id)
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

            # Final progress logging if not already logged at the end
            if processed_count % self.config.progress_interval != 0:
                elapsed = time.time() - start_time
                stats = {
                    "processed": processed_count,
                    "total": total_docs,
                    "elapsed_sec": round(elapsed, 2),
                    "throughput_docs_sec": round(processed_count / elapsed, 2)
                }
                logger.info(f"[IndexProgress] Final: {json.dumps(stats, ensure_ascii=False)}")

            self.processing_queue.put(None)
            self._writer_thread.join()
            writer.commit()
            self._index.reload()
            
            tokenizer_info = self.tokenizer_manager.get_tokenizer_info()
            logger.info(f"Successfully built index with {len(added_ids)} chunks using {tokenizer_info} tokenizer")
            
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

    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Add chunks to the existing index (with ID deduplication)

        Args:
            chunks: List of Chunk objects to add
        Returns:
            List of chunk IDs that were successfully added to the index
        """
        if not chunks:
            logger.warning("No chunks provided for adding")
            return []

        # Check for duplicate IDs and filter
        unique_chunks = []
        duplicate_ids = []
        existing_ids = set()

        # Get existing chunk IDs - directly query from index
        existing_ids = set()
        if self._index is not None:
            try:
                searcher = self._index.searcher()
                # Query all chunk IDs (this can be optimized, but for simplicity, just query directly)
                for chunk in chunks:
                    query = self._index.parse_query(f'id:"{chunk.id}"', ["id"])
                    results = searcher.search(query, 1)
                    if results.hits:
                        existing_ids.add(chunk.id)
                logger.debug(f"Found {len(existing_ids)} existing chunk IDs in index")
            except Exception as e:
                logger.debug(f"Error getting existing chunk IDs: {e}")
                existing_ids = set()

        # Check for duplicate IDs in the list (including with existing chunks)
        seen_ids = set()
        for chunk in chunks:
            if chunk.id in seen_ids:
                # Duplicate within the list
                duplicate_ids.append(chunk.id)
                logger.warning(f"Duplicate chunk ID found: {chunk.id}. Use update_chunks() to update existing chunks.")
            elif chunk.id in existing_ids:
                # Duplicate with existing chunks
                duplicate_ids.append(chunk.id)
                logger.warning(f"Chunk with ID {chunk.id} already exists. Use update() to update existing chunks.")
            else:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)

        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate chunk IDs: {duplicate_ids}")

        if not unique_chunks:
            logger.warning("No unique chunks to add after deduplication")
            return []

        return self._build_index(unique_chunks)

    def save_index(self, index_path: str = None, index_name: str = "index") -> None:
        """Save index state (Tantivy automatically persists)

        Note: Tantivy automatically persists the index, so this method only logs the save location.
        The index_path and index_name parameters are ignored for compatibility.
        """
        self._ensure_index_loaded()
        logger.info(f"Index automatically saved at: {self.config.index_path}")

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build index from chunks (only when index doesn't exist)

        Args:
            chunks: List of Chunk objects to index

        Raises:
            RuntimeError: If index already exists
        """
        # Check if index already exists
        if self.index_exists():
            raise RuntimeError(
                "Index already exists. Use add() to add chunks to existing index, "
                "or delete the existing index first if you want to rebuild it."
            )

        self.from_chunks(chunks)

    def index_exists(self) -> bool:
        """Check if index exists

        Returns:
            bool: True if index exists and has chunks, False otherwise
        """
        try:
            # Check if index is initialized
            if self._index is None:
                return False

            # Check if index has chunks
            if hasattr(self._index, 'searcher'):
                searcher = self._index.searcher()
                # Query all chunk IDs
                from tantivy import Query
                all_query = Query.all_query()
                result = searcher.search(all_query, limit=1)
                return len(result.hits) > 0

            return False
        except Exception as e:
            logger.debug(f"Error checking index existence: {e}")
            return False

    def load_index(self, index_path: Optional[str] = None) -> None:
        """Load index from storage (alias for load_local)

        Args:
            index_path: Index path (ignored, uses configured path)

        Raises:
            FileNotFoundError: If index path does not exist
            RuntimeError: If index is already loaded
        """
        if index_path is not None and index_path != self.config.index_path:
            logger.debug(f"BM25 index uses configured path {self.config.index_path}, ignoring provided path {index_path}")
        self.load_local()

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update chunks in index

        Args:
            chunks: List of Chunk objects to update

        Returns:
            Optional[bool]: True if update successful, False otherwise, None if not implemented
        """
        if not chunks:
            return True

        try:
            chunk_ids = [str(chunk.id) for chunk in chunks if chunk.id is not None]
            if chunk_ids:
                logger.info(f"Update mode: attempting to delete {len(chunk_ids)} existing chunks")
                deleted_count = self._delete_chunks_by_ids(chunk_ids)
                logger.info(f"Update mode: successfully deleted {deleted_count} chunks")

                # Ensure index is reloaded after deletion for consistency
                if self._index is not None:
                    self._index.reload()

            self._build_index(chunks)
            return True
        except Exception as e:
            logger.error(f"Error updating chunks: {e}")
            return False

    def delete_index(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete chunks by IDs

        Args:
            ids: List of IDs to delete. If None, delete all. Default is None

        Returns:
            Optional[bool]: True if deletion successful, False otherwise, None if not implemented
        """
        if ids is None:
            try:
                if self._index is not None:
                    self._index = None
                    self._tokenizers_registered = False

                self._initialize_index()
                logger.info("Successfully deleted all chunks by recreating index")
                return True
            except Exception as e:
                logger.error(f"Error deleting all chunks: {e}")
                return False

        if not ids:
            return True

        unique_chunk_ids = list(set(ids))

        try:
            deleted_count = self._delete_chunks_by_ids(unique_chunk_ids)
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Retrieve chunks by their IDs

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of Chunk objects found

        Raises:
            RuntimeError: If index is not initialized
        """
        if not chunk_ids:
            return []
            
        self._ensure_index_loaded()
        chunks = []

        try:
            searcher = self._index.searcher()
            
            for chunk_id in chunk_ids:
                query = self._index.parse_query(f'id:"{chunk_id}"', ["id"])
                results = searcher.search(query, 1)

                if results.hits:
                    _, doc_address = results.hits[0]
                    tantivy_doc = searcher.doc(doc_address)
                    
                    doc_id_field = tantivy_doc.get_first("id") or ""
                    content_field = tantivy_doc.get_first("content") or ""
                    owner_id_field = tantivy_doc.get_first("owner_id") or ""
                    metadata_field = tantivy_doc.get_first("metadata") or {}

                    if isinstance(metadata_field, str):
                        try:
                            metadata_field = json.loads(metadata_field)
                        except json.JSONDecodeError:
                            metadata_field = {}

                    chunks.append(Chunk(
                        id=doc_id_field,
                        content=content_field,
                        owner_id=owner_id_field,
                        metadata=metadata_field
                    ))

            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by IDs {chunk_ids}: {e}")
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
        
        with os.scandir(self.config.index_path) as entries:
            has_files = any(entries)
        if not has_files:
            raise FileNotFoundError(f"Index directory is empty: {self.config.index_path}")
        
        try:
            # Load existing index without dynamic fields (they're already in the schema)
            self._initialize_index()

            self._set_tokenizer_from_existing_index()

            logger.info(f"Successfully loaded existing index from: {self.config.index_path}")
            return self
        except Exception as e:
            logger.error(f"Failed to load index from {self.config.index_path}: {e}")
            self.close()
            raise

    def from_chunks(self, chunks: List[Chunk]) -> "BM25IndexBuilder":
        """Build index from chunk list (only for initial creation)

        This method is intended for creating a new index from scratch.
        If you want to add chunks to an existing index, use add_chunks() instead.

        Args:
            chunks: List of Chunk objects to index

        Returns:
            Self (BM25IndexBuilder instance)

        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If index is already loaded (use add_chunks instead)
            Exception: If there's an error during index building
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        if self._index is not None:
            raise RuntimeError(
                "Index is already loaded. from_chunks() is only for initial index creation. "
                "To add chunks to existing index, use: builder.add_chunks(chunks)"
            )

        try:
            self._build_index(chunks)
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
    ) -> List[Chunk]:
        """执行搜索并返回列表"""
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

                chunk = Chunk(
                    id=tantivy_doc.get_first("id") or "",
                    content=tantivy_doc.get_first("content") or "",
                    owner_id=tantivy_doc.get_first("owner_id") or "",
                    metadata=metadata_field
                )

                results.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to parse chunk from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
        return results



    def get_vector_db_info(self) -> Dict[str, Any]:
        """Get vector database information

        Returns:
            Dictionary containing database info (size, dimensions, etc.)
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
            f"chunks={num_docs}, "
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
        """Context manager exit, clean up resources"""
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