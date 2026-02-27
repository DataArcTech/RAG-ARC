import logging
import os
import queue
import threading
from typing import Any, Dict, List, Optional

import psutil

from core.utils.path_guard import ensure_writable_dir
from encapsulation.data_model.schema import Chunk
from encapsulation.database.bm25_indexer_constants import EXCLUDED_METADATA_FIELDS
from encapsulation.database.bm25_indexer_executor import _BM25IndexBuilderExecutorMixin, init_jieba_worker
from encapsulation.database.bm25_indexer_indexing import _BM25IndexBuilderIndexingMixin
from encapsulation.database.bm25_indexer_query import _BM25IndexBuilderQueryMixin
from encapsulation.database.bm25_indexer_schema import _BM25IndexBuilderSchemaMixin
from encapsulation.database.bm25_indexer_tantivy import Index
from encapsulation.database.bm25_indexer_tokenization import _BM25IndexBuilderTokenizationMixin
from encapsulation.database.utils.TokenizerManager import TokenizerManager
from framework.shared_module_decorator import shared_module
from framework.virtual_paths import is_io_path

logger = logging.getLogger(__name__)


@shared_module
class BM25IndexBuilder(
    _BM25IndexBuilderExecutorMixin,
    _BM25IndexBuilderTokenizationMixin,
    _BM25IndexBuilderSchemaMixin,
    _BM25IndexBuilderIndexingMixin,
    _BM25IndexBuilderQueryMixin,
):
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
        if is_io_path(getattr(self.config, "index_path", "")):
            try:
                from framework.register import Register

                io_manager = Register().get_object("io_manager")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("io_manager unavailable while resolving io:// index_path") from exc
            self.config.index_path = str(io_manager.resolve_local_dir(str(self.config.index_path), ensure=True))
        runtime_root = os.getenv("RAGARC_RUNTIME_DIR", "io://runtime")
        fallback = os.path.join(runtime_root, os.path.basename(self.config.index_path) or "bm25_index")
        self.config.index_path = ensure_writable_dir(self.config.index_path, fallback)

        # Runtime instance variables (initialized when needed)
        self._index: Optional[Index] = None
        self._schema = None
        self._writer_heap_size: int = 0
        self._tokenizers_registered: bool = False
        self._executor = None
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
                custom_stopwords_file=self.config.stopwords_file,
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
