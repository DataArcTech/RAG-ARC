import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from encapsulation.data_model.schema import Chunk
from encapsulation.database.bm25_indexer_tantivy import Occur, Order, Query

logger = logging.getLogger(__name__)


class _BM25IndexBuilderQueryMixin:
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
            if hasattr(self._index, "searcher"):
                searcher = self._index.searcher()
                # Query all chunk IDs
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

                    chunks.append(
                        Chunk(
                            id=doc_id_field,
                            content=content_field,
                            owner_id=owner_id_field,
                            metadata=metadata_field,
                        )
                    )

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
        **kwargs: Any,
    ) -> List[Chunk]:
        """执行搜索并返回列表"""

        # Use config defaults if parameters not provided
        k = k if k is not None else self.config.k
        filters = filters or {}
        with_score = with_score if with_score is not None else self.config.with_score
        use_phrase_query = (
            use_phrase_query
            if use_phrase_query is not None
            else self.config.search_kwargs.get("use_phrase_query", False)
        )

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
            phrase_query = " ".join(query_tokens)
            main_query = self._index.parse_query(f'content_tokens:"{phrase_query}"', ["content_tokens"])
        else:
            # Use standard BM25 query on content_tokens field for proper tokenization
            query_str = " ".join(query_tokens)
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
        final_query = Query.boolean_query([(Occur.Must, main_query)] + filter_subqueries) if filter_subqueries else main_query

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
                order=order,
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
                    metadata=metadata_field,
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
            "executor_active": self._executor is not None and not self._executor_closed,
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
        except Exception:
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

