import logging
from typing import Any, List, Optional, Union, Tuple, cast, TYPE_CHECKING, Dict

from tantivy import Query, Occur, Order

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.schema import Chunk

if TYPE_CHECKING:
    from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig

logger = logging.getLogger(__name__)



class TantivyBM25Retriever(BaseRetriever):
    """
    TantivyBM25Retriever is a high-performance chunk retriever based on the Tantivy search engine.
    
    This class implements BM25 retrieval for chunk collections by leveraging Tantivy's capabilities,
    supporting dynamic filtering, phrase queries, and robust error handling.
    
    Key features:
    - Supports arbitrary metadata field filtering
    - Supports phrase queries for improved relevance
    - Robust exception handling and logging
    - Supports sorting fields and custom sort directions
    - Compatible with both synchronous and asynchronous operations
    
    Configuration parameters (from config):
        index_config (BM25IndexBuilderConfig): BM25 index configuration
        search_kwargs (dict): Additional search parameters including use_phrase_query and other options
        
    Runtime instance variables:
        index_builder: BM25IndexBuilder instance
        searcher: Tantivy searcher instance
        
    Core methods:
        - invoke: Main entry point for synchronous retrieval
        - _get_relevant_chunks: Execute search and return structured results
        - reload_searcher: Reload searcher to reflect latest index state
        
    Performance considerations:
        - Phrase queries provide better relevance but may be slower
        - Filtering increases search complexity, consider performance implications
        - Reloading searcher ensures index consistency
        
    Typical usage:
        >>> config = TantivyBM25RetrieverConfig(index_config=index_config)
        >>> retriever = config.build()
        >>> results = retriever.invoke("query statement")
        >>> results = retriever.invoke("query", filters={"category": "news", "author": "john"})
    """

    def __init__(self, config: "TantivyBM25RetrieverConfig"):
        logger.info("TantivyBM25Retriever: Initializing...")
        self.config = config
        logger.info("TantivyBM25Retriever: Building index...")
        self._index = self.config.index_config.build()
        logger.info("TantivyBM25Retriever: Index built, loading existing index...")
        self._load_existing_index()
        logger.info("TantivyBM25Retriever: Index loaded successfully")

        # Runtime instance variables
        self.searcher = None

    def _load_existing_index(self) -> None:
        """Try to load an existing index"""
        try:
            if hasattr(self._index, 'load_index'):
                # Check if the index has an index_path in its config
                if hasattr(self._index.config, 'index_path') and self._index.config.index_path:
                    self._index.load_index(self._index.config.index_path)
                else:
                    self._index.load_index()
                logger.info(f"Successfully loaded existing index for {self.get_name()}")
        except Exception as e:
            message = f"Index not found for retriever {self.get_name()}: {e}"
            logger.warning(f"{message}. Index will be empty until chunks are added.")
            # Don't raise an error, just continue with an empty index


    def reload_searcher(self) -> None:
        """Reload searcher to reflect latest index state

        This method should be called after index modifications to ensure
        the searcher reflects the latest index state.
        """
        try:
            index_instance = self.index
            self.searcher = index_instance.index.searcher()
            logger.debug("Searcher reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading searcher: {e}")
            raise

    def _build_filter_query(self, filters: Dict[str, Union[str, List[str]]]) -> List[Tuple[Occur, Query]]:
        """Build dynamic filter query supporting arbitrary fields
        
        Args:
            filters: Dictionary of field names and their values to filter by
            
        Returns:
            List of (Occur, Query) tuples for boolean query construction
        """
        filter_queries = []
        for field_name, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            if not values:
                continue
            try:
                index_instance = self.index
                q = Query.term_set_query(index_instance.index.schema, field_name, values)
                filter_queries.append((Occur.Must, q))
            except Exception as e:
                logger.warning(f"Skipping invalid filter field '{field_name}': {e}")
        return filter_queries

    def _build_main_query(self, query_tokens: List[str], use_phrase_query: bool = False) -> Query:
        """Build main query supporting normal BM25 or phrase queries
        
        Args:
            query_tokens: List of preprocessed query tokens
            
        Returns:
            Tantivy Query object
        """
        if not query_tokens:
            return Query.all_query()

        # Remove stopwords and empty/whitespace-only tokens
        index_instance = self.index
        stopwords = set(index_instance.tokenizer_manager.get_stopwords())
        filtered_tokens = [t for t in query_tokens if t not in stopwords and t.strip()]
        if not filtered_tokens:
            return Query.all_query()

        if use_phrase_query and len(filtered_tokens) > 1:
            # Use phrase query (order-sensitive, more precise)
            try:
                # Convert to the exact type required by phrase_query
                phrase_tokens: List[Union[str, Tuple[int, str]]] = cast(List[Union[str, Tuple[int, str]]], filtered_tokens)
                phrase_q = Query.phrase_query(index_instance.index.schema, "content_tokens", phrase_tokens)
                return phrase_q
            except Exception as e:
                logger.warning(f"Falling back to term query due to phrase query error: {e}")

        # Default: BM25 multi-term query with OR logic
        logger.info(f"Building query for tokens: {filtered_tokens}")
        
        # Use consistent term query approach for both single and multi-token queries
        # This ensures consistent behavior regardless of token count
        term_queries = []
        for token in filtered_tokens:
            try:
                term_q = Query.term_query(index_instance.index.schema, "content_tokens", token)
                term_queries.append((Occur.Should, term_q))
            except Exception as e:
                logger.warning(f"Failed to create term query for token '{token}': {e}")
        
        if not term_queries:
            return Query.all_query()
        
        if len(term_queries) == 1:
            # Single token query - extract the term query directly
            logger.info(f"Using single token query: '{filtered_tokens[0]}' on content_tokens field")
            return term_queries[0][1]  # Return the Query object without boolean wrapper
        else:
            # Multi-token OR query
            logger.info(f"Using OR query for tokens: {filtered_tokens} on content_tokens field")
            return Query.boolean_query(term_queries)

    def _get_relevant_chunks(
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
        """Execute search and return structured results

        Args:
            query: Query string
            k: Number of chunks to return (default from config)
            filters: Dictionary of field names and their values to filter by
            order_by_field: Field to sort by
            order_desc: Whether to sort in descending order
            with_score: Whether to include score in metadata (default from config)
            use_phrase_query: Whether to use phrase queries (default from config)
            **kwargs: Additional parameters (including owner_id for user isolation)

        Returns:
            List of Chunk objects
        """
        # Extract owner_id for user isolation
        owner_id = kwargs.pop('owner_id', None)
        owner_id_str = str(owner_id) if owner_id is not None else None

        # Use config defaults if parameters not provided
        k = k if k is not None else self.config.search_kwargs.get("k", 5)
        filters = filters or {}

        # Add owner_id to filters if provided
        if owner_id_str is not None:
            filters['owner_id'] = owner_id_str
            logger.debug(f"Added owner_id filter: {owner_id_str}")

        with_score = with_score if with_score is not None else self.config.search_kwargs.get("with_score", False)
        use_phrase_query = use_phrase_query if use_phrase_query is not None else self.config.search_kwargs.get("use_phrase_query", False)

        # Validate k parameter
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        if not query.strip():
            logger.info("Empty query received, returning empty results.")
            return []

        index_instance = self.index
        if index_instance.index is None:
            logger.warning("BM25 index is not loaded. Returning empty results.")
            return []

        # 1. Preprocess query
        try:
            query_tokens = index_instance.tokenizer_manager.get_current_tokenizer()(query)
            logger.debug(f"Query tokens: {query_tokens}")
        except Exception as e:
            logger.error(f"Error during query preprocessing: {e}")
            return []

        # 2. Build main query + filters
        main_query = self._build_main_query(query_tokens, use_phrase_query)
        filter_subqueries = self._build_filter_query(filters)

        final_query = (
            Query.boolean_query([(Occur.Must, main_query)] + filter_subqueries)
            if filter_subqueries else main_query
        )

        # 3. Calculate actual search k (expand search range in filter mode)
        search_k = k * 3 if filter_subqueries else k

        # 4. Execute search
        try:
            # Always create a fresh searcher to ensure we see the latest index state
            # This is critical after deletions to avoid serving stale results
            self.searcher = index_instance.index.searcher()
            order = Order.Desc if order_desc else Order.Asc
            search_result = self.searcher.search(
                final_query,
                limit=search_k,
                order_by_field=order_by_field,
                order=order
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

        # 5. Assemble results
        results = []
        for score, doc_address in search_result.hits[:k]:  # Truncate to k
            try:
                tantivy_doc = self.searcher.doc(doc_address)
                metadata = tantivy_doc.get_first("metadata") or {}
                
                # Enforce owner isolation even if index filter is bypassed unexpectedly
                doc_owner_id = tantivy_doc.get_first("owner_id") or ""
                if owner_id_str is not None and doc_owner_id != owner_id_str:
                    logger.warning(
                        f"Owner mismatch detected in BM25 results, skipping doc. "
                        f"expected={owner_id_str}, got={doc_owner_id}"
                    )
                    continue
                
                # Add score to metadata if with_score is True
                if with_score:
                    metadata = {**metadata, "score": float(score)}
                else:
                    # Ensure score is not included when with_score is False
                    metadata = {k: v for k, v in metadata.items() if k != "score"}
                
                chunk = Chunk(
                    id=tantivy_doc.get_first("id") or "",
                    content=tantivy_doc.get_first("content") or "",
                    owner_id=doc_owner_id,
                    metadata=metadata
                )

                results.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to parse chunk from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
        return results
