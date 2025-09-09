import logging
import warnings
from pydantic import Field, field_validator, ConfigDict
from typing import Any, List, Callable, Optional, Dict, Union, Tuple, cast, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from encapsulation.database.bm25_indexer import BM25IndexBuilder
from tantivy import Index, Query, Occur, Order

from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document

logger = logging.getLogger(__name__)

class TantivyBM25RetrieverConfig(BaseRetrieverConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["tantivy_bm25"] = "tantivy_bm25"
    
    # Runtime dependencies (injected by index builder)
    index: Optional["Index"] = Field(default=None, exclude=True, description="Tantivy index instance")
    preprocess_func: Optional[Callable[[str], List[str]]] = Field(default=None, exclude=True, description="Text preprocessing function")
    stopwords: Optional[List[str]] = Field(default=None, exclude=True, description="List of stopwords")
    
    # Retrieval parameters (from index configuration)
    search_kwargs: dict = Field(
        default_factory=lambda: {"use_phrase_query": False}, 
        exclude=True, 
        description="""Additional search parameters. Supported parameters:
        - use_phrase_query (bool): Whether to use phrase queries for better relevance (default: False)
        - k (int): Number of documents to return (overrides config default)
        - filters (dict): Dictionary of field names and their values to filter by
        - order_by_field (str): Field to sort by
        - order_desc (bool): Whether to sort in descending order (default: True)
        - with_score (bool): Whether to include score in metadata (overrides config default)
        """
    )

    def build(self) -> "TantivyBM25Retriever":
        """Build the TantivyBM25Retriever instance"""
        if self.index is None:
            raise ValueError("TantivyBM25RetrieverConfig.index must not be None. Please provide a valid tantivy.Index instance.")
        if not isinstance(self.index, Index):
            raise TypeError(f"Expected tantivy.Index, got {type(self.index).__name__}")

        if self.preprocess_func is None:
            raise ValueError("TantivyBM25RetrieverConfig.preprocess_func must not be None. Please provide a valid preprocessing function.")
        if not callable(self.preprocess_func):
            raise TypeError("preprocess_func must be callable")

        return TantivyBM25Retriever(config=self)


class TantivyBM25Retriever(BaseRetriever[TantivyBM25RetrieverConfig]):
    """
    TantivyBM25Retriever is a high-performance document retriever based on the Tantivy search engine.
    
    This class implements BM25 retrieval for document collections by leveraging Tantivy's capabilities,
    supporting dynamic filtering, phrase queries, and robust error handling.
    
    Key features:
    - Supports arbitrary metadata field filtering
    - Supports phrase queries for improved relevance
    - Robust exception handling and logging
    - Supports sorting fields and custom sort directions
    - Compatible with both synchronous and asynchronous operations
    
    Configuration parameters (from config):
        index (Index): Tantivy index instance
        preprocess_func (Callable): Text preprocessing function
        stopwords (List[str]): List of stopwords to filter out
        k (int): Default number of documents to return
        with_score (bool): Whether to include scores by default
        search_kwargs (dict): Additional search parameters including use_phrase_query and other options
        
    Runtime instance variables:
        searcher: Tantivy searcher instance
        
    Core methods:
        - invoke: Main entry point for synchronous retrieval
        - _get_relevant_documents: Execute search and return structured results
        - reload_searcher: Reload searcher to reflect latest index state
        
    Performance considerations:
        - Phrase queries provide better relevance but may be slower
        - Filtering increases search complexity, consider performance implications
        - Reloading searcher ensures index consistency
        
    Typical usage:
        >>> config = TantivyBM25RetrieverConfig(index=index, preprocess_func=preprocess_func)
        >>> retriever = config.build()
        >>> results = retriever.invoke("query statement")
        >>> results = retriever.invoke("query", filters={"category": "news", "author": "john"})
    """

    # Runtime instance variables
    searcher = None

    def _ensure_searcher(self):
        """Ensure searcher is initialized"""
        if self.searcher is None:
            self.searcher = self.config.index.searcher()

    def reload_searcher(self) -> None:
        """Reload searcher to reflect latest index state
        
        This method should be called after index modifications to ensure
        the searcher reflects the latest index state.
        """
        try:
            self.searcher = self.config.index.searcher()
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
                q = Query.term_set_query(self.config.index.schema, field_name, values)
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
        stopwords = set(self.config.stopwords or [])
        filtered_tokens = [t for t in query_tokens if t not in stopwords and t.strip()]
        if not filtered_tokens:
            return Query.all_query()

        if use_phrase_query and len(filtered_tokens) > 1:
            # Use phrase query (order-sensitive, more precise)
            try:
                # Convert to the exact type required by phrase_query
                phrase_tokens: List[Union[str, Tuple[int, str]]] = cast(List[Union[str, Tuple[int, str]]], filtered_tokens)
                phrase_q = Query.phrase_query(self.config.index.schema, "content_tokens", phrase_tokens)
                return phrase_q
            except Exception as e:
                logger.warning(f"Falling back to term query due to phrase query error: {e}")

        # Default: BM25 multi-term query
        query_str = " ".join(filtered_tokens)
        fields = ["content_tokens"]
        return self.config.index.parse_query(query_str, fields)

    def _get_relevant_documents(
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
        """Execute search and return structured results
        
        Args:
            query: Query string
            k: Number of documents to return (default from config)
            filters: Dictionary of field names and their values to filter by
            order_by_field: Field to sort by
            order_desc: Whether to sort in descending order
            with_score: Whether to include score in metadata (default from config)
            use_phrase_query: Whether to use phrase queries (default from config)
            **kwargs: Additional parameters
            
        Returns:
            List of Document objects
        """
        # Use config defaults if parameters not provided
        k = k if k is not None else self.config.k
        filters = filters or {}
        with_score = with_score if with_score is not None else self.config.with_score
        use_phrase_query = use_phrase_query if use_phrase_query is not None else self.config.search_kwargs.get("use_phrase_query", False)
        
        # Validate k parameter
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")
        
        # Merge additional search_kwargs from config
        merged_search_kwargs = {**self.config.search_kwargs, **kwargs}

        if not query.strip():
            logger.info("Empty query received, returning empty results.")
            return []

        # 1. Preprocess query
        try:
            query_tokens = self.config.preprocess_func(query)
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
            self._ensure_searcher()
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
                
                # Add score to metadata if with_score is True
                if with_score:
                    metadata = {**metadata, "score": float(score)}
                else:
                    # Ensure score is not included when with_score is False
                    metadata = {k: v for k, v in metadata.items() if k != "score"}
                
                document = Document(
                    id=tantivy_doc.get_first("id") or "",
                    content=tantivy_doc.get_first("content") or "",
                    metadata=metadata
                )

                results.append(document)
            except Exception as e:
                logger.warning(f"Failed to parse document from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results
