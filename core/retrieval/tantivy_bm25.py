import logging
import warnings
from typing import Any, List, Callable, Optional, Dict, Union, Tuple, Sequence, cast
from tantivy import Index, Query, Occur, Order

from core.retrieval.base import BaseRetriever
from core.utils.data_model import Document

logger = logging.getLogger(__name__)


class TantivyBM25Retriever(BaseRetriever):
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
    
    Main parameters:
        index (Index): Tantivy index instance
        preprocess_func (Callable): Text preprocessing function
        top_k (int): Default number of documents to return
        stopwords (List[str]): List of stopwords to filter out
        use_phrase_query (bool): Whether to enable phrase queries
        
    Core methods:
        - invoke: Main entry point for synchronous retrieval
        - _get_relevant_documents: Execute search and return structured results
        - get_relevant_documents_with_score: Get documents with their scores
        - reload_searcher: Reload searcher to reflect latest index state
        
    Performance considerations:
        - Phrase queries provide better relevance but may be slower
        - Filtering increases search complexity, consider performance implications
        - Reloading searcher ensures index consistency
        
    Typical usage:
        >>> retriever = TantivyBM25Retriever(index, preprocess_func)
        >>> results = retriever.invoke("query statement")
        >>> results = retriever.invoke("query", filters={"source": "news"})
        
    Attributes:
        index: Tantivy index instance
        preprocess_func: Text preprocessing function
        top_k: Number of documents to return
        stopwords: Set of stopwords to filter out
        use_phrase_query: Whether phrase queries are enabled
        searcher: Tantivy searcher instance
    """

    def __init__(
        self,
        index: Index,
        preprocess_func: Callable[[str], List[str]],
        top_k: int = 10,
        stopwords: Optional[List[str]] = None,
        use_phrase_query: bool = False,
        **kwargs: Any
    ):
        """Initialize Tantivy BM25 Retriever
        
        Args:
            index: Tantivy index instance
            preprocess_func: Text preprocessing function
            top_k: Default number of documents to return, must be greater than 0
            stopwords: List of stopwords to filter out
            use_phrase_query: Whether to enable phrase queries
            **kwargs: Additional parameters
        """
        if not isinstance(index, Index):
            raise TypeError(f"Expected tantivy.Index type, but got {type(index)}")
        
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, but got {top_k}")
            
        if not callable(preprocess_func):
            raise ValueError("preprocess_func must be a callable function")

        self.index = index
        self.preprocess_func = preprocess_func
        self.top_k = top_k
        self.stopwords = set(stopwords or [])
        self.use_phrase_query = use_phrase_query
        self.searcher = self.index.searcher()

    def reload_searcher(self) -> None:
        """Reload searcher to reflect latest index state
        
        This method should be called after index modifications to ensure
        the searcher reflects the latest index state.
        """
        try:
            self.searcher = self.index.searcher()
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
                q = Query.term_set_query(self.index.schema, field_name, values)
                filter_queries.append((Occur.Must, q))
            except Exception as e:
                logger.warning(f"Skipping invalid filter field '{field_name}': {e}")
        return filter_queries

    def _build_main_query(self, query_tokens: List[str]) -> Query:
        """Build main query supporting normal BM25 or phrase queries
        
        Args:
            query_tokens: List of preprocessed query tokens
            
        Returns:
            Tantivy Query object
        """
        if not query_tokens:
            return Query.all_query()

        # Remove stopwords (optional)
        filtered_tokens = [t for t in query_tokens if t not in self.stopwords]
        if not filtered_tokens:
            return Query.all_query()

        if self.use_phrase_query and len(filtered_tokens) > 1:
            # Use phrase query (order-sensitive, more precise)
            try:
                # Convert to the exact type required by phrase_query
                phrase_tokens: List[Union[str, Tuple[int, str]]] = cast(List[Union[str, Tuple[int, str]]], filtered_tokens)
                phrase_q = Query.phrase_query(self.index.schema, "content_tokens", phrase_tokens)
                return phrase_q
            except Exception as e:
                logger.warning(f"Falling back to term query due to phrase query error: {e}")

        # Default: BM25 multi-term query
        query_str = " ".join(filtered_tokens)
        fields = ["content_tokens"]
        return self.index.parse_query(query_str, fields)

    def _get_relevant_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        order_by_field: Optional[str] = None,
        order_desc: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Execute search and return structured results
        
        Args:
            query: Query string
            top_k: Number of documents to return
            filters: Dictionary of field names and their values to filter by
            order_by_field: Field to sort by
            order_desc: Whether to sort in descending order
            **kwargs: Additional parameters
            
        Returns:
            List of dictionaries containing document data and scores
        """
        top_k = top_k or self.top_k
        filters = filters or {}

        if not query.strip():
            logger.info("Empty query received, returning empty results.")
            return []

        # 1. Preprocess query
        try:
            query_tokens = self.preprocess_func(query)
            logger.debug(f"Query tokens: {query_tokens}")
        except Exception as e:
            logger.error(f"Error during query preprocessing: {e}")
            return []

        # 2. Build main query + filters
        main_query = self._build_main_query(query_tokens)
        filter_subqueries = self._build_filter_query(filters)

        final_query = (
            Query.boolean_query([(Occur.Must, main_query)] + filter_subqueries)
            if filter_subqueries else main_query
        )

        # 3. Calculate actual search top_k (expand search range in filter mode)
        search_top_k = top_k * 3 if filter_subqueries else top_k

        # 4. Execute search
        try:
            order = Order.Desc if order_desc else Order.Asc
            search_result = self.searcher.search(
                final_query,
                limit=search_top_k,
                order_by_field=order_by_field,
                order=order
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

        # 5. Assemble results
        results = []
        for score, doc_address in search_result.hits[:top_k]:  # Truncate to top_k
            try:
                tantivy_doc = self.searcher.doc(doc_address)
                doc_data = {
                    "id": tantivy_doc.get_first("id") or "",
                    "content": tantivy_doc.get_first("content") or "",
                    "metadata": tantivy_doc.get_first("metadata") or {},
                    "score": float(score),
                }

                results.append(doc_data)
            except Exception as e:
                logger.warning(f"Failed to parse document from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results

    def invoke(self, query: str, **kwargs: Any) -> List[Document]:
        """Invoke retriever to get relevant documents
        
        Main entry point for synchronous retriever invocation.
        
        Args:
            query: Query string
            **kwargs: Other parameters passed to the retriever
                top_k: Number of documents to return
                filters: Dictionary of field names and their values to filter by
                order_by_field: Field to sort by
                order_desc: Whether to sort in descending order
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> retriever.invoke("query")
            >>> retriever.invoke("query", filters={"source": "news"})
        """
        raw_results = self._get_relevant_documents(query, **kwargs)
        return [
            Document(
                id=res["id"],
                content=res["content"],
                metadata={**res.get("metadata", {}), "score": res["score"]}
            )
            for res in raw_results
        ]

    def get_relevant_documents_with_score(self, query: str, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Get documents and their raw scores as tuples
        
        Args:
            query: Query string
            **kwargs: Other parameters passed to the retriever
            
        Returns:
            List of (Document, score) tuples
        """
        raw_results = self._get_relevant_documents(query, **kwargs)
        return [
            (
                Document(id=res["id"], content=res["content"], metadata=res.get("metadata", {})),
                res["score"]
            )
            for res in raw_results
        ]

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get retriever configuration information
        
        Returns:
            Dictionary containing retriever configuration information
        """
        return {
            "top_k": self.top_k,
            "use_phrase_query": self.use_phrase_query,
            "stopwords_count": len(self.stopwords),
            "preprocess_func": self.preprocess_func.__name__ if hasattr(self.preprocess_func, '__name__') else str(self.preprocess_func)
        }

    def update_top_k(self, top_k: int) -> None:
        """Update number of documents to return
        
        Args:
            top_k: New number of documents to return, must be greater than 0
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, but got {top_k}")
        self.top_k = top_k

    def __repr__(self) -> str:
        """String representation of the TantivyBM25Retriever instance"""
        return (
            f"{self.__class__.__name__}(top_k={self.top_k}, "
            f"use_phrase={self.use_phrase_query})"
        )