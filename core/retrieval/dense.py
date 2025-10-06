from typing import Any, List, Dict, ClassVar, Collection, TYPE_CHECKING
import logging

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.schema import Chunk
from core.utils.retrieval_helper import RetrievalHelper

if TYPE_CHECKING:
    from config.core.retrieval.dense_config import DenseRetrieverConfig

logger = logging.getLogger(__name__)




class DenseRetriever(BaseRetriever):
    """
    Based on vector database for dense retrieval.

    Supports multiple search types: similarity search, threshold filtering, MMR diversity search
    """
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """Allowed search types"""
    
    def __init__(self, config: "DenseRetrieverConfig"):
        super().__init__(config)
        logger.info("DenseRetriever: Initializing embedding model...")
        # initialize embedding model
        if hasattr(config, 'index_config') and config.index_config is not None:
            if hasattr(config.index_config, 'embedding_config') and config.index_config.embedding_config is not None:
                self.embedding = config.index_config.embedding_config.build()
        logger.info("DenseRetriever: Embedding model initialized")

        logger.info("DenseRetriever: Building index...")
        self._index = self.config.index_config.build()
        logger.info("DenseRetriever: Index built, loading existing index...")
        self._load_existing_index()
        logger.info("DenseRetriever: Index loaded, ensuring initialization...")
        self._ensure_index_initialized()
        logger.info("DenseRetriever: Initialization complete")

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
    
    def get_index(self):
        """Get the vector index"""
        if self._index is None:
            raise ValueError("Index not initialized")
        return self._index

    def _ensure_index_initialized(self) -> None:
        """Ensure the index is initialized (built by IndexManager)"""
        # Check if the index exists
        if not hasattr(self, '_index') or self._index is None:
            raise RuntimeError("Index not initialized. Please use IndexManager to build the index first.")

        # Check if the index contains data
        if hasattr(self._index, 'index_exists') and not self._index.index_exists():
            raise RuntimeError("Index exists but contains no data. Please use IndexManager to build the index first.")

        logger.debug(f"Index initialized successfully for {self.get_name()}")

    def _validate_search_config(self, search_type: str, search_kwargs: Dict[str, Any]) -> None:
        """Validate search configuration

        Args:
            search_type: search type
            search_kwargs: search parameters

        Raises:
            ValueError: if search type is not in allowed types
            ValueError: if using similarity_score_threshold but no valid score_threshold specified
        """
        if search_type not in self.allowed_search_types:
            msg = (
                f"search_type '{search_type}' is not allowed. "
                f"Valid values are: {self.allowed_search_types}"
            )
            raise ValueError(msg)

        if search_type == "similarity_score_threshold":
            score_threshold = search_kwargs.get("score_threshold")
            if (score_threshold is None or
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "When using 'similarity_score_threshold' search type, "
                    "a valid score_threshold (float between 0 and 1) must be specified in search_kwargs"
                )
                raise ValueError(msg)

    def similarity_search(self, query: str, include_score: bool = False, **kwargs: Any) -> List[Chunk]:
        """Similarity search

        Args:
            query: query string
            include_score: whether to include similarity score in Chunk.metadata["score"]
            **kwargs: other search parameters

        Returns:
            list of chunks, if include_score=True, then score is stored in metadata["score"]
        """
        index = self.get_index()
        if index is None or not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []


        query_embedding = self.embedding.embed(query)
        return self.similarity_search_by_vector(query_embedding, include_score=include_score, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], include_score: bool = False, **kwargs: Any) -> List[Chunk]:
        """Vector similarity search

        Args:
            embedding: query embedding vector
            include_score: whether to include similarity score in Chunk.metadata["score"]
            **kwargs: other search parameters

        Returns:
            list of chunks, if include_score=True, then score is stored in metadata["score"]
        """
        index = self.get_index()
        if index is None:
            return []

        # Merge search parameters
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        search_kwargs["metric"] = self.config.metric

        # Execute FAISS search
        chunks_and_scores = RetrievalHelper.vector_search_with_faiss(index, embedding, search_kwargs)

        if include_score:
            # Add scores to chunks' metadata
            chunks = []
            for chunk, score in chunks_and_scores:
                # Create a copy of the chunk to avoid modifying the original
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={**chunk.metadata, "score": score}
                )
                chunks.append(chunk_copy)
            return chunks
        else:
            return [chunk for chunk, _ in chunks_and_scores]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Max marginal relevance search (diversity)"""
        index = self.get_index()
        if index is None:
            return []


        query_embedding = self.embedding.embed(query)

        # Merge search parameters
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        fetch_k = search_kwargs.get("fetch_k", 20)

        # Get candidate chunks (using internal method to get scores)
        chunks_and_scores = RetrievalHelper.vector_search_with_faiss(
            index, query_embedding, {**search_kwargs, "k": fetch_k, "metric": self.config.metric}
        )

        if not chunks_and_scores:
            return []

        # Prepare MMR search parameters
        search_kwargs["normalize_for_cosine"] = (
            (hasattr(index.config, 'normalize_L2') and index.config.normalize_L2) or
            (hasattr(index.config, 'metric') and index.config.metric == "cosine")
        )

        # Use MMR to select chunks
        return RetrievalHelper.mmr_search(
            query_embedding, chunks_and_scores, self.embedding, search_kwargs
        )
    
    def _get_relevant_chunks(
        self,
        query: str,
        search_type: str = "similarity",
        **kwargs: Any
    ) -> List[Chunk]:
        """Execute search and return relevant chunks"""
        # Merge search parameters
        search_kwargs = {**self.config.search_kwargs, **kwargs}

        # Get parameters
        k = search_kwargs.get("k", 5)
        with_score = search_kwargs.get("with_score", False)

        # Validate parameters
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}")

        # Validate search configuration
        self._validate_search_config(search_type, search_kwargs)
        
        if not query.strip():
            logger.info("Empty query, returning empty results")
            return []
        
        try:
            if search_type == "similarity":
                docs = self.similarity_search(query, include_score=with_score, **search_kwargs)

            elif search_type == "similarity_score_threshold":
                # Search with score threshold, always include score for threshold filtering
                docs = self.similarity_search(query, include_score=True, **search_kwargs)

                # Apply score threshold filtering
                score_threshold = search_kwargs.get("score_threshold")
                if score_threshold is not None:
                    filtered_docs = []
                    for doc in docs:
                        score = doc.metadata.get("score", 0.0)
                        if score >= score_threshold:
                            filtered_docs.append(doc)
                    docs = filtered_docs

                    if len(docs) == 0:
                        logger.warning(f"Using relevance score threshold {score_threshold} did not retrieve any relevant chunks")

                # If score is not needed, remove score information
                if not with_score:
                    for doc in docs:
                        if "score" in doc.metadata:
                            doc.metadata = {k: v for k, v in doc.metadata.items() if k != "score"}

            elif search_type == "mmr":
                docs = self.max_marginal_relevance_search(query, **search_kwargs)

                # If score is needed, re-get score information
                if with_score:
                    docs = self.similarity_search(query, include_score=True, k=len(docs), **search_kwargs)

            else:
                raise ValueError(f"Unsupported search type: {search_type}")

            logger.debug(f"Retrieved {len(docs)} chunks, search type: {search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"Error occurred while retrieving chunks: {e}")
            raise
