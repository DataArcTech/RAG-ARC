from typing import Any, List, Dict, ClassVar, Collection, TYPE_CHECKING, Optional
import logging
import os

from core.retrieval.base import BaseRetriever
from encapsulation.data_model.schema import Chunk
from core.utils.retrieval_helper import RetrievalHelper
from core.utils.owner_guard import normalize_owner_id, is_admin_owner

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

        self._ensure_index_initialized()
        logger.info("DenseRetriever: Initialization complete")


    def _ensure_index_initialized(self) -> None:
        """Ensure the index is initialized (built by IndexManager)"""
        # Check if the index exists
        if not hasattr(self, '_index') or self._index is None:
            raise RuntimeError("Index not initialized. Please use IndexManager to build the index first.")

        # Check if the index contains data
        if hasattr(self._index, 'index_exists') and not self._index.index_exists():
            raise RuntimeError("Index exists but contains no data. Please use IndexManager to build the index first.")

        logger.debug(f"Index initialized successfully for {self.get_name()}")

    @staticmethod
    def _chunk_matches_owner(chunk: Chunk, owner_key: str) -> bool:
        """
        Return True when the chunk belongs to the requested owner.

        owner_key is already normalized (canonical string form).
        """
        if not owner_key:
            return False

        chunk_owner: Optional[Any] = getattr(chunk, "owner_id", None)
        if chunk_owner is None and getattr(chunk, "metadata", None):
            chunk_owner = chunk.metadata.get("owner_id")

        if chunk_owner is None:
            return False

        normalized_chunk_owner = normalize_owner_id(chunk_owner)
        return normalized_chunk_owner == owner_key

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
            **kwargs: other search parameters (including owner_id for user isolation)

        Returns:
            list of chunks, if include_score=True, then score is stored in metadata["score"]
        """
        if self._index is None or not hasattr(self._index, 'index') or self._index.index is None or self._index.index.ntotal == 0:
            return []


        query_embedding = self._embed_query_aligned(query)
        return self.similarity_search_by_vector(query_embedding, include_score=include_score, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], include_score: bool = False, **kwargs: Any) -> List[Chunk]:
        """Vector similarity search

        Args:
            embedding: query embedding vector
            include_score: whether to include similarity score in Chunk.metadata["score"]
            **kwargs: other search parameters (including owner_id for user isolation)

        Returns:
            list of chunks, if include_score=True, then score is stored in metadata["score"]
        """
        if self._index is None:
            return []

        # Extract owner_id for filtering
        owner_id = kwargs.pop('owner_id', None)
        if owner_id is None:
            logger.warning("Dense retrieval requires an owner_id; returning no results")
            return []

        owner_key = normalize_owner_id(owner_id)
        if owner_key is None:
            logger.warning("owner_id '%s' could not be normalized; returning no results", owner_id)
            return []

        global_scope = is_admin_owner(owner_id)
        if global_scope:
            owner_key = None

        # Merge search parameters
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        search_kwargs["metric"] = self.config.metric

        # If owner_id filtering is needed, over-fetch to ensure we get enough results
        original_k = search_kwargs.get("k", 5)
        if owner_key is not None:
            # Over-fetch multiplier: fetch more results to compensate for filtering
            # Default to 3x, can be configured via over_fetch_multiplier parameter
            over_fetch_multiplier = kwargs.pop('over_fetch_multiplier', 3)
            search_kwargs["k"] = min(original_k * over_fetch_multiplier, self._index.index.ntotal if self._index.index else original_k)
            logger.debug(f"Over-fetching {search_kwargs['k']} results (original k={original_k}, multiplier={over_fetch_multiplier})")

        # Execute FAISS search
        chunks_and_scores = RetrievalHelper.vector_search_with_faiss(self._index, embedding, search_kwargs)

        # Filter by owner_id if provided (user isolation)
        if owner_key is not None:
            before_filter_count = len(chunks_and_scores)
            chunks_and_scores = [
                (chunk, score) for chunk, score in chunks_and_scores
                if self._chunk_matches_owner(chunk, owner_key)
            ]
            after_filter_count = len(chunks_and_scores)
            logger.debug(f"Filtered results by owner_id={owner_key}: {after_filter_count}/{before_filter_count} chunks")

            # Trim to original k if we have more than needed
            if len(chunks_and_scores) > original_k:
                chunks_and_scores = chunks_and_scores[:original_k]
                logger.debug(f"Trimmed to {original_k} chunks")
            elif len(chunks_and_scores) < original_k:
                logger.warning(f"Only retrieved {len(chunks_and_scores)} chunks for owner_id={owner_key}, requested {original_k}")


        if include_score:
            # Add scores to chunks' metadata
            chunks = []
            for chunk, score in chunks_and_scores:
                # Create a copy of the chunk to avoid modifying the original
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    owner_id=chunk.owner_id,
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
        if self._index is None:
            return []


        query_embedding = self._embed_query_aligned(query)

        # Extract owner_id for filtering
        owner_id = kwargs.pop('owner_id', None)
        if owner_id is None:
            logger.warning("Dense retrieval requires an owner_id; returning no results")
            return []

        owner_key = normalize_owner_id(owner_id)
        if owner_key is None:
            logger.warning("owner_id '%s' could not be normalized; returning no results", owner_id)
            return []

        global_scope = is_admin_owner(owner_id)
        if global_scope:
            owner_key = None

        # Merge search parameters
        search_kwargs = {**self.config.search_kwargs, **kwargs}
        fetch_k = search_kwargs.get("fetch_k", 20)

        # If owner_id filtering is needed, over-fetch to ensure we get enough candidates
        if owner_key is not None:
            # Over-fetch multiplier for MMR (need more candidates for diversity)
            over_fetch_multiplier = kwargs.pop('over_fetch_multiplier', 3)
            original_fetch_k = fetch_k
            fetch_k = min(fetch_k * over_fetch_multiplier, self._index.index.ntotal if self._index.index else fetch_k)
            logger.debug(f"MMR over-fetching {fetch_k} candidates (original fetch_k={original_fetch_k}, multiplier={over_fetch_multiplier})")

        # Get candidate chunks (using internal method to get scores)
        chunks_and_scores = RetrievalHelper.vector_search_with_faiss(
            self._index, query_embedding, {**search_kwargs, "k": fetch_k, "metric": self.config.metric}
        )

        if not chunks_and_scores:
            return []

        # Filter by owner_id if provided (user isolation)
        if owner_key is not None:
            before_filter_count = len(chunks_and_scores)
            chunks_and_scores = [
                (chunk, score) for chunk, score in chunks_and_scores
                if self._chunk_matches_owner(chunk, owner_key)
            ]
            after_filter_count = len(chunks_and_scores)
            logger.debug(f"MMR filtered results by owner_id={owner_key}: {after_filter_count}/{before_filter_count} chunks")

            if after_filter_count == 0:
                logger.warning(f"No chunks found for owner_id={owner_key} after filtering")
                return []
            elif after_filter_count < search_kwargs.get("k", 4):
                logger.warning(f"Only {after_filter_count} chunks available for MMR, requested {search_kwargs.get('k', 4)}")

        # Prepare MMR search parameters
        search_kwargs["normalize_for_cosine"] = (
            (hasattr(self._index.config, 'normalize_L2') and self._index.config.normalize_L2) or
            (hasattr(self._index.config, 'metric') and self._index.config.metric == "cosine")
        )

        # Use MMR to select chunks
        return RetrievalHelper.mmr_search(
            query_embedding, chunks_and_scores, self.embedding, search_kwargs
        )

    def _embed_query_aligned(self, query: str) -> List[float]:
        """
        Embed query text while keeping vector dimensionality aligned with the loaded FAISS index.

        If an index exists and embedding dimensionality mismatches, try to auto-align by setting
        OpenAI `dimensions` (text-embedding-3-* only) when embedding_dimensions is not explicitly configured.
        """
        query_embedding = self.embedding.embed(query)
        if not isinstance(query_embedding, list):
            return query_embedding

        override_supported = False
        if hasattr(self.embedding, "supports_dimension_override"):
            try:
                override_supported = bool(self.embedding.supports_dimension_override())
            except Exception:  # noqa: BLE001
                override_supported = False

        expected_dim = None
        try:
            if self._index is not None and getattr(self._index, "index", None) is not None:
                expected_dim = int(getattr(self._index.index, "d", 0) or 0)
        except Exception:  # noqa: BLE001
            expected_dim = None

        if not expected_dim or len(query_embedding) == expected_dim:
            return query_embedding

        allow_auto = os.getenv("AUTO_ALIGN_EMBEDDING_DIM", "1") == "1"
        if not allow_auto:
            raise RuntimeError(f"Embedding dimension mismatch: got={len(query_embedding)} expected={expected_dim}")

        # Only align when embedding config didn't explicitly set embedding_dimensions.
        try:
            cfg = getattr(self.config.index_config, "embedding_config", None)
            configured = getattr(cfg, "embedding_dimensions", None) if cfg is not None else None
        except Exception:  # noqa: BLE001
            configured = None

        if configured is not None and override_supported:
            raise RuntimeError(
                f"Embedding dimension mismatch: got={len(query_embedding)} expected={expected_dim} "
                f"(explicit embedding_dimensions={configured} is set)"
            )

        if configured is not None and not override_supported:
            raise RuntimeError(
                f"Embedding dimension mismatch: got={len(query_embedding)} expected={expected_dim}. "
                f"Your embedding provider does not support forcing dimensions at runtime "
                f"(EMBEDDING_DIMENSIONS={configured} is treated as a hint). Rebuild the index or switch embedding model."
            )

        if hasattr(self.embedding, "set_embedding_dimensions"):
            try:
                if self.embedding.set_embedding_dimensions(expected_dim):
                    query_embedding = self.embedding.embed(query)
            except Exception:  # noqa: BLE001
                pass

        if not isinstance(query_embedding, list) or len(query_embedding) != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: got={len(query_embedding) if isinstance(query_embedding, list) else 'unknown'} "
                f"expected={expected_dim}. Rebuild the FAISS index or use a compatible embedding model."
            )
        return query_embedding
    
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
