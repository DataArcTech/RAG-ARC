from typing import List, Optional, Dict, Any, Tuple, Literal, Union, cast, TYPE_CHECKING
import re
import numpy as np
import logging

from .base import AbstractChunker

if TYPE_CHECKING:
    from config.core.file_management.chunker.chunker_config import SemanticChunkerConfig

logger = logging.getLogger(__name__)


BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}
Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]



class SemanticChunker(AbstractChunker):
    """
    SemanticChunker is a text chunker that uses semantic similarity analysis to create meaningful text chunks.

    This class implements advanced chunking based on semantic coherence by analyzing the similarity between sentences using embeddings
    to determine natural breakpoints, creating chunks based on topic coherence rather than arbitrary size limits.

    Key features:
    - Embedding-based semantic similarity analysis for intelligent breakpoint detection
    - Multiple statistical threshold methods for breakpoint determination
    - Adaptive chunking that respects semantic boundaries
    - Sentence-level granularity with configurable context buffering
    - Support for both automatic and fixed chunk count modes
    - Minimum chunk size filtering to avoid overly fragmented results
    - Cosine distance calculation with optimized similarity computation
    - Configurable sentence splitting patterns for different languages

    Main parameters (from config):
        embedding: Embedding model configuration for generating sentence vectors
        buffer_size (int): Context window size around each sentence, defaults to 1
        breakpoint_threshold_type (str): Method for determining breakpoints, defaults to 'percentile'
        breakpoint_threshold_amount (float): Threshold value for breakpoint detection, optional
        number_of_chunks (int): Fixed number of chunks to create, optional (overrides threshold)
        sentence_split_regex (str): Pattern for sentence splitting, defaults to r"(?<=[.?!])\s+"
        min_chunk_size (int): Minimum chunk size filter, optional
        add_start_index (bool): Whether to track character positions, defaults to False

    Core methods:
        - chunk_text: Main chunking method using semantic analysis
        - get_chunker_info: Get chunker configuration information
        - _split_text_semantically: Execute semantic analysis and breakpoint detection
        - _calculate_sentence_distances: Compute semantic distances between sentences
        - _calculate_breakpoint_threshold: Determine optimal breakpoint threshold

    Performance considerations:
        - Requires embedding model for sentence vectorization (higher computational cost)
        - Semantic analysis provides superior content coherence but slower processing
        - Buffer size affects context quality vs. processing time trade-off
        - Suitable for high-quality chunking where content meaning matters most
        - For large documents, consider balancing chunk count with processing time
        - Embedding model choice significantly impacts quality and speed

    Typical usage:
        >>> # Embedding config should be provided in chunker config
        >>> config = SemanticChunkerConfig(
        ...     embedding=embedding_config,
        ...     breakpoint_threshold_type='percentile',
        ...     breakpoint_threshold_amount=90
        ... )
        >>> chunker = SemanticChunker(config=config)
        >>> chunks = chunker.chunk_text("long text with multiple topics...")
        >>> info = chunker.get_chunker_info()

    Breakpoint strategies:
        - Percentile: Use statistical percentile of distance distribution
        - Standard deviation: Use mean + N * std deviation as threshold
        - Interquartile: Use IQR-based outlier detection
        - Gradient: Analyze gradient changes in distance sequence

    Dependencies:
        - numpy: For numerical computations and statistical analysis
        - Embedding model: For generating sentence vectors
        - Optional: simsimd for optimized similarity calculations
    """

    def __init__(self, config: "SemanticChunkerConfig"):
        """Initialize SemanticChunker with embedding model"""
        super().__init__(config)
        # Build embeddings immediately since we always need it
        embedding_config = getattr(self.config, 'embedding', None)
        if embedding_config is None:
            raise ValueError("SemanticChunker requires an embedding config in config")
        self.embeddings = embedding_config.build()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantically coherent chunks.

        This method analyzes the semantic relationships between sentences in the text to identify
        natural topic boundaries. It uses embedding-based similarity analysis to determine where
        content topics change significantly, creating chunks that maintain semantic coherence.

        Args:
            text: Text content to be chunked
            metadata: Optional source metadata, will be passed to each chunk
            **kwargs: Runtime parameter overrides
                buffer_size (int): Context window size, overrides config value
                breakpoint_threshold_type (str): Threshold method, overrides config value
                Other semantic chunking parameters

        Returns:
            List of chunk dictionaries, each containing:
            - content: Text content of the chunk (semantically coherent)
            - metadata: Chunk-specific metadata
                - chunk_id: Unique identifier for the chunk
                - chunk_index: Index of chunk in sequence
                - start_idx: Starting position in original text
                - end_idx: Ending position in original text
                - character_count: Character count of this chunk
                - strategy: Chunking strategy identifier ('semantic')
            - source_metadata: Original document metadata (if provided)

        Raises:
            ValueError: If embedding configuration is missing or invalid
            Exception: If semantic analysis or chunking process fails
        """
        # Get config parameters
        buffer_size = getattr(self.config, 'buffer_size', 1)
        breakpoint_threshold_type = getattr(self.config, 'breakpoint_threshold_type', 'percentile')
        breakpoint_threshold_amount = getattr(self.config, 'breakpoint_threshold_amount', None)
        number_of_chunks = getattr(self.config, 'number_of_chunks', None)
        sentence_split_regex = getattr(self.config, 'sentence_split_regex', r"(?<=[.?!])\s+")
        min_chunk_size = getattr(self.config, 'min_chunk_size', None)
        add_start_index = getattr(self.config, 'add_start_index', False)

        # Allow runtime parameter overrides
        buffer_size = kwargs.get('buffer_size', buffer_size)
        breakpoint_threshold_type = kwargs.get('breakpoint_threshold_type', breakpoint_threshold_type)

        try:
            embeddings = self.embeddings

            chunks = self._split_text_semantically(
                text, embeddings, buffer_size, breakpoint_threshold_type,
                breakpoint_threshold_amount, number_of_chunks, sentence_split_regex, min_chunk_size
            )

            # Convert to standardized format
            result = []
            current_pos = 0

            for i, chunk_content in enumerate(chunks):
                start_idx = text.find(chunk_content, current_pos) if not add_start_index else current_pos
                if start_idx == -1:
                    start_idx = current_pos

                end_idx = start_idx + len(chunk_content)
                current_pos = end_idx

                chunk_dict = {
                    'content': chunk_content,
                    'metadata': {
                        'chunk_id': i,
                        'chunk_index': i,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'character_count': len(chunk_content),
                        'strategy': 'semantic'
                    }
                }

                # Add source metadata if provided
                if metadata:
                    chunk_dict['source_metadata'] = metadata.copy()

                result.append(chunk_dict)

            logger.info(f"Split text into {len(result)} semantic chunks")
            return result

        except Exception as e:
            logger.error(f"Failed to chunk text with semantic strategy: {str(e)}")
            raise

    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about this chunker's configuration and capabilities.

        Returns:
            Dictionary containing detailed chunker information:
            - strategy: Chunking strategy type ('semantic')
            - buffer_size: Context window size for sentence analysis
            - breakpoint_threshold_type: Method used for breakpoint detection
            - breakpoint_threshold_amount: Threshold value (if configured)
            - number_of_chunks: Fixed chunk count (if configured)
            - sentence_split_regex: Pattern used for sentence splitting
            - min_chunk_size: Minimum chunk size filter
            - add_start_index: Whether character position tracking is enabled
            - supported_features: List of supported feature capabilities
            - parameters: Complete copy of current configuration parameters
        """
        buffer_size = getattr(self.config, 'buffer_size', 1)
        breakpoint_threshold_type = getattr(self.config, 'breakpoint_threshold_type', 'percentile')
        breakpoint_threshold_amount = getattr(self.config, 'breakpoint_threshold_amount', None)
        number_of_chunks = getattr(self.config, 'number_of_chunks', None)
        sentence_split_regex = getattr(self.config, 'sentence_split_regex', r"(?<=[.?!])\s+")
        min_chunk_size = getattr(self.config, 'min_chunk_size', None)
        add_start_index = getattr(self.config, 'add_start_index', False)

        return {
            'strategy': 'semantic',
            'buffer_size': buffer_size,
            'breakpoint_threshold_type': breakpoint_threshold_type,
            'breakpoint_threshold_amount': breakpoint_threshold_amount,
            'number_of_chunks': number_of_chunks,
            'sentence_split_regex': sentence_split_regex,
            'min_chunk_size': min_chunk_size,
            'add_start_index': add_start_index,
            'supported_features': ['semantic_boundaries', 'embedding_based', 'adaptive_chunking'],
            'parameters': {
                'buffer_size': buffer_size,
                'breakpoint_threshold_type': breakpoint_threshold_type,
                'breakpoint_threshold_amount': breakpoint_threshold_amount,
                'number_of_chunks': number_of_chunks,
                'sentence_split_regex': sentence_split_regex,
                'min_chunk_size': min_chunk_size,
                'add_start_index': add_start_index
            }
        }

    def _split_text_semantically(
        self,
        text: str,
        embeddings,
        buffer_size: int,
        breakpoint_threshold_type: BreakpointThresholdType,
        breakpoint_threshold_amount: Optional[float],
        number_of_chunks: Optional[int],
        sentence_split_regex: str,
        min_chunk_size: Optional[int]
    ) -> List[str]:
        """
        Split text using semantic similarity analysis.

        This method implements the core semantic chunking algorithm by analyzing sentence-level
        semantic similarity and identifying natural breakpoints based on statistical thresholds.

        Args:
            text: Text to be analyzed and split
            embeddings: Embedding model for sentence vectorization
            buffer_size: Context window for sentence combination
            breakpoint_threshold_type: Statistical method for threshold determination
            breakpoint_threshold_amount: Specific threshold value (optional)
            number_of_chunks: Target number of chunks (optional)
            sentence_split_regex: Pattern for sentence boundary detection
            min_chunk_size: Minimum chunk size filter

        Returns:
            List of semantically coherent text chunks
        """
        # Splitting the essay (by default on '.', '?', and '!')
        single_sentences_list = re.split(sentence_split_regex, text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        # similarly, the following np.gradient would fail
        if (
            breakpoint_threshold_type == "gradient"
            and len(single_sentences_list) == 2
        ):
            return single_sentences_list

        distances, sentences = self._calculate_sentence_distances(single_sentences_list, embeddings, buffer_size)

        if number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances, number_of_chunks)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances, breakpoint_threshold_type, breakpoint_threshold_amount)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])

            if (
                min_chunk_size is not None
                and len(combined_text) < min_chunk_size
            ):
                continue
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str], embeddings, buffer_size: int
    ) -> Tuple[List[float], List[dict]]:
        """
        Calculate semantic distances between consecutive sentences.

        This method processes sentences with context buffering, generates embeddings,
        and computes cosine distances to measure semantic similarity changes.

        Args:
            single_sentences_list: List of individual sentences
            embeddings: Embedding model for vectorization
            buffer_size: Context window size for sentence combination

        Returns:
            Tuple of (distance_list, sentence_metadata_list)
        """
        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = self._combine_sentences(_sentences, buffer_size)
        embeddings_list = embeddings.embed_chunks(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings_list[i]

        return self._calculate_cosine_distances(sentences)

    def _calculate_breakpoint_threshold(
        self, distances: List[float], breakpoint_threshold_type: BreakpointThresholdType, breakpoint_threshold_amount: Optional[float]
    ) -> Tuple[float, List[float]]:
        """
        Calculate optimal breakpoint threshold using statistical analysis.

        This method applies various statistical methods to determine the threshold value
        for identifying significant semantic breaks in the text.

        Args:
            distances: List of semantic distances between sentences
            breakpoint_threshold_type: Statistical method to use
            breakpoint_threshold_amount: Specific threshold value (optional)

        Returns:
            Tuple of (threshold_value, processed_distances)

        Raises:
            ValueError: If breakpoint_threshold_type is not supported
        """
        if breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[breakpoint_threshold_type]

        if breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, breakpoint_threshold_amount),
            ), distances
        elif breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(
                distances
            ) + breakpoint_threshold_amount * iqr, distances
        elif breakpoint_threshold_type == "gradient":
            # Calculate the threshold based on the distribution of gradient of distance array. # noqa: E501
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float], number_of_chunks: int) -> float:
        """
        Calculate threshold to achieve target number of chunks.

        This method uses linear interpolation to determine the appropriate threshold
        that will result in approximately the specified number of chunks.

        Args:
            distances: List of semantic distances
            number_of_chunks: Target number of chunks to create

        Returns:
            Calculated threshold value

        Raises:
            ValueError: If number_of_chunks is None when this method is called
        """
        if number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(number_of_chunks, x1), x2)

        # Linear interpolation formula
        if x2 == x1:
            y = y2
        else:
            y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def _cosine_similarity(self, X: Matrix, Y: Matrix) -> np.ndarray:
        """
        Calculate row-wise cosine similarity between two matrices.

        This method computes cosine similarity with optimization through simsimd if available,
        falling back to numpy implementation for compatibility.

        Args:
            X: First matrix of vectors
            Y: Second matrix of vectors

        Returns:
            Cosine similarity matrix

        Raises:
            ValueError: If matrix dimensions don't match
        """
        if len(X) == 0 or len(Y) == 0:
            return np.array([])

        X = np.array(X)
        Y = np.array(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Number of columns in X and Y must be the same. X has shape {X.shape} "
                f"and Y has shape {Y.shape}."
            )
        try:
            import simsimd as simd
            X = np.array(X, dtype=np.float32)
            Y = np.array(Y, dtype=np.float32)
            Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
            return Z
        except ImportError:
            X_norm = np.linalg.norm(X, axis=1)
            Y_norm = np.linalg.norm(Y, axis=1)
            # Ignore divide by zero errors run time warnings as those are handled below.
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
            similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
            return similarity

    def _combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        """
        Combine sentences with surrounding context for better semantic analysis.

        This method creates context-aware sentence representations by including
        neighboring sentences in each sentence's combined representation.

        Args:
            sentences: List of sentence dictionaries with metadata
            buffer_size: Number of neighboring sentences to include on each side

        Returns:
            List of sentences with combined_sentence field added
        """
        for i in range(len(sentences)):
            combined_sentence = ""
            for j in range(i - buffer_size, i):
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]["sentence"] + " "
            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += " " + sentences[j]["sentence"]
            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_cosine_distances(self, sentences: List[dict]) -> Tuple[List[float], List[dict]]:
        """
        Calculate cosine distances between consecutive sentence embeddings.

        This method computes the semantic distance between adjacent sentences
        to identify potential breakpoints for chunking.

        Args:
            sentences: List of sentences with embedding information

        Returns:
            Tuple of (distance_list, updated_sentences_with_distances)
        """
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            # Calculate cosine similarity
            similarity = self._cosine_similarity([embedding_current], [embedding_next])[0][0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]["distance_to_next"] = distance
        return distances, sentences