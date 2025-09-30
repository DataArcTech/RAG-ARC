from typing import List
from abc import ABC, abstractmethod
from collections import defaultdict

from encapsulation.data_model.schema import Chunk


class FusionMethod(ABC):
    """Abstract base class for fusion methods"""
    
    @abstractmethod
    def fuse(self, results: List[List[Chunk]], top_k: int) -> List[Chunk]:
        """
        Fuse results from multiple retrievers
        
        Args:
            results: list of results from each retriever, each list contains Chunk objects
            top_k: number of final results to return
            
        Returns:
            List of fused Chunk objects, scores stored in metadata['score']
        """
        pass


class RRFusion(FusionMethod):
    """Reciprocal Rank Fusion (RRF) method"""
    
    def __init__(self, k: float = 60.0):
        """
        Args:
            k: constant in RRF, defaults to 60.0
        """
        self.k = k
    
    def fuse(self, results: List[List[Chunk]], top_k: int) -> List[Chunk]:
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        chunk_map = {}
        
        for retriever_results in results:
            for rank, chunk in enumerate(retriever_results, 1):  # rank starts from 1
                rrf_score = 1.0 / (self.k + rank)
                # Use chunk content as key to deduplicate
                content_key = chunk.content
                rrf_scores[content_key] += rrf_score
                chunk_map[content_key] = chunk
        
        # Sort by RRF scores
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results, put scores in chunk's metadata
        fused_chunks = []
        for content, rrf_score in sorted_items[:top_k]:
            chunk = chunk_map[content]
            # Add RRF score to chunk's metadata
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["score"] = rrf_score
            
            fused_chunks.append(chunk)
        
        return fused_chunks


class WeightedSumFusion(FusionMethod):
    """Weighted sum fusion method"""

    def __init__(self, weights: List[float]):
        """
        Args:
            weights: list of weights for each retriever
        """
        if not weights or len(weights) == 0:
            raise ValueError("Weights list cannot be empty")
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        self.weights = [w / total_weight for w in weights]

    def fuse(self, results: List[List[Chunk]], top_k: int) -> List[Chunk]:
        if len(results) != len(self.weights):
            raise ValueError(f"Number of result lists ({len(results)}) must match number of weights ({len(self.weights)})")

        # Calculate weighted scores
        weighted_scores = defaultdict(float)
        chunk_map = {}

        for retriever_idx, retriever_results in enumerate(results):
            weight = self.weights[retriever_idx]

            for chunk in retriever_results:
                content_key = chunk.content
                # Get original score, use 1.0 if not present
                original_score = chunk.metadata.get('score', 1.0) if chunk.metadata else 1.0
                weighted_scores[content_key] += weight * original_score
                chunk_map[content_key] = chunk

        # Sort by weighted scores
        sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused_chunks = []
        for content, weighted_score in sorted_items[:top_k]:
            chunk = chunk_map[content]
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["score"] = weighted_score
            fused_chunks.append(chunk)

        return fused_chunks


class RankFusion(FusionMethod):
    """Rank-based fusion method"""

    def __init__(self):
        """Rank fusion does not require additional parameters"""
        pass

    def fuse(self, results: List[List[Chunk]], top_k: int) -> List[Chunk]:
        # Calculate rank scores (higher rank = higher score)
        rank_scores = defaultdict(float)
        chunk_map = {}

        for retriever_results in results:
            max_rank = len(retriever_results)
            for rank, chunk in enumerate(retriever_results):
                content_key = chunk.content
                # Rank score: highest rank gets highest score
                rank_score = max_rank - rank
                rank_scores[content_key] += rank_score
                chunk_map[content_key] = chunk

        # Sort by rank scores
        sorted_items = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        fused_chunks = []
        for content, rank_score in sorted_items[:top_k]:
            chunk = chunk_map[content]
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["score"] = rank_score
            fused_chunks.append(chunk)

        return fused_chunks
