"""
Retrieval helper module

Contains various retrieval-related helper functions, such as relevance score functions, search result processing, etc.
"""

import math
import warnings
import logging
from typing import Callable, Any, Tuple, List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result data structure"""
    chunk: Any  # Avoid circular imports, use Any
    score: float
    distance: float


class RetrievalHelper:
    """Retrieval helper class, contains various relevance score and search processing methods"""
    
    @staticmethod
    def euclidean_relevance_score_fn(distance: float) -> float:
        """Euclidean distance relevance score function
        
        Returns similarity score in [0, 1], where 1 means most similar
        
        Args:
            distance: Euclidean distance
            
        Returns:
            Relevance score, range [0, 1]
        """
        return 1.0 - distance / math.sqrt(2)
    
    @staticmethod
    def cosine_relevance_score_fn(distance: float) -> float:
        """Cosine distance relevance score function
        
        Normalize distance to [0, 1] range
        
        Args:
            distance: cosine distance
            
        Returns:
            Relevance score, range [0, 1]
        """
        return 1.0 - distance
    
    @staticmethod
    def max_inner_product_relevance_score_fn(distance: float) -> float:
        """Maximum inner product relevance score function
        
        Normalize distance to [0, 1] range
        
        Args:
            distance: inner product distance
            
        Returns:
            Relevance score, range [0, 1]
        """
        if distance > 0:
            return 1.0 - distance
        return -1.0 * distance
    
    @staticmethod
    def get_relevance_score_fn(metric_type: str) -> Callable[[float], float]:
        """Select relevance score function based on metric type
        
        Args:
            metric_type: metric type, can be 'euclidean', 'cosine', 'inner_product', etc.
            
        Returns:
            Relevance score function
            
        Raises:
            NotImplementedError: if metric type is not supported
        """
        if metric_type.lower() in ['euclidean', 'l2']:
            return RetrievalHelper.euclidean_relevance_score_fn
        elif metric_type.lower() in ['cosine', 'cos']:
            return RetrievalHelper.cosine_relevance_score_fn
        elif metric_type.lower() in ['inner_product', 'ip', 'dot']:
            return RetrievalHelper.max_inner_product_relevance_score_fn
        else:
            raise NotImplementedError(f"Unsupported metric type: {metric_type}")
    
    @staticmethod
    def process_search_results_with_relevance_scores(
        docs_and_scores: List[Tuple[Any, float]],
        relevance_score_fn: Callable[[float], float],
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Any, float]]:
        """Process search results and apply relevance scores
        
        Args:
            docs_and_scores: list of (chunk, score) tuples
            relevance_score_fn: relevance score function
            score_threshold: optional score threshold, used for filtering results
            
        Returns:
            list of processed (chunk, relevance score) tuples
        """
        # Apply relevance score function
        docs_and_similarities = [
            (doc, relevance_score_fn(score)) 
            for doc, score in docs_and_scores
        ]
        
        # Verify score range
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                f"Relevance score must be between 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )
        
        # Apply score threshold filtering
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    f"Using relevance score threshold {score_threshold} did not retrieve any relevant chunks",
                )
        
        return docs_and_similarities
    
    @staticmethod
    def maximal_marginal_relevance(
        query_embedding: List[float],
        embedding_list: List[List[float]],
        lambda_mult: float = 0.5,
        k: int = 4
    ) -> List[int]:
        """
        
        选择既与查询相关又彼此多样化的文档索引
        
        Args:
            query_embedding: 查询的嵌入向量
            embedding_list: 候选文档的嵌入向量列表
            lambda_mult: 多样性权重，0到1之间，0表示最大多样性，1表示最大相关性
            k: 要返回的文档数量
            
        Returns:
            选中文档的索引列表
        """
        if not embedding_list:
            return []
        
        import numpy as np
        
        # 转换为numpy数组
        query_vec = np.array(query_embedding)
        embeddings = np.array(embedding_list)
        
        # 计算查询与所有文档的相似度
        query_similarities = np.dot(embeddings, query_vec)
        
        # 初始化选中的文档索引列表
        selected_indices = []
        remaining_indices = list(range(len(embedding_list)))
        
        # 选择第一个最相似的文档
        if remaining_indices:
            best_idx = remaining_indices[np.argmax(query_similarities)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # 迭代选择剩余文档
        while len(selected_indices) < k and remaining_indices:
            best_score = float('-inf')
            best_idx = None
            
            for idx in remaining_indices:
                # 计算与查询的相关性
                relevance = query_similarities[idx]
                
                # 计算与已选文档的最大相似度（多样性惩罚）
                if selected_indices:
                    selected_embeddings = embeddings[selected_indices]
                    similarities = np.dot(selected_embeddings, embeddings[idx])
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0
                
                # MMR分数：lambda_mult * 相关性 - (1 - lambda_mult) * 多样性惩罚
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected_indices
    
    @staticmethod
    def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
        """归一化嵌入向量
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            归一化后的嵌入向量列表
        """
        import numpy as np
        
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # 避免除零
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings_array / norms
        return normalized.tolist()
    
    @staticmethod
    def calculate_similarity(
        embedding1: List[float], 
        embedding2: List[float], 
        metric: str = 'cosine'
    ) -> float:
        """计算两个嵌入向量之间的相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            metric: 相似度度量，支持'cosine', 'euclidean', 'dot_product'
            
        Returns:
            相似度分数
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        if metric == 'cosine':
            # 余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        elif metric == 'euclidean':
            # 欧几里得距离（转换为相似度）
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
        elif metric == 'dot_product':
            # 点积
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
    
    @staticmethod
    def mmr_select_chunks(
        docs_and_scores: List[Tuple[Any, float]],
        embeddings: List[List[float]],
        query_embedding: List[float],
        k: int,
        lambda_mult: float = 0.5,
    ) -> List[Any]:
        """Maximal marginal relevance search (diversity)
        
        Args:
            docs_and_scores: list of (chunk, score) tuples
            embeddings: list of candidate chunks' embedding vectors
            query_embedding: query embedding vector
            k: number of chunks to select
            lambda_mult: diversity weight, between 0 and 1, 0 means max diversity, 1 means max relevance
            
        Returns:
            list of selected chunks
        """
        import numpy as np
        
        if k >= len(docs_and_scores):
            return [doc for doc, _ in docs_and_scores]
        
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(docs_and_scores)))
        
        # Select first chunk (most similar)
        first_idx = remaining_indices.pop(0)
        selected_indices.append(first_idx)
        selected_embeddings.append(embeddings[first_idx])
        
        # Select remaining k-1 chunks
        for _ in range(k - 1):
            if not remaining_indices:
                break
                
            mmr_scores = []
            for idx in remaining_indices:
                # Calculate similarity with query
                query_sim = np.dot(query_embedding, embeddings[idx])
                
                # Calculate maximum similarity with selected chunks
                max_sim = 0
                for selected_emb in selected_embeddings:
                    sim = np.dot(selected_emb, embeddings[idx])
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            # Select chunk with highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
            remaining_indices.remove(best_idx)
        
        return [docs_and_scores[idx][0] for idx in selected_indices]
    
    @staticmethod
    def select_relevance_score_fn_by_metric(metric: str) -> Callable[[float], float]:
        """Select relevance score function based on vector database metric type
        
        Args:
            metric: metric type, supports 'cosine', 'l2', 'ip', etc.
            
        Returns:
            relevance score function
            
        Raises:
            ValueError: if metric type is not supported
        """
        if metric.lower() in ['cosine', 'cos']:
            return RetrievalHelper.cosine_relevance_score_fn
        elif metric.lower() in ['l2', 'euclidean']:
            return RetrievalHelper.euclidean_relevance_score_fn
        elif metric.lower() in ['ip', 'inner_product', 'dot']:
            return RetrievalHelper.max_inner_product_relevance_score_fn
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
    
    @staticmethod
    def normalize_vectors_for_cosine(embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize vectors for cosine similarity

        Args:
            embeddings: list of embedding vectors

        Returns:
            normalized embedding vectors
        """
        import numpy as np

        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings_array / norms
        return normalized.tolist()

    @staticmethod
    def vector_search_with_faiss(
        index: Any,
        embedding: List[float],
        search_kwargs: Dict[str, Any]
    ) -> List[Tuple[Any, float]]:
        """Execute vector search with FAISS

        Args:
            index: FAISS index object
            embedding: query embedding vector
            search_kwargs: search parameters, including k, score_threshold, etc.

        Returns:
            list of (chunk, score) tuples
        """
        import numpy as np
        import faiss

        if not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []

        # Get search parameters
        k = search_kwargs.get("k", 5)
        score_threshold = search_kwargs.get("score_threshold")
        metric = search_kwargs.get("metric", "cosine")

        # Prepare query vector
        query_vector = np.array([embedding]).astype(np.float32)

        # Check if normalization is needed
        if hasattr(index.config, 'normalize_L2') and index.config.normalize_L2:
            faiss.normalize_L2(query_vector)
        elif hasattr(index.config, 'metric') and index.config.metric == "cosine":
            faiss.normalize_L2(query_vector)

        # Calculate fetch_k to account for soft-deleted documents
        # Fetch more results than needed to compensate for deleted documents
        deleted_count = len(index.deleted_ids) if hasattr(index, 'deleted_ids') else 0
        fetch_k = min(k + deleted_count, index.index.ntotal)

        # Execute search
        distances, indices = index.index.search(query_vector, fetch_k)

        results = []
        skipped_count = 0
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            doc_id = index.index_to_docstore_id[idx]

            # Skip soft-deleted documents
            if hasattr(index, 'deleted_ids') and doc_id in index.deleted_ids:
                skipped_count += 1
                continue

            doc = index.docstore[doc_id]

            # For cosine metric, FAISS returns similarity score instead of distance
            if metric == "cosine":
                similarity_score = float(distance)
            else:
                # For other metrics, convert distance to similarity score
                relevance_score_fn = RetrievalHelper.select_relevance_score_fn_by_metric(metric)
                similarity_score = relevance_score_fn(float(distance))

            results.append((doc, similarity_score))

            # Stop once we have k results (after filtering deleted documents)
            if len(results) >= k:
                break


        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply score threshold filtering (if specified)
        if score_threshold is not None:
            results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]

            if len(results) == 0:
                logger.warning(
                    f"Using score threshold {score_threshold} did not retrieve any relevant chunks"
                )

        return results

    @staticmethod
    def mmr_search(
        query_embedding: List[float],
        chunks_and_scores: List[Tuple[Any, float]],
        embedding_model: Any,
        search_kwargs: Dict[str, Any]
    ) -> List[Any]:
        """Maximal marginal relevance search (diversity)

        Args:
            query_embedding: query embedding vector
            chunks_and_scores: list of candidate (chunk, score) tuples
            embedding_model: embedding model
            search_kwargs: search parameters, including k, lambda_mult, etc.

        Returns:
            list of selected chunks
        """
        import numpy as np

        if not chunks_and_scores:
            return []

        # Get search parameters
        k = search_kwargs.get("k", 4)
        lambda_mult = search_kwargs.get("lambda_mult", 0.5)
        normalize_for_cosine = search_kwargs.get("normalize_for_cosine", True)

        # Get candidate chunks' embedding vectors
        candidate_embeddings = []
        for doc, _ in chunks_and_scores:
            doc_embedding = embedding_model.embed(doc.content)
            candidate_embeddings.append(doc_embedding)

        # Convert to numpy arrays
        query_emb_norm = np.array(query_embedding)
        candidate_embs_norm = np.array(candidate_embeddings)

        # Normalize if needed
        if normalize_for_cosine:
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )

        # Use MMR to select chunks
        return RetrievalHelper.mmr_select_chunks(
            chunks_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )

    @staticmethod
    def add_scores_to_chunks(
        chunks: List[Any],
        chunks_with_scores: List[Tuple[Any, float]]
    ) -> List[Any]:
        """Add scores to chunks' metadata

        Args:
            chunks: list of chunks
            chunks_with_scores: list of (chunk, score) tuples

        Returns:
            list of chunks with scores added to metadata
        """
        score_dict = {chunk.id: score for chunk, score in chunks_with_scores}

        for chunk in chunks:
            if chunk.id in score_dict:
                chunk.metadata = {**(chunk.metadata or {}), "score": score_dict[chunk.id]}

        return chunks