"""
检索辅助函数模块

包含各种检索相关的辅助函数，如相关性评分函数、搜索结果处理等
"""

import math
import warnings
import logging
from typing import Callable, Any, Tuple, List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Any  # 避免循环导入，使用Any
    score: float
    distance: float


class RetrievalHelper:
    """检索辅助函数类，包含各种相关性评分和搜索处理方法"""
    
    @staticmethod
    def euclidean_relevance_score_fn(distance: float) -> float:
        """欧几里得距离的相关性评分函数
        
        返回[0, 1]范围内的相似性分数，其中1表示最相似
        
        Args:
            distance: 欧几里得距离
            
        Returns:
            相关性分数，范围[0, 1]
        """
        return 1.0 - distance / math.sqrt(2)
    
    @staticmethod
    def cosine_relevance_score_fn(distance: float) -> float:
        """余弦距离的相关性评分函数
        
        将距离归一化为[0, 1]范围内的分数
        
        Args:
            distance: 余弦距离
            
        Returns:
            相关性分数，范围[0, 1]
        """
        return 1.0 - distance
    
    @staticmethod
    def max_inner_product_relevance_score_fn(distance: float) -> float:
        """最大内积的相关性评分函数
        
        将距离归一化为[0, 1]范围内的分数
        
        Args:
            distance: 内积距离
            
        Returns:
            相关性分数，范围[0, 1]
        """
        if distance > 0:
            return 1.0 - distance
        return -1.0 * distance
    
    @staticmethod
    def get_relevance_score_fn(metric_type: str) -> Callable[[float], float]:
        """根据度量类型选择相关性评分函数
        
        Args:
            metric_type: 度量类型，可以是'euclidean', 'cosine', 'inner_product'等
            
        Returns:
            相关性评分函数
            
        Raises:
            NotImplementedError: 如果度量类型不支持
        """
        if metric_type.lower() in ['euclidean', 'l2']:
            return RetrievalHelper.euclidean_relevance_score_fn
        elif metric_type.lower() in ['cosine', 'cos']:
            return RetrievalHelper.cosine_relevance_score_fn
        elif metric_type.lower() in ['inner_product', 'ip', 'dot']:
            return RetrievalHelper.max_inner_product_relevance_score_fn
        else:
            raise NotImplementedError(f"不支持的度量类型: {metric_type}")
    
    @staticmethod
    def process_search_results_with_relevance_scores(
        docs_and_scores: List[Tuple[Any, float]],
        relevance_score_fn: Callable[[float], float],
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Any, float]]:
        """处理搜索结果并应用相关性评分
        
        Args:
            docs_and_scores: 文档和分数的元组列表
            relevance_score_fn: 相关性评分函数
            score_threshold: 可选的分数阈值，用于过滤结果
            
        Returns:
            处理后的(文档, 相关性分数)元组列表
        """
        # 应用相关性评分函数
        docs_and_similarities = [
            (doc, relevance_score_fn(score)) 
            for doc, score in docs_and_scores
        ]
        
        # 验证分数范围
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                f"相关性分数必须在0和1之间，得到 {docs_and_similarities}",
                stacklevel=2,
            )
        
        # 应用分数阈值过滤
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "使用相关性分数阈值 %s 没有检索到相关文档",
                    score_threshold,
                )
        
        return docs_and_similarities
    
    @staticmethod
    def maximal_marginal_relevance(
        query_embedding: List[float],
        embedding_list: List[List[float]],
        lambda_mult: float = 0.5,
        k: int = 4
    ) -> List[int]:
        """最大边际相关性算法
        
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
    def mmr_select_documents(
        docs_and_scores: List[Tuple[Any, float]],
        embeddings: List[List[float]],
        query_embedding: List[float],
        k: int,
        lambda_mult: float = 0.5,
    ) -> List[Any]:
        """最大边际相关性文档选择算法
        
        Args:
            docs_and_scores: 文档和分数的元组列表
            embeddings: 候选文档的嵌入向量列表
            query_embedding: 查询的嵌入向量
            k: 要选择的文档数量
            lambda_mult: 多样性权重，0到1之间
            
        Returns:
            选中的文档列表
        """
        import numpy as np
        
        if k >= len(docs_and_scores):
            return [doc for doc, _ in docs_and_scores]
        
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(docs_and_scores)))
        
        # 选择第一个文档（最相似的）
        first_idx = remaining_indices.pop(0)
        selected_indices.append(first_idx)
        selected_embeddings.append(embeddings[first_idx])
        
        # 选择剩余的k-1个文档
        for _ in range(k - 1):
            if not remaining_indices:
                break
                
            mmr_scores = []
            for idx in remaining_indices:
                # 计算与查询的相似性
                query_sim = np.dot(query_embedding, embeddings[idx])
                
                # 计算与已选择文档的最大相似性
                max_sim = 0
                for selected_emb in selected_embeddings:
                    sim = np.dot(selected_emb, embeddings[idx])
                    max_sim = max(max_sim, sim)
                
                # MMR分数
                mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim
                mmr_scores.append((idx, mmr_score))
            
            # 选择MMR分数最高的文档
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
            remaining_indices.remove(best_idx)
        
        return [docs_and_scores[idx][0] for idx in selected_indices]
    
    @staticmethod
    def select_relevance_score_fn_by_metric(metric: str) -> Callable[[float], float]:
        """根据向量数据库度量类型选择相关性评分函数
        
        Args:
            metric: 度量类型，支持'cosine', 'l2', 'ip'等
            
        Returns:
            相关性评分函数
            
        Raises:
            ValueError: 如果度量类型不支持
        """
        if metric.lower() in ['cosine', 'cos']:
            return RetrievalHelper.cosine_relevance_score_fn
        elif metric.lower() in ['l2', 'euclidean']:
            return RetrievalHelper.euclidean_relevance_score_fn
        elif metric.lower() in ['ip', 'inner_product', 'dot']:
            return RetrievalHelper.max_inner_product_relevance_score_fn
        else:
            raise ValueError(f"不支持的度量类型: {metric}")
    
    @staticmethod
    def normalize_vectors_for_cosine(embeddings: List[List[float]]) -> List[List[float]]:
        """为余弦相似度归一化向量

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
    def vector_search_with_faiss(
        index: Any,
        embedding: List[float],
        search_kwargs: Dict[str, Any]
    ) -> List[Tuple[Any, float]]:
        """使用FAISS执行向量搜索

        Args:
            index: FAISS索引对象
            embedding: 查询嵌入向量
            search_kwargs: 搜索参数，包含k, score_threshold等

        Returns:
            (文档, 分数)元组列表
        """
        import numpy as np
        import faiss

        if not hasattr(index, 'index') or index.index is None or index.index.ntotal == 0:
            return []

        # 获取搜索参数
        k = search_kwargs.get("k", 5)
        score_threshold = search_kwargs.get("score_threshold")
        metric = search_kwargs.get("metric", "cosine")

        # 准备查询向量
        query_vector = np.array([embedding]).astype(np.float32)

        # 检查是否需要归一化
        if hasattr(index.config, 'normalize_L2') and index.config.normalize_L2:
            faiss.normalize_L2(query_vector)
        elif hasattr(index.config, 'metric') and index.config.metric == "cosine":
            faiss.normalize_L2(query_vector)

        # 执行搜索
        k = min(k, index.index.ntotal)
        distances, indices = index.index.search(query_vector, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示无效结果
                continue

            doc_id = index.index_to_docstore_id[idx]
            doc = index.docstore[doc_id]

            # 对于cosine度量，FAISS返回的是相似度分数而不是距离
            if metric == "cosine":
                similarity_score = float(distance)
            else:
                # 对于其他度量，需要转换距离为相似度
                relevance_score_fn = RetrievalHelper.select_relevance_score_fn_by_metric(metric)
                similarity_score = relevance_score_fn(float(distance))

            results.append((doc, similarity_score))

        # 按相似度分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 应用分数阈值过滤（如果指定）
        if score_threshold is not None:
            results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]

            if len(results) == 0:
                logger.warning(
                    f"使用分数阈值 {score_threshold} 没有检索到相关文档"
                )

        return results

    @staticmethod
    def mmr_search(
        query_embedding: List[float],
        docs_and_scores: List[Tuple[Any, float]],
        embedding_model: Any,
        search_kwargs: Dict[str, Any]
    ) -> List[Any]:
        """最大边际相关性搜索

        Args:
            query_embedding: 查询嵌入向量
            docs_and_scores: 候选(文档, 分数)元组列表
            embedding_model: 嵌入模型
            search_kwargs: 搜索参数，包含k, lambda_mult等

        Returns:
            选中的文档列表
        """
        import numpy as np

        if not docs_and_scores:
            return []

        # 获取搜索参数
        k = search_kwargs.get("k", 4)
        lambda_mult = search_kwargs.get("lambda_mult", 0.5)
        normalize_for_cosine = search_kwargs.get("normalize_for_cosine", True)

        # 获取候选文档的嵌入向量
        candidate_embeddings = []
        for doc, _ in docs_and_scores:
            doc_embedding = embedding_model.embed_query(doc.content)
            candidate_embeddings.append(doc_embedding)

        # 转换为numpy数组
        query_emb_norm = np.array(query_embedding)
        candidate_embs_norm = np.array(candidate_embeddings)

        # 归一化处理（如果需要）
        if normalize_for_cosine:
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )

        # 使用MMR选择文档
        return RetrievalHelper.mmr_select_documents(
            docs_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )

    @staticmethod
    def add_scores_to_documents(
        docs: List[Any],
        docs_with_scores: List[Tuple[Any, float]]
    ) -> List[Any]:
        """为文档添加分数到元数据中

        Args:
            docs: 文档列表
            docs_with_scores: (文档, 分数)元组列表

        Returns:
            添加了分数元数据的文档列表
        """
        score_dict = {doc.id: score for doc, score in docs_with_scores}

        for doc in docs:
            if doc.id in score_dict:
                doc.metadata = {**(doc.metadata or {}), "score": score_dict[doc.id]}

        return docs