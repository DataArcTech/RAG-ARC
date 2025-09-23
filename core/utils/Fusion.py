from typing import List
from abc import ABC, abstractmethod
from collections import defaultdict

from encapsulation.data_model.schema import Document


class FusionMethod(ABC):
    """融合方法的抽象基类"""
    
    @abstractmethod
    def fuse(self, results: List[List[Document]], top_k: int) -> List[Document]:
        """
        融合多个检索器的结果
        
        Args:
            results: 每个检索器的结果列表，每个列表包含Document对象
            top_k: 返回的最终结果数量
            
        Returns:
            融合后的Document列表，分数存储在metadata['score']中
        """
        pass


class RRFusion(FusionMethod):
    """Reciprocal Rank Fusion (RRF) 方法"""
    
    def __init__(self, k: float = 60.0):
        """
        Args:
            k: RRF中的常数，默认为60.0
        """
        self.k = k
    
    def fuse(self, results: List[List[Document]], top_k: int) -> List[Document]:
        # 计算RRF分数
        rrf_scores = defaultdict(float)
        document_map = {}
        
        for retriever_results in results:
            for rank, document in enumerate(retriever_results, 1):  # rank从1开始
                rrf_score = 1.0 / (self.k + rank)
                # 使用文档内容作为key来去重
                content_key = document.content
                rrf_scores[content_key] += rrf_score
                document_map[content_key] = document
        
        # 按RRF分数排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果，将分数放到文档的metadata中
        fused_documents = []
        for content, rrf_score in sorted_items[:top_k]:
            document = document_map[content]
            # 将RRF分数添加到文档的metadata中
            if document.metadata is None:
                document.metadata = {}
            document.metadata["score"] = rrf_score
            
            fused_documents.append(document)
        
        return fused_documents


class WeightedSumFusion(FusionMethod):
    """加权求和融合方法"""

    def __init__(self, weights: List[float]):
        """
        Args:
            weights: 每个检索器的权重列表
        """
        if not weights or len(weights) == 0:
            raise ValueError("Weights list cannot be empty")
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero")
        self.weights = [w / total_weight for w in weights]

    def fuse(self, results: List[List[Document]], top_k: int) -> List[Document]:
        if len(results) != len(self.weights):
            raise ValueError(f"Number of result lists ({len(results)}) must match number of weights ({len(self.weights)})")

        # 计算加权分数
        weighted_scores = defaultdict(float)
        document_map = {}

        for retriever_idx, retriever_results in enumerate(results):
            weight = self.weights[retriever_idx]

            for document in retriever_results:
                content_key = document.content
                # 获取原始分数，如果没有则使用1.0
                original_score = document.metadata.get('score', 1.0) if document.metadata else 1.0
                weighted_scores[content_key] += weight * original_score
                document_map[content_key] = document

        # 按加权分数排序
        sorted_items = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        fused_documents = []
        for content, weighted_score in sorted_items[:top_k]:
            document = document_map[content]
            if document.metadata is None:
                document.metadata = {}
            document.metadata["score"] = weighted_score
            fused_documents.append(document)

        return fused_documents


class RankFusion(FusionMethod):
    """基于排名的融合方法"""

    def __init__(self):
        """排名融合不需要额外参数"""
        pass

    def fuse(self, results: List[List[Document]], top_k: int) -> List[Document]:
        # 计算排名分数（排名越靠前分数越高）
        rank_scores = defaultdict(float)
        document_map = {}

        for retriever_results in results:
            max_rank = len(retriever_results)
            for rank, document in enumerate(retriever_results):
                content_key = document.content
                # 排名分数：最高排名得到最高分数
                rank_score = max_rank - rank
                rank_scores[content_key] += rank_score
                document_map[content_key] = document

        # 按排名分数排序
        sorted_items = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        fused_documents = []
        for content, rank_score in sorted_items[:top_k]:
            document = document_map[content]
            if document.metadata is None:
                document.metadata = {}
            document.metadata["score"] = rank_score
            fused_documents.append(document)

        return fused_documents
