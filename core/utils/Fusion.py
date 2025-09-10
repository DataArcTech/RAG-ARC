from typing import List
from abc import ABC, abstractmethod
from collections import defaultdict

from core.utils.data_model import Document


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
