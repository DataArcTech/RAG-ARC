import faiss
import pickle
import os
import uuid
import numpy as np
import logging
from typing import Any, Optional, List, Literal, Dict


from encapsulation.database.vector_db.base import BaseIndex, BaseIndexConfig
from core.utils.data_model import Document
from framework.shared_module_decorator import shared_module

logger = logging.getLogger(__name__)



class FaissIndexConfig(BaseIndexConfig):
    """FAISS 索引配置"""
    type: Literal["faiss"] = "faiss"
    metric: Literal["cosine", "l2", "ip"]
    index_type: Literal["flat", "ivf", "hnsw"]
    nlist: int = 100
    m: int = 8
    efConstruction: int = 40
    efSearch: int = 16
    train_size: int = 10000
    normalize_L2: bool = True

    def build(self) -> "FaissIndex":
        return FaissIndex(self)

    

@shared_module
class FaissIndex(BaseIndex[FaissIndexConfig]):
    """FAISS向量数据库实现"""
    
    config: FaissIndexConfig
    
    def __init__(self, config: FaissIndexConfig):
        super().__init__(config=config)
        self.docstore: dict[str, Document] = {}
        self.index_to_docstore_id: dict[int, str] = {}
        self.index: Optional[faiss.Index] = None
    def load_index(self, index_path: Optional[str] = None):
        """从磁盘加载索引
        
        Args:
            index_path: 索引路径，如果为None则使用配置中的路径
        """
        # Use provided index_path or fall back to config
        target_path = index_path or self.config.index_path
        
        # Find .faiss file
        faiss_files = [f for f in os.listdir(target_path) if f.endswith('.faiss')]
        if faiss_files:
            faiss_path = os.path.join(target_path, faiss_files[0])
            self.index = faiss.read_index(faiss_path)
        
        # Find .pkl file  
        pkl_files = [f for f in os.listdir(target_path) if f.endswith('.pkl')]
        if pkl_files:
            pkl_path = os.path.join(target_path, pkl_files[0])
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Load document store and mappings
            self.docstore = data.get("docstore", {})
            self.index_to_docstore_id = data.get("index_to_docstore_id", {})
            
            # Override config with saved parameters
            self.config.index_type = data.get("index_type", self.config.index_type)
            self.config.metric = data.get("metric", self.config.metric)
    
    def build_index(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """构建索引

        Args:
            documents: 用于构建索引的文档列表
            embeddings: 预计算的文档嵌入向量
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided for building index")
        self.add(documents, embeddings)
        

    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration
        
        Args:
            dimension: Vector dimension
            
        Returns:
            Configured FAISS index instance
            
        Raises:
            ValueError: If unsupported index type or metric is specified
        """
        if self.config.metric == "cosine":
            # Cosine similarity uses inner product, requires normalized vectors
            if self.config.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, self.config.m)
                index.hnsw.efConstruction = self.config.efConstruction
                index.hnsw.efSearch = self.config.efSearch
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
        elif self.config.metric == "l2":
            if self.config.index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, self.config.m)
                index.hnsw.efConstruction = self.config.efConstruction
                index.hnsw.efSearch = self.config.efSearch
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
        elif self.config.metric == "ip":
            if self.config.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, self.config.m)
                index.hnsw.efConstruction = self.config.efConstruction
                index.hnsw.efSearch = self.config.efSearch
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.metric}")
            
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity
        
        Args:
            vectors: Input vectors to normalize
            
        Returns:
            Normalized vectors (in-place normalization)
        """
        if self.config.normalize_L2 or self.config.metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors
    
    def add(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """添加文档到索引（会根据ID去重，重复ID的文档不会被添加）

        Args:
            documents: 要添加的文档列表
            embeddings: 预计算的文档嵌入向量，如果为None则抛出异常

        Returns:
            成功添加的文档ID列表
        """
        doc_list = list(documents)
        if not doc_list:
            return []

        if embeddings is None:
            raise ValueError("Embeddings must be provided. Index does not handle embedding computation.")

        if len(embeddings) != len(doc_list):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of documents ({len(doc_list)})")

        # 检查重复ID并过滤
        unique_documents = []
        unique_embeddings = []
        duplicate_ids = []

        for i, doc in enumerate(doc_list):
            if doc.id in self.docstore:
                duplicate_ids.append(doc.id)
                logger.warning(f"Document with ID {doc.id} already exists. Use update() to update existing documents.")
            else:
                unique_documents.append(doc)
                unique_embeddings.append(embeddings[i])

        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate document IDs: {duplicate_ids}")

        if not unique_documents:
            logger.warning("No unique documents to add after deduplication")
            return []

        texts = [doc.content for doc in unique_documents]
        metadatas = [doc.metadata for doc in unique_documents]
        ids = [doc.id for doc in unique_documents if doc.id is not None]

        # Convert embeddings to numpy array
        embeddings_np = np.array(unique_embeddings).astype(np.float32)
        
        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings_np.shape[1]
            self.index = self._create_index(dimension)
        
        # Normalize vectors
        embeddings_np = self._normalize_vectors(embeddings_np)
        
        # Train IVF index if not trained and we have enough data
        if (hasattr(self.index, 'is_trained') and
            not self.index.is_trained and
            len(unique_embeddings) >= 100):
            self.index.train(embeddings_np)
        
        # Generate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of texts")
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in doc_list]
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")
        
        # Get current index size
        start_index = self.index.ntotal
        
        # Add vectors to index
        self.index.add(embeddings_np)
        
        # Store documents and mappings
        for i, doc in enumerate(unique_documents):
            self.docstore[doc.id] = doc
            self.index_to_docstore_id[start_index + i] = doc.id

        self.save_index(self.config.index_path)
        
        return ids
    
    
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """从索引中删除文档
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他删除条件
            
        Returns:
            删除是否成功
        """
        if ids is None:
            # Delete all
            self.docstore.clear()
            self.index_to_docstore_id.clear()
            if self.index is not None:
                self.index.reset()
            return True
        
        if not ids:
            return True
        
        # Check if IDs to delete exist
        for doc_id in ids:
            if doc_id not in self.docstore:
                return False
        
        # Get documents to keep
        remaining_docs = []
        remaining_texts = []
        remaining_metadatas = []
        remaining_ids = []
        
        for doc_id, doc in self.docstore.items():
            if doc_id not in ids:
                remaining_docs.append(doc)
                remaining_texts.append(doc.content)
                remaining_metadatas.append(doc.metadata)
                remaining_ids.append(doc_id)
        
        # Clear current storage
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        if self.index is not None:
            self.index.reset()
        
        # Re-add remaining documents
        if remaining_texts:
            self.add(remaining_docs)
        
        return True
    
    def update(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Update documents in index

        Args:
            documents: List of Document objects to update
            embeddings: 预计算的文档嵌入向量
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided for updating documents")

        # For FAISS, update is implemented as delete + add
        # Extract IDs from documents
        doc_ids = [doc.id for doc in documents if doc.id is not None]

        # Delete existing documents with these IDs
        if doc_ids:
            self.delete(doc_ids)

        # Add updated documents
        self.add(documents, embeddings)
    
    def get_by_ids(self, ids: List[str]) -> List['Document']:
        """Retrieve documents by their IDs
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents corresponding to the provided IDs
            Missing IDs are silently skipped
        """
        return [self.docstore[doc_id] for doc_id in ids if doc_id in self.docstore]
    

    
    def save_index(self, index_path: str, index_name: str = "index") -> None:
        """Save vector database to local filesystem
        
        Args:
            index_path: Directory path to save the vector database
            index_name: Base name for saved files (without extension)
                       Creates {index_name}.faiss and {index_name}.pkl
        """
        os.makedirs(index_path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(index_path, f"{index_name}.faiss"))
        
        # Save other data
        data = {
            "docstore": self.docstore,
            "index_to_docstore_id": self.index_to_docstore_id,
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "normalize_L2": self.config.normalize_L2,
        }
        
        with open(os.path.join(index_path, f"{index_name}.pkl"), "wb") as f:
            pickle.dump(data, f)

    def build_index(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """构建索引（仅在索引不存在时使用）

        Args:
            documents: 用于构建索引的文档列表
            embeddings: 预计算的文档嵌入向量

        Raises:
            RuntimeError: 如果索引已存在
        """
        if self.index_exists():
            raise RuntimeError(
                "Index already exists. Use add() to add documents to existing index, "
                "or delete the existing index first if you want to rebuild it."
            )

        # 清空现有数据
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        self.index = None

        # 对于build_index，需要在文档列表内部去重
        unique_documents = []
        unique_embeddings = []
        seen_ids = set()
        duplicate_ids = []

        for i, doc in enumerate(documents):
            if doc.id in seen_ids:
                duplicate_ids.append(doc.id)
                logger.warning(f"Duplicate document ID found in build_index: {doc.id}. Skipping duplicate.")
            else:
                seen_ids.add(doc.id)
                unique_documents.append(doc)
                if embeddings:
                    unique_embeddings.append(embeddings[i])

        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate document IDs in build_index: {duplicate_ids}")

        # 添加去重后的文档来构建索引
        self.add(unique_documents, unique_embeddings if embeddings else None)

    def index_exists(self) -> bool:
        """检查索引是否存在

        Returns:
            bool: 索引是否存在且包含文档
        """
        return self.index is not None and self.index.ntotal > 0

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """向量相似度搜索并返回分数

        Args:
            embedding: 查询向量
            k: 返回结果数量
            **kwargs: 其他搜索参数

        Returns:
            List[tuple[Document, float]]: 文档和相似度分数的元组列表
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Convert embedding to numpy array
        query_vector = np.array([embedding]).astype(np.float32)

        # Normalize if needed
        if self.config.normalize_L2:
            faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, k)

        # Convert results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                break

            if idx in self.index_to_docstore_id:
                doc_id = self.index_to_docstore_id[idx]
                if doc_id in self.docstore:
                    doc = self.docstore[doc_id]

                    # Convert distance to similarity score based on metric
                    if self.config.metric == "cosine":
                        # For cosine similarity with IndexFlatIP and normalized vectors,
                        # FAISS returns the inner product which equals cosine similarity
                        # Range is [-1, 1], where 1 is most similar
                        similarity_score = float(score)
                    elif self.config.metric == "ip":  # Inner product
                        # For inner product, higher is better (already similarity)
                        similarity_score = float(score)
                    else:  # L2 distance
                        # For L2, lower distance means higher similarity
                        # Convert to similarity: similarity = 1 / (1 + distance)
                        similarity_score = 1.0 / (1.0 + score)

                    results.append((doc, similarity_score))

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息

        Returns:
            Dict[str, Any]: 索引统计信息
        """
        if self.index is None:
            return {
                "num_documents": 0,
                "index_type": self.config.index_type,
                "metric": self.config.metric,
                "dimension": 0,
                "is_trained": False,
                "docstore_size": len(self.docstore)
            }

        return {
            "num_documents": self.index.ntotal,
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "dimension": self.index.d,
            "is_trained": getattr(self.index, 'is_trained', True),
            "docstore_size": len(self.docstore)
        }