import faiss
import pickle
import os
import uuid
import numpy as np
from typing import Any, Optional, List, Tuple, Literal
from pydantic import Field, field_validator

from encapsulation.database.vector_db.base import BaseVectorDB, BaseVectorDBConfig
from core.utils.data_model import Document



class FaissVectorDBConfig(BaseVectorDBConfig):
    """Configuration for FAISS Vector Database"""
    type: Literal["faiss"] = "faiss"
    
    # index path
    index_path: Optional[str] = Field(default=None, description="Path to FAISS index file")
    
    # FAISS specific configuration
    index_type: str = Field(default="flat", description="FAISS index type: 'flat', 'ivf', 'hnsw'")

    # Index construction parameters
    nlist: int = Field(default=100, description="Number of clusters for IVF index")
    m: int = Field(default=8, description="Number of connections for HNSW index")
    efConstruction: int = Field(default=40, description="Construction parameter for HNSW index")
    efSearch: int = Field(default=16, description="Search parameter for HNSW index")
    
    # Training parameters
    train_size: int = Field(default=10000, description="Maximum number of vectors to use for index training")
      
    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v: str) -> str:
        allowed_types = {"flat", "ivf", "hnsw"}
        if v not in allowed_types:
            raise ValueError(f"index_type must be one of {allowed_types}, but got {v}")
        return v
    
    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        allowed_metrics = {"cosine", "l2", "ip"}
        if v not in allowed_metrics:
            raise ValueError(f"metric must be one of {allowed_metrics}, but got {v}")
        return v
    
    def build(self):
        """Build the FAISS Vector Database"""
        if self.embedding is None:
            raise ValueError("Embedding model is required but not provided")
            
        # embedding 现在应该是已经加载好的模型实例，直接使用
        return FaissVectorDB(config=self)
    


class FaissVectorDB(BaseVectorDB[FaissVectorDBConfig]):
    """
    FAISS-based vector database implementation for high-performance similarity search and retrieval.
    
    This class provides a complete vector database solution using Facebook's FAISS library,
    supporting multiple index types, distance metrics, and advanced features like Maximal
    Marginal Relevance (MMR) for diverse result sets.
    
    Key features:
    - Multiple index types: flat, IVF (Inverted File), HNSW (Hierarchical Navigable Small World)
    - Multiple distance metrics: cosine similarity, L2 distance, inner product
    - Vector normalization support for cosine similarity
    - Maximal Marginal Relevance (MMR) search for diversity
    - Persistent storage with save/load functionality
    - Asynchronous operations support
    - Dynamic document addition and deletion
    
    Configuration parameters (from config):
        embedding: Embedding model interface for text vectorization
        index_type (str): FAISS index type ('flat', 'ivf', 'hnsw')
        metric (str): Distance metric ('cosine', 'l2', 'ip')
        normalize_L2 (bool): Whether to normalize vectors for cosine similarity
        nlist, m, efConstruction, efSearch: Index construction parameters
        train_size: Maximum number of vectors to use for index training
        k: Default number of documents to return in search
        with_score: Whether to include relevance scores in results
        search_kwargs: Additional search parameters
        
    Runtime instance variables:
        docstore: Document storage mapping
        index_to_docstore_id: Index to document ID mapping
        index: FAISS index instance
        
    Core methods:
        - add_documents: Add documents to the vector store
        - similarity_search_by_vector: Search by embedding vector
        - max_marginal_relevance_search: MMR-based diverse search
        - delete: Remove documents by IDs
        - save_local/load_local: Persist and restore index
        - initialize_from_documents: Create instance from document collection
        
    Performance considerations:
        - Flat index: Best for small collections (<10K documents)
        - IVF index: Good for medium collections (10K-1M documents)
        - HNSW index: Best for large collections (>1M documents)
        - Cosine similarity requires vector normalization
        - Index training required for IVF with sufficient data (>=100 vectors)
        
    Typical usage:
        >>> config = FaissVectorDBConfig(embedding=embedding_model)
        >>> vs = config.build()
        >>> vs.initialize_from_documents(documents)
        >>> docs = vs.similarity_search("query")
        >>> vs.save_local("./index")
    """
    
    # Runtime instance variables (initialized when needed)
    _docstore: Optional[dict[str, 'Document']] = None
    _index_to_docstore_id: Optional[dict[int, str]] = None
    _index: Optional[faiss.Index] = None
    
    @property
    def docstore(self) -> dict[str, 'Document']:
        """Lazy-initialized document store"""
        if self._docstore is None:
            self._docstore = {}
        return self._docstore
    
    @docstore.setter
    def docstore(self, value: dict[str, 'Document']) -> None:
        """Set document store"""
        self._docstore = value
    
    @property
    def index_to_docstore_id(self) -> dict[int, str]:
        """Lazy-initialized index to document ID mapping"""
        if self._index_to_docstore_id is None:
            self._index_to_docstore_id = {}
        return self._index_to_docstore_id
    
    @index_to_docstore_id.setter
    def index_to_docstore_id(self, value: dict[int, str]) -> None:
        """Set index to document ID mapping"""
        self._index_to_docstore_id = value
    
    @property
    def index(self) -> Optional[faiss.Index]:
        """FAISS index instance"""
        return self._index
    
    @index.setter
    def index(self, value: Optional[faiss.Index]) -> None:
        """Set FAISS index instance"""
        self._index = value
    
    def load_from_folder(self, folder_path: str):
        """Initialize this instance by loading from folder
        
        Args:
            folder_path: Directory path containing .faiss and .pkl files
        """
        # Find .faiss file
        faiss_files = [f for f in os.listdir(folder_path) if f.endswith('.faiss')]
        if faiss_files:
            faiss_path = os.path.join(folder_path, faiss_files[0])
            self.index = faiss.read_index(faiss_path)
        
        # Find .pkl file  
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        if pkl_files:
            pkl_path = os.path.join(folder_path, pkl_files[0])
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Load document store and mappings
            self.docstore = data.get("docstore", {})
            self.index_to_docstore_id = data.get("index_to_docstore_id", {})
            
            # Override config with saved parameters
            self.config.index_type = data.get("index_type", self.config.index_type)
            self.config.metric = data.get("metric", self.config.metric) 
            self.config.normalize_L2 = data.get("normalize_L2", self.config.normalize_L2)
    
    def initialize_from_documents(self, documents: List[Document]):
        """Initialize this instance from documents
        
        Args:
            documents: List of Document objects to add to the vector database
        """
        self.add_documents(documents)
        

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
    
    def add_documents(self,documents: List[Document]) -> List[str]:
        """Add texts to vector database
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs for added documents
            
        Raises:
            ValueError: If number of IDs or metadatas doesn't match number of texts
        """
        doc_list = list(documents)
        if not doc_list:
            return []

        texts = [doc.content for doc in doc_list]
        metadatas = [doc.metadata for doc in doc_list]
        ids = [doc.id for doc in doc_list if doc.id is not None]
        
        # Embed documents
        embeddings = self.config.embedding.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype(np.float32)
        
        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings_np.shape[1]
            self.index = self._create_index(dimension)
        
        # Normalize vectors
        embeddings_np = self._normalize_vectors(embeddings_np)
        
        # Train IVF index if not trained and we have enough data
        if (hasattr(self.index, 'is_trained') and 
            not self.index.is_trained and 
            len(embeddings) >= 100):
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

        for i, doc in enumerate(doc_list):
            self.docstore[doc.id] = doc
            self.index_to_docstore_id[start_index + i] = doc.id
        
        return ids
    
    
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from vector database
        
        Note: FAISS doesn't support direct deletion, so this method rebuilds
        the entire index with remaining documents. This can be expensive for
        large collections.
        
        Args:
            ids: List of document IDs to delete; if None, deletes all documents
            **kwargs: Additional arguments
            
        Returns:
            True if deletion successful, False if some IDs not found, None if not implemented
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
            self.add_documents(remaining_docs)
        
        return True
    
    def get_by_ids(self, ids: List[str]) -> List['Document']:
        """Retrieve documents by their IDs
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents corresponding to the provided IDs
            Missing IDs are silently skipped
        """
        return [self.docstore[doc_id] for doc_id in ids if doc_id in self.docstore]
    
    def save_local(self, index_path: str, index_name: str = "index") -> None:
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
    
    def as_retriever(self, **kwargs: Any):
        """Create a retriever from the current vector database
        
        检索器使用向量数据库中配置的检索参数。
        所有检索相关的配置都在向量数据库配置中定义。
        
        Args:
            **kwargs: 可选的检索器配置覆盖参数
                     - k (int): 覆盖默认返回文档数量
                     - with_score (bool): 覆盖默认分数包含设置
                     - search_kwargs (dict): 额外搜索参数
        
        Returns:
            DenseRetriever instance
            
        Raises:
            RuntimeError: If vector database is not initialized
            
        Examples:
            # 创建检索器 - 使用向量数据库中的配置
            retriever = vector_db.as_retriever()
            
            # 运行时可以覆盖向量数据库中的默认配置
            results = retriever.invoke(
                "查询文本",
                k=5,                    # 覆盖向量数据库中的默认k值
                search_type="mmr",      # 指定搜索类型
                lambda_mult=0.7,        # MMR多样性参数
                score_threshold=0.8     # 相关性分数阈值
            )
        """
        from core.retrieval.dense import DenseRetrieverConfig
        
        if self.index is None:
            raise RuntimeError("向量数据库未初始化。请先调用 initialize_from_folder() 或 initialize_from_documents() 初始化数据库。")
        
        
        # 从配置中获取检索参数，允许kwargs覆盖
        retriever_k = kwargs.get('k', getattr(self.config, 'k', 10))
        retriever_with_score = kwargs.get('with_score', getattr(self.config, 'with_score', True))
        retriever_search_kwargs = kwargs.get('search_kwargs', getattr(self.config, 'search_kwargs', {}))
        
        # 创建检索器配置，注入运行时依赖
        retriever_config = DenseRetrieverConfig(
            # 注入运行时依赖
            vectorstore=self,
            metric=self.config.metric,
            
            # 从向量数据库配置中获取检索参数
            k=retriever_k,
            with_score=retriever_with_score,
            search_kwargs=retriever_search_kwargs.copy()
        )
        
        # 创建并返回检索器
        retriever = retriever_config.build()
        
        return retriever
    

    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List['Document']:
        """Return documents most similar to query
        
        Args:
            query: Input text
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of documents most similar to query
        """
        # Embed query
        query_embedding = self.config.embedding.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List['Document']:
        """Return documents most similar to embedding vector
        
        Args:
            embedding: Embedding to look up documents similar to
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of documents most similar to query vector
        """
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple['Document', float]]:
        """Search by vector with scores
        
        Args:
            embedding: Embedding vector to search with
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Prepare query vector
        query_vector = np.array([embedding]).astype(np.float32)
        query_vector = self._normalize_vectors(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            doc_id = self.index_to_docstore_id[idx]
            doc = self.docstore[doc_id]
            results.append((doc, float(distance)))
        
        return results
    
    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, score_threshold: Optional[float] = None, **kwargs: Any
    ) -> List[Tuple['Document', float]]:
        """Return documents and relevance scores in the range [0, 1]
        
        Args:
            query: Input text
            k: Number of documents to return
            score_threshold: Optional score threshold to filter results
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, relevance_score) tuples
        """
        # Get relevance score function based on metric
        relevance_score_fn = self._select_relevance_score_fn()
        
        # Get documents with distance scores
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        
        # Convert distances to relevance scores
        docs_and_similarities = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]
        
        # Apply score threshold if specified
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
        
        return docs_and_similarities
    
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple['Document', float]]:
        """Search with distance scores
        
        Args:
            query: Input text
            k: Number of documents to return
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, distance_score) tuples
        """
        # Embed query
        query_embedding = self.config.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(query_embedding, k, **kwargs)
    
    def _select_relevance_score_fn(self):
        """Select relevance score function based on metric"""
        from core.utils.retrieval_helper import RetrievalHelper
        
        try:
            return RetrievalHelper.select_relevance_score_fn_by_metric(self.config.metric)
        except ValueError:
            # Default to cosine if metric not supported
            return RetrievalHelper.cosine_relevance_score_fn
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List['Document']:
        """Return docs selected using the maximal marginal relevance
        
        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch to pass to MMR algorithm
            lambda_mult: Number between 0 and 1 that determines the degree of diversity
            **kwargs: Additional search parameters
            
        Returns:
            List of Documents selected by maximal marginal relevance
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self.config.embedding.embed_query(query)
        
        # Get candidate documents
        docs_and_scores = self.similarity_search_by_vector_with_score(
            query_embedding, fetch_k, **kwargs
        )
        
        if not docs_and_scores:
            return []
        
        # Get document embeddings
        candidate_embeddings = []
        for doc, _ in docs_and_scores:
            # Re-embed document content (in practice, you might want to cache these)
            doc_embedding = self.config.embedding.embed_query(doc.content)
            candidate_embeddings.append(doc_embedding)
        
        # Normalize embeddings for cosine similarity
        query_emb_norm = np.array(query_embedding)
        candidate_embs_norm = np.array(candidate_embeddings)
        
        if self.config.normalize_L2 or self.config.metric == "cosine":
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )
        
        # Use MMR selection from retrieval helper
        from core.utils.retrieval_helper import RetrievalHelper
        selected_docs = RetrievalHelper.mmr_select_documents(
            docs_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )
        
        return selected_docs

