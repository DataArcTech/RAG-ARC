import faiss
import pickle
import os
import uuid
import numpy as np
from typing import Any, Optional, List, Tuple, TYPE_CHECKING, Sequence, Iterable
import asyncio
from concurrent.futures import ThreadPoolExecutor

from encapsulation.database.vector_db.base import VectorDB
from core.utils.data_model import Document

if TYPE_CHECKING:
    from framework.config import AbstractConfig


class FaissVectorDB(VectorDB):
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
    
    Main parameters:
        config (AbstractConfig): Configuration object containing embedding, index_type, metric, etc.
        embedding: Embedding model for text vectorization
        index_type (str): FAISS index type ('flat', 'ivf', 'hnsw')
        metric (str): Distance metric ('cosine', 'l2', 'ip')
        normalize_L2 (bool): Whether to normalize vectors for cosine similarity
        index: FAISS index instance
        
    Core methods:
        - add_texts/aadd_texts: Add text documents to the vector store
        - similarity_search_by_vector: Search by embedding vector
        - max_marginal_relevance_search: MMR-based diverse search
        - delete: Remove documents by IDs
        - save_local/load_local: Persist and restore index
        - from_documents: Create instance from document collection
        
    Performance considerations:
        - Flat index: Best for small collections (<10K documents)
        - IVF index: Good for medium collections (10K-1M documents)
        - HNSW index: Best for large collections (>1M documents)
        - Cosine similarity requires vector normalization
        - Index training required for IVF with sufficient data (>=100 vectors)
        
    Typical usage:
        >>> config = VectorStoreConfig(embedding=embedding_model)
        >>> vs = FaissVectorDB(config)
        >>> ids = vs.add_texts(["text1", "text2"])
        >>> docs = vs.similarity_search("query")
        >>> vs.save_local("./index")
    
    Attributes:
        embedding: Embedding model interface
        index_type: FAISS index type
        metric: Distance metric
        normalize_L2: Vector normalization flag
        index: FAISS index instance
        docstore: Document storage mapping
        index_to_docstore_id: Index to document ID mapping
    """
    
    def __init__(self, config: "AbstractConfig"):
        """Initialize FAISS vector database with basic configuration
        
        Creates a basic instance ready for index setup through instance methods.
        The index is not created yet - use setup methods to populate it.
        
        Args:
            config: Configuration object containing embedding, index_type, metric, etc.
                   Required attributes:
                   - embedding: Embedding model interface
                   Optional attributes:
                   - index_type: Index type ('flat', 'ivf', 'hnsw'), defaults to 'flat'
                   - metric: Distance metric ('cosine', 'l2', 'ip'), defaults to 'cosine'
                   - normalize_L2: Vector normalization flag, defaults to False
        """
        super().__init__(config)
        
        # Extract parameters from config
        self.embedding = config.embedding
        self.index_type = getattr(config, 'index_type', 'flat')
        self.metric = getattr(config, 'metric', 'cosine')
        self.normalize_L2 = getattr(config, 'normalize_L2', False)
        
        # Initialize storage (empty, no index yet)
        self.docstore: dict[str, 'Document'] = {}
        self.index_to_docstore_id: dict[int, str] = {}
        self.index = None  # Will be populated by initialize methods
    
    def initialize_from_folder(self, folder_path: str):
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
            self.index_type = data.get("index_type", self.index_type)
            self.metric = data.get("metric", self.metric) 
            self.normalize_L2 = data.get("normalize_L2", self.normalize_L2)
    
    def initialize_from_documents(self, documents: List[Document]):
        """Initialize this instance from documents
        
        Args:
            documents: List of Document objects to add to the vector database
        """
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.id for doc in documents if doc.id is not None]
        
        self.add_texts(texts, metadatas=metadatas, ids=ids if ids else None)
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration
        
        Args:
            dimension: Vector dimension
            
        Returns:
            Configured FAISS index instance
            
        Raises:
            ValueError: If unsupported index type or metric is specified
        """
        if self.metric == "cosine":
            # Cosine similarity uses inner product, requires normalized vectors
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        elif self.metric == "ip":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")
            
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity
        
        Args:
            vectors: Input vectors to normalize
            
        Returns:
            Normalized vectors (in-place normalization)
        """
        if self.normalize_L2 or self.metric == "cosine":
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
        embeddings = self.embedding.embed_documents(texts)
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
    
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save vector database to local filesystem
        
        Args:
            folder_path: Directory path to save the vector database
            index_name: Base name for saved files (without extension)
                       Creates {index_name}.faiss and {index_name}.pkl
        """
        os.makedirs(folder_path, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(folder_path, f"{index_name}.faiss"))
        
        # Save other data
        data = {
            "docstore": self.docstore,
            "index_to_docstore_id": self.index_to_docstore_id,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize_L2": self.normalize_L2,
        }
        
        with open(os.path.join(folder_path, f"{index_name}.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    
    
