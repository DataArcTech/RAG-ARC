import faiss
import pickle
import os
import uuid
import numpy as np
from typing import Any, Optional, List, Dict

from encapsulation.database.vector_db.base import VectorDB
from encapsulation.data_model.schema import Document
from framework.shared_module_decorator import shared_module

import logging

logger = logging.getLogger(__name__)


@shared_module
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
        config: Configuration object containing embedding, index_type, metric, etc.
        embedding: Embedding model for text vectorization
        index_type (str): FAISS index type ('flat', 'ivf', 'hnsw')
        metric (str): Distance metric ('cosine', 'l2', 'ip')
        normalize_L2 (bool): Whether to normalize vectors for cosine similarity
        index: FAISS index instance

    Core methods:
        - _add_documents/aadd_documents: Add Document objects to the vector store
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
        >>> vs = config.build()
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

    def __init__(self, config):
        """Initialize FaissVectorDB with config

        Args:
            config: Configuration object containing embedding and other parameters
        """
        super().__init__(config)
        logger.info("Initializing FaissVectorDB")

        # Build embedding model from config if available
        if hasattr(config, 'embedding_config') and config.embedding_config is not None:
            logger.info("Building embedding model from config")
            self.embedding_model = config.embedding_config.build()
            logger.info(f"Embedding model initialized: {type(self.embedding_model).__name__}")
        else:
            logger.warning("No embedding model provided in config")
            self.embedding_model = None
        
        # initialize faiss attributes
        self.index = None  # faiss index
        self.docstore = {}  # Dictionary to store documents by ID
        self.index_to_docstore_id = {}  # Mapping from index position to document ID

    
    def load_index(self, path: str):
        """Load index from filesystem path

        Args:
            path: Directory path containing .faiss and .pkl files
        """
        logger.info(f"Loading index from path: {path}")

        # Validate path exists
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not os.path.isdir(path):
            logger.error(f"Path is not a directory: {path}")
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Check for required files
        faiss_files = [f for f in os.listdir(path) if f.endswith('.faiss')]
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]

        if not faiss_files:
            logger.error(f"No .faiss file found in {path}")
            raise FileNotFoundError(f"No .faiss file found in {path}")

        if not pkl_files:
            logger.error(f"No .pkl file found in {path}")
            raise FileNotFoundError(f"No .pkl file found in {path}")

        logger.info(f"Found {len(faiss_files)} .faiss file(s) and {len(pkl_files)} .pkl file(s)")

        # Find .faiss file
        if faiss_files:
            faiss_path = os.path.join(path, faiss_files[0])
            logger.info(f"Loading FAISS index from: {faiss_path}")
            self.index = faiss.read_index(faiss_path)
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, dimension {self.index.d}")

        # Find .pkl file
        if pkl_files:
            pkl_path = os.path.join(path, pkl_files[0])
            logger.info(f"Loading metadata from: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Load document store and mappings
            self.docstore = data.get("docstore", {})
            self.index_to_docstore_id = data.get("index_to_docstore_id", {})
            logger.info(f"Loaded {len(self.docstore)} documents from metadata")

            # Save pkl file parameters to override config values
            self.index_type = data.get("index_type")
            self.metric = data.get("metric")
            self.normalize_L2 = data.get("normalize_L2")
            logger.info(f"Index configuration: type={self.index_type}, metric={self.metric}")
    
    def build_index(self, documents: List[Document]):
        """Build index from documents

        Args:
            documents: List of Document objects to build index from
        """
        logger.info(f"Building index from {len(documents)} documents")

        # Check if index file is already built/loaded
        if self.index is not None and self.index.ntotal > 0:
            logger.error(f"Index already contains {self.index.ntotal} vectors")
            raise ValueError("Index already contains data. Use update_index() to add more documents or delete() first.")

        self._add_documents(documents)
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration"""
        # Use saved values if available (from pkl), otherwise use config
        index_type = getattr(self, 'saved_index_type', getattr(self.config, 'index_type', 'flat'))
        metric = getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine'))
        
        if metric == "cosine":
            if index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        elif metric == "l2":
            if index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        elif metric == "ip":
            if index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
            
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        # Use saved values if available (from pkl), otherwise use config
        normalize_L2 = getattr(self, 'saved_normalize_L2', getattr(self.config, 'normalize_L2', False))
        metric = getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine'))
        
        if normalize_L2 or metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors
    
    def _add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Internal method to add Document objects to vector database

        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments

        Returns:
            List of document IDs for added documents
        """
        if not documents:
            return []

        # Extract texts for embedding
        texts = [doc.content for doc in documents]

        # Compute embeddings
        embeddings = self.embedding_model.embed(texts)

        embeddings_np = np.array(embeddings).astype(np.float32)
        embeddings_np = self._normalize_vectors(embeddings_np)

        # Create index if it doesn't exist
        if self.index is None:
            dimension = embeddings_np.shape[1]
            self.index = self._create_index(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}")

        # Train IVF index if not trained and we have enough data
        if (hasattr(self.index, 'is_trained') and
            not self.index.is_trained and
            len(embeddings) >= 100):
            self.index.train(embeddings_np)

        # Get current index size
        start_index = self.index.ntotal

        # Add vectors to index
        self.index.add(embeddings_np)
        logger.info(f"Added {len(embeddings_np)} vectors to index (total: {self.index.ntotal})")

        # Store documents directly
        doc_ids = []
        for i, doc in enumerate(documents):
            # Generate ID if not provided
            doc_id = doc.id if doc.id is not None else str(uuid.uuid4())
            doc.id = doc_id  # Ensure doc has an ID
            doc_ids.append(doc_id)

            self.docstore[doc_id] = doc
            self.index_to_docstore_id[start_index + i] = doc_id

        logger.info(f"Stored {len(doc_ids)} documents in docstore")
        return doc_ids

    def delete_index(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
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
        if self.index is None:
            logger.warning("No index to delete from")
            return True

        if ids is None or not ids:
            raise ValueError("Dangerous operation: delete_index requires specific IDs. Use delete_all_index() if you want to clear all data.")

        logger.info(f"Deleting {len(ids)} documents from index")

        # Check if IDs to delete exist
        missing_ids = [doc_id for doc_id in ids if doc_id not in self.docstore]
        if missing_ids:
            logger.warning(f"IDs not found: {missing_ids}")
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

        logger.info(f"Keeping {len(remaining_docs)} documents, rebuilding index")

        # Clear current storage
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        if self.index is not None:
            self.index.reset()

        # Re-add remaining documents
        if remaining_docs:
            self._add_documents(remaining_docs)
            logger.info(f"Index rebuilt with {len(remaining_docs)} remaining documents")
        else:
            logger.info("Index is now empty after deletion")

        return True

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all documents from vector database

        Args:
            confirm: Set to True to confirm deletion

        Returns:
            True if deletion successful, False otherwise
        """
        if not confirm:
            raise ValueError("Dangerous operation: delete_all_index requires confirm=True")
        
        if self.index is None:
            logger.warning("No index to delete from")
            return True
        
        logger.info("Deleting all documents from index")
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        if self.index is not None:
            self.index.reset()
        logger.info("All documents deleted successfully")
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

    def update_index(self, documents: List[Document]) -> List[str]:
        """Update documents in index

        Args:
            documents: List of Document objects to update

        Returns:
            Ltst  of document IDs that were successfully added to the index
        """
        logger.info(f"Updating index with {len(documents)} documents")

        # Check if embedding model is available
        if self.embedding_model is None:
            logger.error("No embedding model available for update")
            return []

        try:
            # Add updated documents (embeddings will be generated automatically)
            doc_ids = self._add_documents(documents)
            logger.info(f"Update completed: {self.index.ntotal} total vectors")
            return doc_ids

        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            return []
    
    def save_index(self, path: str, name: str = "index") -> None:
        """Save index to filesystem path

        Args:
            path: Directory path to save the vector database
            name: Base name for saved files (without extension)
                 Creates {name}.faiss and {name}.pkl
        """
        logger.info(f"Saving index to path: {path} with name: {name}")
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss_path = os.path.join(path, f"{name}.faiss")
            faiss.write_index(self.index, faiss_path)
            logger.info(f"FAISS index saved: {faiss_path} ({self.index.ntotal} vectors)")
        else:
            logger.warning("No FAISS index to save")

        # Save other data
        data = {
            "docstore": self.docstore,
            "index_to_docstore_id": self.index_to_docstore_id,
            "index_type": getattr(self, 'saved_index_type', getattr(self.config, 'index_type', 'flat')),
            "metric": getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine')),
            "normalize_L2": getattr(self, 'saved_normalize_L2', getattr(self.config, 'normalize_L2', False)),
        }

        pkl_path = os.path.join(path, f"{name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Metadata saved: {pkl_path} ({len(self.docstore)} documents)")
    
    
    def get_vector_db_info(self) -> Dict[str, Any]:
        """Get vector database information

        Returns:
            Dictionary containing database info (size, dimensions, etc.)
        """
        # Get embedding model name safely
        embedding_model_name = 'unknown'
        if hasattr(self.config, 'embedding_config') and self.config.embedding_config is not None:
            if hasattr(self.config.embedding_config, 'model_name'):
                embedding_model_name = self.config.embedding_config.model_name

        info = {
            "type": "faiss",
            "index_type": getattr(self, 'saved_index_type', getattr(self.config, 'index_type', 'flat')),
            "metric": getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine')),
            "normalize_L2": getattr(self, 'saved_normalize_L2', getattr(self.config, 'normalize_L2', False)),
            "document_count": len(self.docstore),
            "embedding_model": embedding_model_name
        }

        # Add index-specific info if index exists
        if self.index is not None:
            info.update({
                "vector_count": self.index.ntotal,
                "dimension": self.index.d,
                "is_trained": getattr(self.index, 'is_trained', True)
            })
        else:
            info.update({
                "vector_count": 0,
                "dimension": 0,
                "is_trained": False
            })

        logger.info(f"Vector DB info: {info['document_count']} docs, {info['vector_count']} vectors")
        return info