import faiss
import pickle
import os
import uuid
import json
import numpy as np
from typing import Any, Optional, List, Dict, TYPE_CHECKING

from encapsulation.database.vector_db.base import VectorDB
from encapsulation.data_model.schema import Chunk
from framework.shared_module_decorator import shared_module
from core.utils.path_guard import ensure_writable_dir
from core.utils.faiss_lock import FAISS_LOCK

if TYPE_CHECKING:
    from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig

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
    - Dynamic chunk addition and soft-delete (avoids expensive index rebuilding)

    Main parameters:
        config: Configuration object containing embedding, index_type, metric, etc.
        embedding: Embedding model for text vectorization
        index_type (str): FAISS index type ('flat', 'ivf', 'hnsw')
        metric (str): Distance metric ('cosine', 'l2', 'ip')
        normalize_L2 (bool): Whether to normalize vectors for cosine similarity
        index: FAISS index instance

    Core methods:
        - _add_chunks/aadd_chunks: Add Chunk objects to the vector store
        - similarity_search_by_vector: Search by embedding vector
        - max_marginal_relevance_search: MMR-based diverse search
        - delete_index: Soft-delete chunks by IDs (fast, no rebuild)
        - hard_delete_index: Hard-delete chunks by IDs (slow, rebuilds index)
        - save_index/load_index: Persist and restore index
        - from_chunks: Create instance from chunk collection

    Performance considerations:
        - Flat index: Best for small collections (<10K chunks)
        - IVF index: Good for medium collections (10K-1M chunks)
        - HNSW index: Best for large collections (>1M chunks)
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
        docstore: Chunk storage mapping
        index_to_docstore_id: Index to chunk ID mapping
    """

    def __init__(self, config: "FaissVectorDBConfig"):
        """Initialize FaissVectorDB with config

        Args:
            config: Configuration object containing embedding and other parameters
        """
        super().__init__(config)
        logger.info("Initializing FaissVectorDB")

        # Build embedding model from config
        self.embedding_model = self.config.embedding_config.build()

        # initialize faiss attributes
        self.index = None  # faiss index
        self.docstore = {}  # Dictionary to store chunks by ID
        self.index_to_docstore_id = {}  # Mapping from index position to chunk ID
        self.deleted_ids = set()  # Set to track soft-deleted chunk IDs
        self._lock = FAISS_LOCK

    
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
            with self._lock:
                self.index = faiss.read_index(faiss_path)
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, dimension {self.index.d}")

        # Find .pkl file
        if pkl_files:
            pkl_path = os.path.join(path, pkl_files[0])
            logger.info(f"Loading metadata from: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Load chunk store and mappings
            self.docstore = data.get("docstore", {})
            self.index_to_docstore_id = data.get("index_to_docstore_id", {})
            self.deleted_ids = set(data.get("deleted_ids", []))
            stored_fingerprint = data.get("embedding_fingerprint")
            current_fingerprint = self._embedding_fingerprint()
            if stored_fingerprint and stored_fingerprint != current_fingerprint:
                logger.warning(
                    "FAISS embedding fingerprint mismatch; requiring rebuild. stored=%s current=%s",
                    stored_fingerprint,
                    current_fingerprint,
                )
                with self._lock:
                    self.index = None
                    self.docstore = {}
                    self.index_to_docstore_id = {}
                    self.deleted_ids = set()
                raise ValueError("FAISS embedding fingerprint mismatch (requires rebuild)")
            logger.info(f"Loaded {len(self.docstore)} chunks from metadata ({len(self.deleted_ids)} soft-deleted)")

            # Dimension validation must never require a live embedding API call. Prefer persisted dimension,
            # then config-provided `embedding_dimensions` if available.
            stored_dim = data.get("embedding_dim")
            cfg = getattr(self.embedding_model, "config", None)
            configured_dim = getattr(cfg, "embedding_dimensions", None) if cfg is not None else None
            expected_dim = stored_dim if stored_dim is not None else configured_dim
            if expected_dim is not None and self.index is not None and getattr(self.index, "d", None) is not None:
                loaded_dim = int(self.index.d)
                if loaded_dim != int(expected_dim):
                    logger.warning(
                        "FAISS index dimension mismatch: index.d=%s expected_dim=%s. "
                        "Ignoring loaded index and requiring a rebuild.",
                        loaded_dim,
                        expected_dim,
                    )
                    with self._lock:
                        self.index = None
                        self.docstore = {}
                        self.index_to_docstore_id = {}
                        self.deleted_ids = set()
                    raise ValueError(
                        f"FAISS index dimension mismatch: index.d={loaded_dim} expected_dim={int(expected_dim)}"
                    )

            # Replay persisted index parameters so load+update stays consistent with the on-disk index.
            self.saved_index_type = data.get("index_type")
            self.saved_metric = data.get("metric")
            self.saved_normalize_L2 = data.get("normalize_L2")
            # Backward-compatible mirrors (some callers may read these directly).
            self.index_type = self.saved_index_type
            self.metric = self.saved_metric
            self.normalize_L2 = self.saved_normalize_L2
            logger.info(
                "Index configuration (replayed): type=%s metric=%s normalize_L2=%s",
                self.saved_index_type,
                self.saved_metric,
                self.saved_normalize_L2,
            )

    def _infer_embedding_dim(self) -> Optional[int]:
        cfg = getattr(self.embedding_model, "config", None)
        candidate = getattr(cfg, "embedding_dimensions", None)
        if candidate is not None:
            try:
                dim = int(candidate)
                if dim > 0:
                    return dim
            except Exception:
                pass
        try:
            probe = self.embedding_model.embed(["dimension probe"])
            if isinstance(probe, list) and probe:
                first = probe[0]
                if isinstance(first, list):
                    dim = len(first)
                else:
                    dim = len(probe)  # type: ignore[arg-type]
                return int(dim) if dim > 0 else None
        except Exception:
            return None

    def _embedding_fingerprint(self) -> str:
        cfg = getattr(self.embedding_model, "config", None)
        payload: Dict[str, Any] = {}
        if cfg is not None:
            payload = {
                "type": getattr(cfg, "type", None) or cfg.__class__.__name__,
                "loading_method": getattr(cfg, "loading_method", None),
                "model_name": getattr(cfg, "model_name", None),
                "embedding_dimensions": getattr(cfg, "embedding_dimensions", None),
            }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    
    def build_index(self, chunks: List[Chunk]):
        """Build index from chunks

        Args:
            chunks: List of Chunk objects to build index from
        """
        logger.info(f"Building index from {len(chunks)} chunks")

        # Check if index file is already built/loaded
        if self.index is not None and self.index.ntotal > 0:
            logger.error(f"Index already contains {self.index.ntotal} vectors")
            raise ValueError("Index already contains data. Use update_index() to add more chunks or delete() first.")

        self._add_chunks(chunks)
    
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
            with self._lock:
                faiss.normalize_L2(vectors)
        return vectors
    
    def _add_chunks(
        self,
        chunks: List[Chunk],
        **kwargs: Any,
    ) -> List[str]:
        """Internal method to add Chunk objects to vector database

        Args:
            chunks: List of Chunk objects to add
            **kwargs: Additional arguments

        Returns:
            List of chunk IDs for added chunks
        """
        if not chunks:
            return []

        # Extract texts for embedding (reuse precomputed embeddings when available)
        embeddings_by_row: list[Optional[np.ndarray]] = [None] * len(chunks)
        texts_to_embed: list[str] = []
        rows_to_embed: list[int] = []

        for idx, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata", None) or {}

            precomputed = metadata.get("embedding")
            if precomputed is not None:
                try:
                    emb = np.array(precomputed, dtype=np.float32)
                    if emb.ndim != 1:
                        raise ValueError("embedding must be a 1D vector")
                    embeddings_by_row[idx] = emb
                    continue
                except Exception:
                    embeddings_by_row[idx] = None

            index_text = metadata.get("index_text")
            if not isinstance(index_text, str) or not index_text.strip():
                index_text = chunk.content
            texts_to_embed.append(index_text)
            rows_to_embed.append(idx)

        if texts_to_embed:
            computed = self.embedding_model.embed(texts_to_embed)
            if isinstance(computed, list) and computed and isinstance(computed[0], (int, float)):
                computed = [computed]
            if len(computed) != len(texts_to_embed):
                raise RuntimeError(
                    f"Embedding size mismatch: got {len(computed)} for {len(texts_to_embed)} inputs"
                )
            for row_idx, embedding in zip(rows_to_embed, computed):
                embeddings_by_row[row_idx] = np.array(embedding, dtype=np.float32)

        missing = [i for i, emb in enumerate(embeddings_by_row) if emb is None]
        if missing:
            raise RuntimeError(f"Missing embeddings for {len(missing)} chunks (first_missing_index={missing[0]})")

        embeddings_np = np.vstack([emb for emb in embeddings_by_row if emb is not None]).astype(np.float32)
        embeddings_np = self._normalize_vectors(embeddings_np)

        with self._lock:
            # Create index if it doesn't exist
            if self.index is None:
                dimension = embeddings_np.shape[1]
                self.index = self._create_index(dimension)
                logger.info(f"Created new FAISS index with dimension {dimension}")

            # Train IVF index if not trained and we have enough data
            if (
                hasattr(self.index, "is_trained")
                and not self.index.is_trained
                and embeddings_np.shape[0] >= 100
            ):
                self.index.train(embeddings_np)

            # Get current index size
            start_index = self.index.ntotal

            # Add vectors to index
            self.index.add(embeddings_np)
            logger.info(f"Added {len(embeddings_np)} vectors to index (total: {self.index.ntotal})")

            # Store chunks directly
            doc_ids = []
            for i, chunk in enumerate(chunks):
                # Generate ID if not provided
                chunk_id = chunk.id if chunk.id is not None else str(uuid.uuid4())
                chunk.id = chunk_id  # Ensure chunk has an ID
                doc_ids.append(chunk_id)

                self.docstore[chunk_id] = chunk
                self.index_to_docstore_id[start_index + i] = chunk_id

        logger.info(f"Stored {len(doc_ids)} chunks in docstore")
        return doc_ids

    def delete_index(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete chunks from vector database using soft-delete

        This method uses soft-delete to mark chunks as deleted without rebuilding
        the entire index. Soft-deleted chunks are filtered out during search and retrieval.

        Args:
            ids: List of chunk IDs to delete; if None, raises an error
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False if some IDs not found, None if not implemented
        """
        if self.index is None:
            logger.warning("No index to delete from")
            return True

        if ids is None or not ids:
            raise ValueError("Dangerous operation: delete_index requires specific IDs. Use delete_all_index() if you want to clear all data.")

        logger.info(f"Soft-deleting {len(ids)} chunks from index")

        # Check if IDs to delete exist
        missing_ids = [doc_id for doc_id in ids if doc_id not in self.docstore]
        if missing_ids:
            # Idempotency: treat missing IDs as already deleted (e.g., index rebuilt / partial cleanup).
            logger.warning("IDs not found in FAISS docstore (treating as already deleted): %s", missing_ids)

        with self._lock:
            # Mark chunks as deleted (soft-delete)
            for doc_id in ids:
                if doc_id not in self.docstore:
                    continue
                if doc_id not in self.deleted_ids:
                    self.deleted_ids.add(doc_id)
                    logger.debug(f"Soft-deleted chunk: {doc_id}")

        logger.info(f"Soft-deleted {len(ids)} chunks (total deleted: {len(self.deleted_ids)})")
        return True

    def hard_delete_index(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Hard-delete chunks from vector database by rebuilding the index

        This method physically removes chunks by rebuilding the entire index with
        remaining chunks. This is expensive but reclaims storage space.
        Use this when you need to compact the index after many soft-deletes.

        Args:
            ids: List of chunk IDs to delete; if None, raises an error
            **kwargs: Additional arguments

        Returns:
            True if deletion successful, False if some IDs not found, None if not implemented
        """
        if self.index is None:
            logger.warning("No index to delete from")
            return True

        if ids is None or not ids:
            raise ValueError("Dangerous operation: hard_delete_index requires specific IDs. Use delete_all_index() if you want to clear all data.")

        logger.info(f"Hard-deleting {len(ids)} chunks from index (rebuilding)")

        # Check if IDs to delete exist
        missing_ids = [doc_id for doc_id in ids if doc_id not in self.docstore]
        if missing_ids:
            logger.warning(f"IDs not found: {missing_ids}")
            return False

        # Get chunks to keep (excluding both specified IDs and soft-deleted IDs)
        ids_to_remove = set(ids)
        remaining_docs = []

        for doc_id, doc in self.docstore.items():
            if doc_id not in ids_to_remove and doc_id not in self.deleted_ids:
                remaining_docs.append(doc)

        logger.info(f"Keeping {len(remaining_docs)} chunks, rebuilding index")

        # Clear current storage
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        self.deleted_ids.clear()
        if self.index is not None:
            self.index.reset()

        # Re-add remaining chunks
        if remaining_docs:
            self._add_chunks(remaining_docs)
            logger.info(f"Index rebuilt with {len(remaining_docs)} remaining chunks")
        else:
            logger.info("Index is now empty after deletion")

        return True

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks from vector database

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

        logger.info("Deleting all chunks from index")
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        self.deleted_ids.clear()
        if self.index is not None:
            self.index.reset()
        logger.info("All chunks deleted successfully")
        return True

    def compact_index(self) -> bool:
        """Compact the index by removing all soft-deleted chunks

        This method rebuilds the index to physically remove all soft-deleted chunks,
        reclaiming storage space. Use this periodically when you have many soft-deleted chunks.

        Returns:
            True if compaction successful, False otherwise
        """
        if self.index is None:
            logger.warning("No index to compact")
            return True

        if not self.deleted_ids:
            logger.info("No soft-deleted chunks to compact")
            return True

        logger.info(f"Compacting index: removing {len(self.deleted_ids)} soft-deleted chunks")

        # Get all non-deleted chunks
        remaining_docs = []
        for doc_id, doc in self.docstore.items():
            if doc_id not in self.deleted_ids:
                remaining_docs.append(doc)

        logger.info(f"Keeping {len(remaining_docs)} active chunks, rebuilding index")

        # Clear current storage
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        self.deleted_ids.clear()
        if self.index is not None:
            self.index.reset()

        # Re-add remaining chunks
        if remaining_docs:
            self._add_chunks(remaining_docs)
            logger.info(f"Index compacted: {len(remaining_docs)} chunks remaining")
        else:
            logger.info("Index is now empty after compaction")

        return True


    def get_by_ids(self, ids: List[str]) -> List['Chunk']:
        """Retrieve chunks by their IDs

        Args:
            ids: List of chunk IDs to retrieve

        Returns:
            List of chunks corresponding to the provided IDs
            Missing IDs and soft-deleted IDs are silently skipped
        """
        return [
            self.docstore[doc_id]
            for doc_id in ids
            if doc_id in self.docstore and doc_id not in self.deleted_ids
        ]

    def update_index(self, chunks: List[Chunk]) -> List[str]:
        """Update chunks in index

        Args:
            chunks: List of Chunk objects to update

        Returns:
            List of chunk IDs that were successfully added to the index
        """
        logger.info(f"Updating index with {len(chunks)} chunks")

        # Check if embedding model is available
        if self.embedding_model is None:
            logger.error("No embedding model available for update")
            return []

        try:
            chunk_ids = self._add_chunks(chunks)
            logger.info(f"Update completed: {self.index.ntotal} total vectors")
            return chunk_ids
        except Exception as exc:  # noqa: BLE001
            logger.exception("Update failed")

            if self.index is not None and chunks:
                try:
                    probe_text = getattr(chunks[0], "content", "") or ""
                    probe_emb = self.embedding_model.embed([probe_text])[0]
                    probe_dim = int(np.array(probe_emb, dtype=np.float32).shape[0])
                    if int(getattr(self.index, "d", 0) or 0) != probe_dim:
                        logger.warning(
                            "FAISS index dimension mismatch (index.d=%s, embedding_dim=%s). Rebuilding index.",
                            getattr(self.index, "d", None),
                            probe_dim,
                        )
                        # Rebuild using current embedding model to keep the index consistent.
                        merged: Dict[str, Chunk] = {}
                        for chunk_id, chunk in self.docstore.items():
                            if chunk_id in self.deleted_ids:
                                continue
                            merged[chunk_id] = chunk
                        for chunk in chunks:
                            chunk_id = getattr(chunk, "id", None) or str(uuid.uuid4())
                            chunk.id = chunk_id
                            merged[chunk_id] = chunk

                        with self._lock:
                            self.docstore.clear()
                            self.index_to_docstore_id.clear()
                            self.deleted_ids.clear()
                            if self.index is not None:
                                self.index.reset()
                            self.index = None

                        rebuilt_ids = self._add_chunks(list(merged.values()))
                        logger.info("FAISS index rebuilt successfully (%s chunks).", len(rebuilt_ids))
                        return rebuilt_ids
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to rebuild FAISS index after update failure")

            logger.error("Update failed: %s", str(exc))
            return []
    
    def save_index(self, path: str, name: str = "index") -> None:
        """Save index to filesystem path

        Args:
            path: Directory path to save the vector database
            name: Base name for saved files (without extension)
                 Creates {name}.faiss and {name}.pkl
        """
        logger.info(f"Saving index to path: {path} with name: {name}")
        runtime_root = os.getenv("RAGARC_RUNTIME_DIR", "./local/runtime")
        fallback_dir = os.path.join(runtime_root, os.path.basename(path) or "faiss_index")
        path = ensure_writable_dir(path, fallback_dir)
        os.makedirs(path, exist_ok=True)

        with self._lock:
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
                "deleted_ids": list(self.deleted_ids),  # Convert set to list for serialization
                "index_type": getattr(self, 'saved_index_type', getattr(self.config, 'index_type', 'flat')),
                "metric": getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine')),
                "normalize_L2": getattr(self, 'saved_normalize_L2', getattr(self.config, 'normalize_L2', False)),
                "embedding_fingerprint": self._embedding_fingerprint(),
                "embedding_dim": getattr(self.index, "d", None),
            }

            pkl_path = os.path.join(path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Metadata saved: {pkl_path} ({len(self.docstore)} chunks)")
    
    
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

        # Calculate active (non-deleted) chunk count
        active_chunk_count = len(self.docstore) - len(self.deleted_ids)

        info = {
            "type": "faiss",
            "index_type": getattr(self, 'saved_index_type', getattr(self.config, 'index_type', 'flat')),
            "metric": getattr(self, 'saved_metric', getattr(self.config, 'metric', 'cosine')),
            "normalize_L2": getattr(self, 'saved_normalize_L2', getattr(self.config, 'normalize_L2', False)),
            "chunk_count": active_chunk_count,
            "total_chunks": len(self.docstore),
            "deleted_chunks": len(self.deleted_ids),
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

        logger.info(f"Vector DB info: {info['chunk_count']} active chunks ({info['deleted_chunks']} deleted), {info['vector_count']} vectors")
        return info
