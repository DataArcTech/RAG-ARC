import logging
import threading
import uuid
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jCacheMixin:
    def _get_tls(self) -> threading.local:
        tls = getattr(self, "_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(self, "_tls", tls)
        return tls

    @property
    def _cached_owner_id(self) -> Optional[str]:
        return getattr(self._get_tls(), "cached_owner_id", None)

    @_cached_owner_id.setter
    def _cached_owner_id(self, value: Optional[str]) -> None:
        setattr(self._get_tls(), "cached_owner_id", value)

    @property
    def _cached_store_version(self) -> Optional[Union[int, str]]:
        return getattr(self._get_tls(), "cached_store_version", None)

    @_cached_store_version.setter
    def _cached_store_version(self, value: Optional[Union[int, str]]) -> None:
        setattr(self._get_tls(), "cached_store_version", value)

    @property
    def passage_embeddings_array(self) -> Optional[np.ndarray]:
        return getattr(self._get_tls(), "passage_embeddings_array", None)

    @passage_embeddings_array.setter
    def passage_embeddings_array(self, value: Optional[np.ndarray]) -> None:
        setattr(self._get_tls(), "passage_embeddings_array", value)

    @property
    def passage_node_source_file_ids(self) -> list[Optional[str]]:
        return getattr(self._get_tls(), "passage_node_source_file_ids", [])

    @passage_node_source_file_ids.setter
    def passage_node_source_file_ids(self, value: list[Optional[str]]) -> None:
        setattr(self._get_tls(), "passage_node_source_file_ids", value)

    def invalidate_cache(self):
        """Force invalidation of all cached data."""
        self._cached_owner_id = None
        self._cached_store_version = None
        self.passage_node_keys = []
        self.passage_node_source_file_ids = []
        self.passage_embeddings_array = None

    @staticmethod
    def _owner_to_str(owner_id: Optional[uuid.UUID]) -> Optional[str]:
        return str(owner_id) if owner_id is not None else None

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None, force_rebuild: bool = False):
        """Build mappings between passage nodes and their IDs from Neo4j."""
        current_store_version = self.graph_store.get_cache_version()
        owner_str = self._owner_to_str(owner_id)

        cache_valid = (
            not force_rebuild
            and self._cached_owner_id == owner_str
            and self._cached_store_version == current_store_version
            and self.passage_embeddings_array is not None
        )

        if cache_valid:
            logger.debug("Using cached node mappings for %d passage nodes", len(self.passage_node_keys))
            return

        if self._cached_store_version != current_store_version:
            logger.info(
                "Cache version changed (%s -> %s), rebuilding...",
                self._cached_store_version,
                current_store_version,
            )

        self.passage_node_keys = []

        if owner_str:
            query = """
            MATCH (c:Chunk {owner_id: $owner_id})
            RETURN c.chunk_id AS chunk_id, c.source_file_id AS source_file_id
            ORDER BY c.created_at
            """
            results = self.graph_store._execute_query(query, {"owner_id": owner_str})
        else:
            query = """
            MATCH (c:Chunk)
            RETURN c.chunk_id AS chunk_id, c.source_file_id AS source_file_id
            ORDER BY c.created_at
            """
            results = self.graph_store._execute_query(query)

        # Filter out chunks from deleted files
        chunk_ids = []
        source_file_ids = set()
        for record in results:
            chunk_id = record["chunk_id"]
            source_file_id = record.get("source_file_id")
            if source_file_id:
                source_file_ids.add(source_file_id)
            chunk_ids.append((chunk_id, source_file_id))
        
        # Batch check file status if we have source_file_ids
        deleted_file_ids = set()
        if source_file_ids:
            try:
                from framework.register import Register
                registrator = Register()
                knowledge_module = getattr(registrator, "registrations", {}).get("knowledge")
                if knowledge_module:
                    for file_id in source_file_ids:
                        if not knowledge_module.is_file_active(file_id):
                            deleted_file_ids.add(file_id)
            except Exception as e:
                logger.warning(f"Failed to check file status for filtering: {e}")
        
        # Filter chunks: keep only those from active files or without source_file_id
        filtered_pairs = [
            (chunk_id, source_file_id)
            for chunk_id, source_file_id in chunk_ids
            if not source_file_id or source_file_id not in deleted_file_ids
        ]
        self.passage_node_keys = [chunk_id for chunk_id, _fid in filtered_pairs]
        self.passage_node_source_file_ids = [str(fid) if fid is not None else None for _cid, fid in filtered_pairs]

        passage_embeddings_list = []
        embedding_dim: int | None = None

        read_lock = getattr(self.graph_store, "read_lock", None)
        if callable(read_lock):
            with self.graph_store.read_lock():
                for chunk_id in self.passage_node_keys:
                    if chunk_id in self.graph_store.chunk_embeddings:
                        passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                    else:
                        if embedding_dim is None:
                            if passage_embeddings_list:
                                embedding_dim = len(passage_embeddings_list[0])
                            else:
                                embedding_dim = self.graph_store.embedding_model.get_embedding_dimension()
                        passage_embeddings_list.append(np.zeros(embedding_dim))
        else:
            for chunk_id in self.passage_node_keys:
                if chunk_id in self.graph_store.chunk_embeddings:
                    passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                else:
                    if embedding_dim is None:
                        if passage_embeddings_list:
                            embedding_dim = len(passage_embeddings_list[0])
                        else:
                            embedding_dim = self.graph_store.embedding_model.get_embedding_dimension()
                    passage_embeddings_list.append(np.zeros(embedding_dim))

        if passage_embeddings_list:
            self.passage_embeddings_array = np.array(passage_embeddings_list, dtype=np.float32)
        else:
            self.passage_embeddings_array = np.array([], dtype=np.float32)

        self._cached_owner_id = owner_str
        self._cached_store_version = current_store_version

        logger.info("Built mappings for %d passage nodes", len(self.passage_node_keys))
