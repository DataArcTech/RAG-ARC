import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _NodeMappingsCacheKey:
    owner_id: Optional[str]
    store_version: Union[int, str, None]


@dataclass(frozen=True, slots=True)
class _NodeMappings:
    owner_id: Optional[str]
    store_version: Union[int, str, None]
    passage_node_keys: list[str]
    passage_node_source_file_ids: list[Optional[str]]
    passage_embeddings_array: np.ndarray


class _PrunedHippoRAGNeo4jCacheMixin:
    def _get_tls(self) -> threading.local:
        tls = getattr(self, "_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(self, "_tls", tls)
        return tls

    @property
    def passage_node_keys(self) -> list[str]:
        return getattr(self._get_tls(), "passage_node_keys", [])

    @passage_node_keys.setter
    def passage_node_keys(self, value: list[str]) -> None:
        setattr(self._get_tls(), "passage_node_keys", value)

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

    def _node_mappings_cache_lock(self) -> threading.RLock:
        lock = getattr(self, "_node_mappings_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_node_mappings_lock", lock)
        return lock

    def _node_mappings_cache(self) -> dict[_NodeMappingsCacheKey, _NodeMappings]:
        cache = getattr(self, "_node_mappings_cache_dict", None)
        if cache is None:
            cache = {}
            setattr(self, "_node_mappings_cache_dict", cache)
        return cache

    def _node_mappings_cache_lru(self) -> list[_NodeMappingsCacheKey]:
        lru = getattr(self, "_node_mappings_cache_lru_list", None)
        if lru is None:
            lru = []
            setattr(self, "_node_mappings_cache_lru_list", lru)
        return lru

    def _node_mappings_cache_max_entries(self) -> int:
        """
        Maximum number of (owner_id, store_version) node-mapping entries to cache.

        Kept configurable to avoid uncontrolled in-process global state.
        """
        cfg = getattr(self, "config", None)
        raw = getattr(cfg, "node_mappings_cache_max_entries", None) if cfg is not None else None
        if raw is None:
            # If the retriever wasn't constructed with a config, default to *disabled* to avoid
            # uncontrolled global state.
            return 0
        try:
            n = int(raw)
        except Exception:  # noqa: BLE001
            return 0
        return max(0, n)

    def _node_mappings_cache_touch(self, key: _NodeMappingsCacheKey) -> None:
        lru = self._node_mappings_cache_lru()
        try:
            lru.remove(key)
        except ValueError:
            return
        lru.append(key)

    def _node_mappings_cache_get(self, key: _NodeMappingsCacheKey) -> Optional[_NodeMappings]:
        cache = self._node_mappings_cache()
        item = cache.get(key)
        if item is None:
            return None
        self._node_mappings_cache_touch(key)
        return item

    def _node_mappings_cache_put(self, key: _NodeMappingsCacheKey, item: _NodeMappings) -> None:
        cache = self._node_mappings_cache()
        cache[key] = item
        lru = self._node_mappings_cache_lru()
        try:
            lru.remove(key)
        except ValueError:
            pass
        lru.append(key)

        max_entries = self._node_mappings_cache_max_entries()
        if max_entries <= 0:
            return
        while len(lru) > max_entries:
            victim = lru.pop(0)
            cache.pop(victim, None)

    def invalidate_cache(self):
        """Force invalidation of all cached data."""
        self._cached_owner_id = None
        self._cached_store_version = None
        self.passage_node_keys = []
        self.passage_node_source_file_ids = []
        self.passage_embeddings_array = None
        with self._node_mappings_cache_lock():
            self._node_mappings_cache().clear()
            self._node_mappings_cache_lru().clear()

    @staticmethod
    def _owner_to_str(owner_id: Optional[uuid.UUID]) -> Optional[str]:
        return str(owner_id) if owner_id is not None else None

    def _compute_node_mappings(self, *, owner_str: Optional[str], store_version: Union[int, str, None]) -> _NodeMappings:
        """
        Compute the heavy node-mapping artifacts (chunk id list + embeddings array).

        NOTE: This is intentionally side-effect free. Callers can store the result
        into an in-process cache and bind it into TLS per request thread.
        """
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
        chunk_ids: list[tuple[str, Optional[str]]] = []
        source_file_ids: set[str] = set()
        for record in results:
            chunk_id = record["chunk_id"]
            source_file_id = record.get("source_file_id")
            if source_file_id:
                source_file_ids.add(source_file_id)
            chunk_ids.append((chunk_id, source_file_id))

        # Batch check file status if we have source_file_ids
        deleted_file_ids: set[str] = set()
        if source_file_ids:
            try:
                from framework.register import Register

                registrator = Register()
                knowledge_module = getattr(registrator, "registrations", {}).get("knowledge")
                if knowledge_module:
                    for file_id in source_file_ids:
                        if not knowledge_module.is_file_active(file_id):
                            deleted_file_ids.add(file_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to check file status for filtering: %s", e)

        # Filter chunks: keep only those from active files or without source_file_id
        filtered_pairs = [
            (chunk_id, source_file_id)
            for chunk_id, source_file_id in chunk_ids
            if not source_file_id or source_file_id not in deleted_file_ids
        ]
        passage_node_keys = [chunk_id for chunk_id, _fid in filtered_pairs]
        passage_node_source_file_ids = [str(fid) if fid is not None else None for _cid, fid in filtered_pairs]

        passage_embeddings_list: list[np.ndarray] = []
        embedding_dim: int | None = None

        # Avoid probing remote embedding endpoints for dimension during startup.
        # Prefer deriving dimension from any already-loaded chunk embedding in the graph store.
        def _infer_embedding_dim_from_store() -> int | None:
            try:
                items = getattr(self.graph_store, "chunk_embeddings", None)
                if isinstance(items, dict) and items:
                    any_vec = next(iter(items.values()))
                    if isinstance(any_vec, (list, tuple)) and any_vec:
                        return int(len(any_vec))
                    try:
                        return int(getattr(any_vec, "shape", [None])[0])
                    except Exception:  # noqa: BLE001
                        return None
            except Exception:  # noqa: BLE001
                return None
            return None

        read_lock = getattr(self.graph_store, "read_lock", None)
        if callable(read_lock):
            with self.graph_store.read_lock():
                for chunk_id in passage_node_keys:
                    if chunk_id in self.graph_store.chunk_embeddings:
                        passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                    else:
                        if embedding_dim is None:
                            if passage_embeddings_list:
                                embedding_dim = len(passage_embeddings_list[0])
                            else:
                                embedding_dim = _infer_embedding_dim_from_store() or self.graph_store.embedding_model.get_embedding_dimension()
                        passage_embeddings_list.append(np.zeros(embedding_dim))
        else:
            for chunk_id in passage_node_keys:
                if chunk_id in self.graph_store.chunk_embeddings:
                    passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
                else:
                    if embedding_dim is None:
                        if passage_embeddings_list:
                            embedding_dim = len(passage_embeddings_list[0])
                        else:
                            embedding_dim = _infer_embedding_dim_from_store() or self.graph_store.embedding_model.get_embedding_dimension()
                    passage_embeddings_list.append(np.zeros(embedding_dim))

        if passage_embeddings_list:
            passage_embeddings_array = np.array(passage_embeddings_list, dtype=np.float32)
        else:
            passage_embeddings_array = np.array([], dtype=np.float32)

        return _NodeMappings(
            owner_id=owner_str,
            store_version=store_version,
            passage_node_keys=passage_node_keys,
            passage_node_source_file_ids=passage_node_source_file_ids,
            passage_embeddings_array=passage_embeddings_array,
        )

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None, force_rebuild: bool = False):
        """Build mappings between passage nodes and their IDs from Neo4j."""
        current_store_version = self.graph_store.get_cache_version()
        owner_str = self._owner_to_str(owner_id)
        cache_key = _NodeMappingsCacheKey(owner_id=owner_str, store_version=current_store_version)

        # Fast path: request thread already has the right mapping view.
        if (
            not force_rebuild
            and self._cached_owner_id == owner_str
            and self._cached_store_version == current_store_version
            and self.passage_embeddings_array is not None
        ):
            logger.debug("Using TLS node mappings for %d passage nodes", len(self.passage_node_keys))
            return

        # Shared cache (across request threads) to avoid rebuilding per thread.
        with self._node_mappings_cache_lock():
            if not force_rebuild:
                cached = self._node_mappings_cache_get(cache_key)
                if cached is not None:
                    self.passage_node_keys = cached.passage_node_keys
                    self.passage_node_source_file_ids = cached.passage_node_source_file_ids
                    self.passage_embeddings_array = cached.passage_embeddings_array
                    self._cached_owner_id = cached.owner_id
                    self._cached_store_version = cached.store_version
                    logger.debug("Using shared cached node mappings for %d passage nodes", len(self.passage_node_keys))
                    return

        # Heavy build outside the lock to avoid blocking unrelated owners/requests.
        built = self._compute_node_mappings(owner_str=owner_str, store_version=current_store_version)

        with self._node_mappings_cache_lock():
            if not force_rebuild:
                cached = self._node_mappings_cache_get(cache_key)
                if cached is not None:
                    built = cached
                else:
                    if self._cached_store_version != current_store_version:
                        logger.info(
                            "Cache version changed (%s -> %s), rebuilding...",
                            self._cached_store_version,
                            current_store_version,
                        )
                    self._node_mappings_cache_put(cache_key, built)
            else:
                self._node_mappings_cache_put(cache_key, built)

        # Bind into TLS for this request thread.
        self.passage_node_keys = built.passage_node_keys
        self.passage_node_source_file_ids = built.passage_node_source_file_ids
        self.passage_embeddings_array = built.passage_embeddings_array
        self._cached_owner_id = built.owner_id
        self._cached_store_version = built.store_version

        logger.info("Built mappings for %d passage nodes", len(self.passage_node_keys))
