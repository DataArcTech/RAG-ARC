import os
import logging
import pickle
from typing import List, Dict, Any

from core.utils.path_guard import safe_leaf_name

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jPersistenceMixin:
    def save_index(self, path: str, name: str = "index") -> None:
        """
        Persist the graph database to filesystem.

        This method saves:
        1. Chunk embeddings as pickle
        2. FAISS indices for facts and entities
        3. Neo4j data (already persisted automatically)

        Args:
            path: Directory path to save the index
            name: Base name for index files
        """
        os.makedirs(path, exist_ok=True)

        # 1. Save chunk embeddings (owner-sharded when enabled).
        save_chunks = getattr(self, "_save_chunk_embeddings", None)
        if callable(save_chunks):
            save_chunks(base_path=path, name=name)
        else:
            embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
            with open(embeddings_path, "wb") as f:
                pickle.dump(self.chunk_embeddings, f)
            logger.info("Saved chunk embeddings to %s", embeddings_path)

        # 2. Save FAISS indices
        # IMPORTANT (performance): do NOT scan and load all owner-scoped FAISS indexes on disk here.
        # `iter_owner_scoped_faiss_dbs()` walks the filesystem and will eagerly load every owner index
        # into memory, which can OOM on multi-owner deployments. Persist only what this process has
        # actually touched (in-memory caches), plus the template/global indices.

        # 2.1 Save template/global indices (legacy + still used by some graph ops).
        try:
            fact_db = getattr(self, "fact_faiss_db", None)
            if fact_db is not None:
                fact_index_path = os.path.join(path, "fact_index")
                fact_db.save_index(fact_index_path, "index")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save template fact index: %s", exc)

        try:
            entity_db = getattr(self, "entity_faiss_db", None)
            if entity_db is not None:
                entity_index_path = os.path.join(path, "entity_index")
                entity_db.save_index(entity_index_path, "index")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save template entity index: %s", exc)

        # 2.2 Save loaded owner-scoped indices (only those instantiated in this process).
        saved = 0
        fact_map = getattr(self, "_owner_scoped_fact_faiss", None)
        if isinstance(fact_map, dict):
            for owner_leaf, db in fact_map.items():
                owner = None if owner_leaf == "__GLOBAL__" else owner_leaf
                owner_leaf2 = safe_leaf_name(owner or "__GLOBAL__", default="__GLOBAL__")
                fact_index_path = os.path.join(path, "fact_index", owner_leaf2)
                try:
                    db.save_index(fact_index_path, "index")
                    saved += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to save owner-scoped fact index (owner=%s): %s", owner, exc)
        logger.info("Saved %s loaded owner-scoped fact indexes under %s", saved, os.path.join(path, "fact_index"))

        saved = 0
        entity_map = getattr(self, "_owner_scoped_entity_faiss", None)
        if isinstance(entity_map, dict):
            for owner_leaf, db in entity_map.items():
                owner = None if owner_leaf == "__GLOBAL__" else owner_leaf
                owner_leaf2 = safe_leaf_name(owner or "__GLOBAL__", default="__GLOBAL__")
                entity_index_path = os.path.join(path, "entity_index", owner_leaf2)
                try:
                    db.save_index(entity_index_path, "index")
                    saved += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to save owner-scoped entity index (owner=%s): %s", owner, exc)
        logger.info("Saved %s loaded owner-scoped entity indexes under %s", saved, os.path.join(path, "entity_index"))

        logger.info(f"Index saved to {path}")
        logger.info("Note: Neo4j data is persisted automatically in the database")

    def load_index(self, path: str, name: str = "index") -> None:
        """
        Load persisted graph database from filesystem.

        This method loads:
        1. Chunk embeddings from pickle
        2. FAISS indices for facts and entities
        3. Neo4j data (already persisted in database)
        4. Reloads graph cache from Neo4j

        Args:
            path: Directory path to load the index from
            name: Base name for index files
        """
        # 1. Load chunk embeddings (owner-sharded when enabled).
        load_chunks = getattr(self, "_load_chunk_embeddings", None)
        if callable(load_chunks):
            # `_load_chunk_embeddings` reads from self.storage_path, so temporarily point it at `path`.
            original = getattr(self, "storage_path", None)
            try:
                setattr(self, "storage_path", str(path))
                load_chunks()
            finally:
                if original is not None:
                    setattr(self, "storage_path", original)
        else:
            embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "rb") as f:
                    loaded = pickle.load(f)
                with self.write_lock():
                    self.chunk_embeddings = loaded
                    self._chunk_embeddings_array = None  # Mark for rebuild
                logger.info("Loaded chunk embeddings from %s", embeddings_path)
            else:
                logger.warning("Chunk embeddings file not found: %s", embeddings_path)

        # 2. Load FAISS indices (owner-scoped directories)
        def _load_owner_scoped(kind: str) -> int:
            kind_root = os.path.join(path, f"{kind}_index")
            if not os.path.exists(kind_root):
                logger.warning("%s index not found: %s", kind, kind_root)
                return 0

            # Legacy layout: files directly under kind_root.
            legacy_pkl = os.path.join(kind_root, "index.pkl")
            legacy_faiss = os.path.join(kind_root, "index.faiss")
            loaded_any = 0
            if os.path.exists(legacy_pkl) and os.path.exists(legacy_faiss):
                owner_db = self.get_fact_faiss_db(None) if kind == "fact" else self.get_entity_faiss_db(None)
                owner_db.load_index(kind_root)
                dst = self._faiss_owner_scoped_dir(kind, owner_id=None)
                owner_db.save_index(dst, "index")
                loaded_any += 1

            if not os.path.isdir(kind_root):
                return loaded_any

            for owner_leaf in sorted(os.listdir(kind_root)):
                src = os.path.join(kind_root, owner_leaf)
                if not os.path.isdir(src) or owner_leaf.startswith("."):
                    continue
                owner = None if owner_leaf == "__GLOBAL__" else owner_leaf
                owner_db = self.get_fact_faiss_db(owner) if kind == "fact" else self.get_entity_faiss_db(owner)
                owner_db.load_index(src)
                dst = self._faiss_owner_scoped_dir(kind, owner_id=owner)
                owner_db.save_index(dst, "index")
                loaded_any += 1
            return loaded_any

        loaded_facts = _load_owner_scoped("fact")
        loaded_entities = _load_owner_scoped("entity")
        logger.info("Loaded %s fact indexes and %s entity indexes from %s", loaded_facts, loaded_entities, path)

        # 3. Reload graph cache from Neo4j (force reload)
        self._load_graph_cache(force_reload=True)
        
        # 4. Increment cache version to notify retrievers
        with self.write_lock():
            self._cache_version += 1
            cache_version = self._cache_version

        logger.info(f"Index loaded from {path} (cache_version={cache_version})")
        logger.info("Note: Neo4j data is loaded automatically from the database")

    def get_graph_db_info(self) -> Dict[str, Any]:
        """
        Return statistics or metadata about the graph database.

        Returns:
            Dictionary containing database statistics
        """
        # Count nodes (Facts are now relationships, not nodes)
        chunk_count_query = "MATCH (c:Chunk) RETURN count(c) AS count"
        entity_count_query = "MATCH (e:Entity) RETURN count(e) AS count"

        chunk_count = self._execute_query(chunk_count_query)[0]['count']
        entity_count = self._execute_query(entity_count_query)[0]['count']

        # Count relationships
        mentions_count_query = "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS count"
        fact_count_query = "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count"
        similar_count_query = "MATCH ()-[r:SIMILAR_TO]-() RETURN count(r) AS count"

        mentions_count = self._execute_query(mentions_count_query)[0]['count']
        fact_count = self._execute_query(fact_count_query)[0]['count']
        similar_count = self._execute_query(similar_count_query)[0]['count']

        # Get FAISS index sizes safely
        fact_index_size = 0
        entity_index_size = 0

        if self.fact_faiss_db and self.fact_faiss_db.index:
            fact_index_size = self.fact_faiss_db.index.ntotal

        if self.entity_faiss_db and self.entity_faiss_db.index:
            entity_index_size = self.entity_faiss_db.index.ntotal

        return {
            'database_type': 'Neo4j',
            'nodes': {
                'chunks': chunk_count,
                'entities': entity_count,
                'total': chunk_count + entity_count
            },
            'relationships': {
                'mentions': mentions_count,
                'facts': fact_count,
                'similar_to': similar_count,
                'total': mentions_count + fact_count + similar_count
            },
            'faiss_indices': {
                'facts': fact_index_size,
                'entities': entity_index_size
            },
            'chunk_embeddings': len(self.chunk_embeddings)
        }

    def __del__(self):
        """Close Neo4j driver on cleanup"""
        driver = getattr(self, "_driver", None)
        if driver:
            driver.close()
            logger.info("Neo4j driver closed")
