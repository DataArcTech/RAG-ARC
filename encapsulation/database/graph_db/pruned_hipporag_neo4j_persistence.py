import os
import logging
import pickle
from typing import List, Dict, Any

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

        # 1. Save chunk embeddings
        embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.chunk_embeddings, f)
        logger.info(f"Saved chunk embeddings to {embeddings_path}")

        # 2. Save FAISS indices
        fact_index_path = os.path.join(path, 'fact_index')
        self.fact_faiss_db.save_index(fact_index_path, 'index')
        logger.info(f"Saved fact index to {fact_index_path}")

        entity_index_path = os.path.join(path, 'entity_index')
        self.entity_faiss_db.save_index(entity_index_path, 'index')
        logger.info(f"Saved entity index to {entity_index_path}")

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
        # 1. Load chunk embeddings
        embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                loaded = pickle.load(f)
            with self.write_lock():
                self.chunk_embeddings = loaded
                self._chunk_embeddings_array = None  # Mark for rebuild
            logger.info(f"Loaded chunk embeddings from {embeddings_path}")
        else:
            logger.warning(f"Chunk embeddings file not found: {embeddings_path}")

        # 2. Load FAISS indices
        fact_index_path = os.path.join(path, 'fact_index')
        if os.path.exists(fact_index_path):
            self.fact_faiss_db.load_index(fact_index_path)
            logger.info(f"Loaded fact index from {fact_index_path}")
        else:
            logger.warning(f"Fact index not found: {fact_index_path}")

        entity_index_path = os.path.join(path, 'entity_index')
        if os.path.exists(entity_index_path):
            self.entity_faiss_db.load_index(entity_index_path)
            logger.info(f"Loaded entity index from {entity_index_path}")
        else:
            logger.warning(f"Entity index not found: {entity_index_path}")

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

