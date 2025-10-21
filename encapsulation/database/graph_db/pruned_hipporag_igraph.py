"""
Pruned HippoRAG Graph Store using igraph + FAISS + SQLite

Architecture:
- Fact索引：FAISS Flat (精确搜索，需要全量scores)
- Entity索引：FAISS HNSW (同义边计算，必须加速)
- Chunk索引：numpy数组 (子图规模小，暴力搜索最快)
- 元数据：SQLite (SQL查询、事务、引用计数)
- 图结构：igraph (PPR计算)
"""

import os
import json
import logging
import hashlib
import re
import pickle
import sqlite3
from typing import List, Dict, Any, Optional, Sequence, TYPE_CHECKING
from datetime import datetime
from collections import defaultdict
import numpy as np
import igraph as ig
import faiss

from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Chunk, GraphData
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig

logger = logging.getLogger(__name__)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute MD5 hash ID for content"""
    return prefix + hashlib.md5(content.encode()).hexdigest()


def text_processing(text):
    """
    Text processing function matching HippoRAG
    Removes special characters, converts to lowercase, and strips whitespace
    """
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


@shared_module
class PrunedHippoRAGIGraphStore(GraphStore):
    """
    Pruned HippoRAG Graph Store using igraph + FAISS + SQLite

    1. Fact使用FAISS Flat索引 (精确搜索)
    2. Entity使用FAISS HNSW索引 (同义边加速)
    3. Chunk使用numpy数组 (暴力搜索)
    4. 元数据使用SQLite (SQL查询、引用计数)
    5. 图结构使用igraph (PPR计算)
    """

    def __init__(self, config: "PrunedHippoRAGIGraphConfig"):
        """Initialize Pruned HippoRAG graph store"""
        super().__init__(config)
        
        # 1. Initialize embedding model
        self.embedding_model = config.embedding.build()
        
        # 2. Initialize igraph
        self.graph = ig.Graph(directed=False)
        
        # 3. Initialize FAISS indices (fact: Flat, entity: HNSW)
        self._init_faiss_indices()
        
        # 4. Initialize SQLite database
        self._init_sqlite_db()
        
        # 5. Initialize chunk embeddings (numpy array, not FAISS)
        self.chunk_embeddings = {}  # chunk_id -> embedding (dict)
        self._chunk_embeddings_array = None  # Pre-computed numpy array
        self._chunk_ids_list = None  # Corresponding chunk_id list
        
        # 6. Node mappings
        self.node_to_idx = {}  # node_id -> vertex index
        self.idx_to_node = {}  # vertex index -> node_id
        self.node_to_node_stats = defaultdict(float)  # (from_node, to_node) -> weight
        
        # 7. Storage configuration
        self.storage_path = getattr(config, 'storage_path', './data/graph_index')
        self.index_name = getattr(config, 'index_name', 'index')
        
        # 8. Synonymy edge configuration
        self.add_synonymy_edges = getattr(config, 'add_synonymy_edges', False)
        self.synonymy_edge_topk = getattr(config, 'synonymy_edge_topk', 100)
        self.synonymy_edge_sim_threshold = getattr(config, 'synonymy_edge_sim_threshold', 0.8)

        # 9. Lazy loading with dirty flag for graph cache invalidation
        self._graph_dirty = True  # Mark as dirty initially, will build on first access

        logger.info("Pruned HippoRAG graph store initialized")
        logger.info(f"  - Fact index: FAISS Flat (exact search)")
        logger.info(f"  - Entity index: FAISS HNSW (synonymy edges)")
        logger.info(f"  - Chunk index: numpy array (brute-force search)")
        logger.info(f"  - Metadata: SQLite")
        logger.info(f"  - Graph: igraph")

    def _init_faiss_indices(self):
        """Initialize FAISS indices (fact: Flat, entity: HNSW)"""
        from encapsulation.database.vector_db.faiss import FaissVectorDB
        from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
        
        storage_path = getattr(self.config, 'storage_path', './data/graph_index')
        os.makedirs(storage_path, exist_ok=True)
        
        # ✅ Fact index - use Flat (exact search for all facts)
        fact_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='flat',  # ✅ Flat, not HNSW
            metric='cosine',
            normalize_L2=True,
            index_path=os.path.join(storage_path, 'fact_index'),
            index_name='index'
        )
        self.fact_faiss_db = fact_config.build()
        
        # Try to load existing fact index
        fact_index_path = os.path.join(storage_path, 'fact_index')
        if os.path.exists(fact_index_path):
            try:
                self.fact_faiss_db.load_index(fact_index_path)
                logger.info(f"Loaded existing fact index: {self.fact_faiss_db.index.ntotal} facts")
            except Exception as e:
                logger.warning(f"Failed to load fact index: {e}")
        
        # ✅ Entity index - use HNSW (synonymy edge acceleration)
        entity_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='hnsw',  # ✅ HNSW, only place using approximate search
            metric='cosine',
            normalize_L2=True,
            m=getattr(self.config, 'hnsw_M', 32),
            efConstruction=getattr(self.config, 'hnsw_ef_construction', 200),
            efSearch=getattr(self.config, 'hnsw_ef_search', 100),
            index_path=os.path.join(storage_path, 'entity_index'),
            index_name='index'
        )
        self.entity_faiss_db = entity_config.build()
        
        # Try to load existing entity index
        entity_index_path = os.path.join(storage_path, 'entity_index')
        if os.path.exists(entity_index_path):
            try:
                self.entity_faiss_db.load_index(entity_index_path)
                logger.info(f"Loaded existing entity index: {self.entity_faiss_db.index.ntotal} entities")
            except Exception as e:
                logger.warning(f"Failed to load entity index: {e}")
        
        # ❌ Chunk does NOT use FAISS - use numpy array for brute-force search
        
        logger.info("FAISS indices initialized (fact: Flat, entity: HNSW)")

    def _init_sqlite_db(self):
        """Initialize SQLite database for metadata"""
        storage_path = getattr(self.config, 'storage_path', './data/graph_index')
        os.makedirs(storage_path, exist_ok=True)
        
        self.db_path = os.path.join(storage_path, 'metadata.db')
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                owner_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner_id)')
        
        # Entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_name TEXT NOT NULL,
                entity_type TEXT DEFAULT "Entity",
                attributes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity_name)')
        
        # Facts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                head TEXT NOT NULL,
                relation TEXT NOT NULL,
                tail TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_head ON facts(head)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_tail ON facts(tail)')
        
        # Chunk-Entity relations table (for reference counting)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_entity_relations (
                chunk_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (chunk_id, entity_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ce_entity ON chunk_entity_relations(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ce_chunk ON chunk_entity_relations(chunk_id)')

        # Synonymy edges table (for entity similarity edges)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonymy_edges (
                entity_id_1 TEXT NOT NULL,
                entity_id_2 TEXT NOT NULL,
                similarity REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_id_1, entity_id_2),
                FOREIGN KEY (entity_id_1) REFERENCES entities(entity_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id_2) REFERENCES entities(entity_id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_syn_entity1 ON synonymy_edges(entity_id_1)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_syn_entity2 ON synonymy_edges(entity_id_2)')

        self.conn.commit()
        logger.info(f"SQLite database initialized at {self.db_path}")

    def add_chunk(self, chunk: Chunk):
        """Add chunk to store"""
        chunk_id = chunk.id
        
        # 1. Store to SQLite
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chunks (chunk_id, content, owner_id, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            chunk_id,
            chunk.content,
            chunk.owner_id,
            json.dumps(chunk.metadata) if chunk.metadata else '{}'
        ))
        self.conn.commit()
        
        # 2. ✅ Generate and store embedding (use dict, not FAISS)
        if chunk_id not in self.chunk_embeddings:
            embedding = self.embedding_model.embed(chunk.content)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            # Normalize for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            self.chunk_embeddings[chunk_id] = embedding
            
            # Mark array needs rebuild
            self._chunk_embeddings_array = None
        
        # 3. Add to graph
        if chunk_id not in self.node_to_idx:
            vertex_idx = self.graph.vcount()
            self.graph.add_vertex(name=chunk_id, node_type='chunk')
            self.node_to_idx[chunk_id] = vertex_idx
            self.idx_to_node[vertex_idx] = chunk_id
        
        logger.debug(f"Added chunk {chunk_id}")

    def add_graph_data(self, graph_data: GraphData, chunk_id: str) -> List[str]:
        """
        Add graph data (entities and relations) from extraction
        
        Returns:
            List of new entity IDs
        """
        # Process triples
        processed_triples = []
        for relation in graph_data.relations:
            if len(relation) >= 3:
                processed_triple = [
                    text_processing(relation[0]),
                    text_processing(relation[1]),
                    text_processing(relation[2])
                ]
                if processed_triple[0] and processed_triple[2]:
                    processed_triples.append(processed_triple)
        
        # Extract entities from triples
        triple_entities = set()
        for triple in processed_triples:
            triple_entities.add(triple[0])  # head
            triple_entities.add(triple[2])  # tail
        
        new_entity_ids = []
        cursor = self.conn.cursor()
        
        # Add entities
        for entity_name in triple_entities:
            entity_id = compute_mdhash_id(entity_name, prefix='entity-')
            
            # Check if entity exists
            cursor.execute('SELECT entity_id FROM entities WHERE entity_id = ?', (entity_id,))
            exists = cursor.fetchone()
            
            if not exists:
                # Insert to SQLite
                cursor.execute('''
                    INSERT INTO entities (entity_id, entity_name, entity_type, attributes)
                    VALUES (?, ?, ?, ?)
                ''', (entity_id, entity_name, 'Entity', '{}'))
                
                # Add to graph
                if entity_id not in self.node_to_idx:
                    vertex_idx = self.graph.vcount()
                    self.graph.add_vertex(name=entity_id, node_type='entity', entity_name=entity_name)
                    self.node_to_idx[entity_id] = vertex_idx
                    self.idx_to_node[vertex_idx] = entity_id
                
                new_entity_ids.append(entity_id)
            
            # Add chunk-entity relation
            cursor.execute('''
                INSERT OR IGNORE INTO chunk_entity_relations (chunk_id, entity_id, weight)
                VALUES (?, ?, ?)
            ''', (chunk_id, entity_id, 1.0))

            # Incrementally add chunk-entity edge to graph
            self._add_edge_to_graph(chunk_id, entity_id, 1.0)
        
        # Add facts
        for triple in processed_triples:
            head_name, relation_type, tail_name = triple[0], triple[1], triple[2]
            head_id = compute_mdhash_id(head_name, prefix='entity-')
            tail_id = compute_mdhash_id(tail_name, prefix='entity-')
            
            fact_tuple = tuple(triple)
            fact_text = str(fact_tuple)
            fact_id = compute_mdhash_id(fact_text, prefix='fact-')
            
            # Check if fact exists
            cursor.execute('SELECT fact_id FROM facts WHERE fact_id = ?', (fact_id,))
            exists = cursor.fetchone()
            
            if not exists:
                # Insert to SQLite
                cursor.execute('''
                    INSERT INTO facts (fact_id, head, relation, tail, text)
                    VALUES (?, ?, ?, ?, ?)
                ''', (fact_id, head_name, relation_type, tail_name, fact_text))

                # Incrementally add entity-entity edge to graph
                self._add_edge_to_graph(head_id, tail_id, 1.0)

        self.conn.commit()
        logger.debug(f"Added graph data for chunk {chunk_id}: {len(triple_entities)} entities, {len(processed_triples)} facts")

        return new_entity_ids

    def _add_edge_to_graph(self, from_node: str, to_node: str, weight: float):
        """
        Incrementally add edge to graph (bidirectional for undirected graph)

        This method adds edges to both node_to_node_stats and the igraph object,
        enabling true incremental updates without needing to rebuild the entire graph.
        """
        if from_node == to_node:
            return

        if from_node not in self.node_to_idx or to_node not in self.node_to_idx:
            return

        from_idx = self.node_to_idx[from_node]
        to_idx = self.node_to_idx[to_node]

        # Check if edge already exists in node_to_node_stats
        edge_key_1 = (from_node, to_node)
        edge_key_2 = (to_node, from_node)

        edge_exists = edge_key_1 in self.node_to_node_stats or edge_key_2 in self.node_to_node_stats

        # Update statistics (bidirectional)
        self.node_to_node_stats[edge_key_1] += weight
        self.node_to_node_stats[edge_key_2] += weight

        # Add edge to igraph if it doesn't exist yet
        # For undirected graph, we only need to add one edge
        if not edge_exists:
            try:
                self.graph.add_edge(from_idx, to_idx, weight=weight)
            except Exception as e:
                logger.warning(f"Failed to add edge {from_node} -> {to_node}: {e}")

    def _add_edge_weight(self, from_node: str, to_node: str, weight: float):
        """
        Legacy method - now just calls _add_edge_to_graph
        Kept for backward compatibility
        """
        self._add_edge_to_graph(from_node, to_node, weight)

    def batch_generate_embeddings(self):
        """Batch generate embeddings and add to FAISS"""
        logger.info("Batch generating embeddings...")

        cursor = self.conn.cursor()

        # 1. ✅ Generate entity embeddings and add to FAISS HNSW
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entities = cursor.fetchall()

        new_entities = []
        for entity_id, entity_name in entities:
            # Check if already in FAISS
            if entity_id not in self.entity_faiss_db.docstore:
                new_entities.append(Chunk(
                    id=entity_id,
                    content=entity_name,
                    metadata={'type': 'entity'}
                ))

        if new_entities:
            logger.info(f"Adding {len(new_entities)} entities to FAISS HNSW...")
            # Generate embeddings
            entity_texts = [chunk.content for chunk in new_entities]
            entity_embeddings = self.embedding_model.embed(entity_texts)

            # Store embeddings in chunk metadata BEFORE adding to FAISS
            for chunk, embedding in zip(new_entities, entity_embeddings):
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                chunk.metadata['embedding'] = embedding

            # Now add to FAISS (embeddings will be regenerated, but metadata is preserved)
            self.entity_faiss_db.update_index(new_entities)
            entity_index_path = os.path.join(self.storage_path, 'entity_index')
            self.entity_faiss_db.save_index(entity_index_path, 'index')
            logger.info(f"Saved entity index to {entity_index_path}")

        # 2. ✅ Generate fact embeddings and add to FAISS Flat
        cursor.execute('SELECT fact_id, text FROM facts')
        facts = cursor.fetchall()

        new_facts = []
        for fact_id, fact_text in facts:
            if fact_id not in self.fact_faiss_db.docstore:
                new_facts.append(Chunk(
                    id=fact_id,
                    content=fact_text,
                    metadata={'type': 'fact'}
                ))

        if new_facts:
            logger.info(f"Adding {len(new_facts)} facts to FAISS Flat...")
            self.fact_faiss_db.update_index(new_facts)
            fact_index_path = os.path.join(self.storage_path, 'fact_index')
            self.fact_faiss_db.save_index(fact_index_path, 'index')
            logger.info(f"Saved fact index to {fact_index_path}")

        # 3. ❌ Chunk embeddings already generated in add_chunk, no FAISS needed

        logger.info("Batch embedding generation completed!")

    def build_graph(self):
        """Build graph from SQLite data (including synonymy edges if available)"""
        logger.info("Building graph from SQLite data...")

        # Clear existing graph and rebuild from scratch
        self.graph = ig.Graph(directed=False)
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_to_node_stats = defaultdict(float)

        cursor = self.conn.cursor()

        # Add chunk nodes
        cursor.execute('SELECT chunk_id FROM chunks ORDER BY ROWID')
        chunks = cursor.fetchall()
        for (chunk_id,) in chunks:
            if chunk_id not in self.node_to_idx:
                vertex_idx = self.graph.vcount()
                self.graph.add_vertex(name=chunk_id, node_type='chunk')
                self.node_to_idx[chunk_id] = vertex_idx
                self.idx_to_node[vertex_idx] = chunk_id

        # Add entity nodes
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entities = cursor.fetchall()
        for entity_id, entity_name in entities:
            if entity_id not in self.node_to_idx:
                vertex_idx = self.graph.vcount()
                self.graph.add_vertex(name=entity_id, node_type='entity', entity_name=entity_name)
                self.node_to_idx[entity_id] = vertex_idx
                self.idx_to_node[vertex_idx] = entity_id

        # Collect all edges in node_to_node_stats
        # 1. Add chunk-entity edges
        cursor.execute('SELECT chunk_id, entity_id, weight FROM chunk_entity_relations')
        relations = cursor.fetchall()
        for chunk_id, entity_id, weight in relations:
            # Update statistics (bidirectional)
            self.node_to_node_stats[(chunk_id, entity_id)] += weight
            self.node_to_node_stats[(entity_id, chunk_id)] += weight

        # 2. Add entity-entity edges from facts
        # Build entity name to ID mapping for fast lookup
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_name_to_id = {name: eid for eid, name in cursor.fetchall()}

        cursor.execute('SELECT head, tail FROM facts')
        facts = cursor.fetchall()
        for head_name, tail_name in facts:
            head_id = entity_name_to_id.get(head_name)
            tail_id = entity_name_to_id.get(tail_name)
            if head_id and tail_id and head_id != tail_id:
                # Update statistics (bidirectional)
                self.node_to_node_stats[(head_id, tail_id)] += 1.0
                self.node_to_node_stats[(tail_id, head_id)] += 1.0

        # 3. Add synonymy edges (if table exists and has data)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synonymy_edges'")
        if cursor.fetchone():
            cursor.execute('SELECT entity_id_1, entity_id_2, similarity FROM synonymy_edges')
            synonymy_edges = cursor.fetchall()
            if synonymy_edges:
                logger.info(f"Loading {len(synonymy_edges)} unidirectional synonymy edges from SQLite...")
                for entity_id_1, entity_id_2, similarity in synonymy_edges:
                    # ✅ FIXED: Keep synonymy edges UNIDIRECTIONAL (same as standard version)
                    # SQLite stores unidirectional edges, we keep them unidirectional in node_to_node_stats
                    # This is different from chunk-entity and entity-fact edges which are bidirectional
                    self.node_to_node_stats[(entity_id_1, entity_id_2)] = similarity

        # Build graph from node_to_node_stats
        # ✅ FIXED: Directly add all edges from node_to_node_stats (same as standard version)
        # No need to filter by from_idx < to_idx because:
        # - Chunk-entity edges are already bidirectional in node_to_node_stats
        # - Entity-fact edges are already bidirectional in node_to_node_stats
        # - Synonymy edges are unidirectional in node_to_node_stats
        valid_edges = []
        edge_weights = []

        for (from_node, to_node), weight in self.node_to_node_stats.items():
            if from_node in self.node_to_idx and to_node in self.node_to_idx:
                from_idx = self.node_to_idx[from_node]
                to_idx = self.node_to_idx[to_node]
                valid_edges.append((from_idx, to_idx))
                edge_weights.append(weight)

        # Add edges to graph
        if valid_edges:
            self.graph.add_edges(valid_edges)
            self.graph.es['weight'] = edge_weights

        logger.info(f"Graph built: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")

    def _add_synonymy_edges(self):
        """Add synonymy edges using FAISS HNSW and save to SQLite"""
        if not self.add_synonymy_edges:
            logger.info("Synonymy edges disabled")
            return

        from tqdm import tqdm
        logger.info("Computing synonymy edges using FAISS HNSW...")

        cursor = self.conn.cursor()

        # Clear existing synonymy edges
        cursor.execute('DELETE FROM synonymy_edges')
        self.conn.commit()

        # Get all entities
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entities = cursor.fetchall()

        if not entities:
            logger.warning("No entities found, skipping synonymy edge addition")
            return

        # Build entity ID to name mapping for fast lookup
        entity_id_to_name = {eid: name for eid, name in entities}

        # Build a set to track existing entity-entity edges (fact edges only)
        # We only check entity-entity edges to avoid duplicates, not chunk-entity edges
        existing_entity_entity_edges = set()

        # Add fact edges (entity-entity edges)
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_name_to_id = {name: eid for eid, name in cursor.fetchall()}

        cursor.execute('SELECT head, tail FROM facts')
        for head_name, tail_name in cursor.fetchall():
            head_id = entity_name_to_id.get(head_name)
            tail_id = entity_name_to_id.get(tail_name)
            if head_id and tail_id:
                existing_entity_entity_edges.add((head_id, tail_id))
                existing_entity_entity_edges.add((tail_id, head_id))

        logger.info(f"Built existing entity-entity edge set with {len(existing_entity_entity_edges)} directional edges")

        num_synonym_edges = 0
        edges_to_add = []  # Batch collect edges for SQLite

        # Pre-extract and normalize all embeddings for batch search
        logger.info("Preparing embeddings for batch FAISS search...")
        valid_entities = []
        embeddings_list = []

        for entity_id, entity_name in entities:
            # Filter short entities (same as original)
            if len(re.sub('[^A-Za-z0-9]', '', entity_name)) <= 2:
                continue

            # Get entity from FAISS docstore
            entity_chunk = self.entity_faiss_db.docstore.get(entity_id)
            if not entity_chunk:
                continue

            # Get embedding from docstore
            embedding = entity_chunk.metadata.get('embedding')
            if embedding is None:
                continue  # Skip if no embedding

            if isinstance(embedding, list):
                embedding = np.array(embedding).astype(np.float32)
            else:
                embedding = embedding.astype(np.float32)

            valid_entities.append((entity_id, entity_name))
            embeddings_list.append(embedding)

        if not valid_entities:
            logger.warning("No valid entities for synonymy edge computation")
            return

        # Batch normalize embeddings
        embeddings_array = np.array(embeddings_list).astype(np.float32)
        if self.entity_faiss_db.config.normalize_L2 or self.entity_faiss_db.config.metric == "cosine":
            faiss.normalize_L2(embeddings_array)

        logger.info(f"Prepared {len(valid_entities)} valid entities for synonymy edge computation")

        # Batch FAISS search
        logger.info("Performing batch FAISS search...")
        k = min(self.synonymy_edge_topk, self.entity_faiss_db.index.ntotal)
        distances_batch, indices_batch = self.entity_faiss_db.index.search(embeddings_array, k)
        logger.info("Batch FAISS search completed")

        # Process results
        logger.info("Processing search results...")

        # Debug: Log first entity's results
        if len(valid_entities) > 0:
            first_entity_id, first_entity_name = valid_entities[0]
            first_distances = distances_batch[0]
            first_indices = indices_batch[0]
            logger.info(f"DEBUG: First entity '{first_entity_name}' top-5 neighbors:")
            for j in range(min(5, len(first_distances))):
                if first_indices[j] != -1 and first_indices[j] in self.entity_faiss_db.index_to_docstore_id:
                    neighbor_id = self.entity_faiss_db.index_to_docstore_id[first_indices[j]]
                    neighbor_name = entity_id_to_name.get(neighbor_id, "Unknown")
                    logger.info(f"  {j+1}. {neighbor_name}: distance={first_distances[j]:.4f}")

        for i, ((entity_id, entity_name), distances, indices) in enumerate(tqdm(
            zip(valid_entities, distances_batch, indices_batch),
            total=len(valid_entities),
            desc="Computing synonymy edges"
        )):
            # Log progress every 1000 entities
            if i > 0 and i % 1000 == 0:
                logger.info(f"Processed {i}/{len(valid_entities)} entities, found {num_synonym_edges} synonymy edges so far")

            # Process results (same as original version)
            num_added = 0
            for idx, distance in zip(indices, distances):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue

                # Get neighbor entity ID from index
                if idx not in self.entity_faiss_db.index_to_docstore_id:
                    continue

                neighbor_entity_id = self.entity_faiss_db.index_to_docstore_id[idx]

                # Skip deleted entities
                if neighbor_entity_id in self.entity_faiss_db.deleted_ids:
                    continue

                # Skip self
                if neighbor_entity_id == entity_id:
                    continue

                # Get neighbor name for validation (from cache, not SQLite)
                neighbor_name = entity_id_to_name.get(neighbor_entity_id)
                if not neighbor_name:
                    continue

                # FAISS with metric='cosine' returns NEGATIVE inner product
                # For normalized vectors: inner_product = cosine_similarity
                # So: distance = -cosine_similarity
                # Therefore: cosine_similarity = -distance
                similarity = -float(distance)

                # Check threshold (same as original)
                if similarity < self.synonymy_edge_sim_threshold:
                    break  # Distances are sorted, can break early

                # Check if edge already exists in EITHER direction (same as standard version)
                edge_key = (entity_id, neighbor_entity_id)
                reverse_edge_key = (neighbor_entity_id, entity_id)

                if edge_key not in existing_entity_entity_edges and reverse_edge_key not in existing_entity_entity_edges:
                    # Add UNIDIRECTIONAL edge (only one direction to avoid duplication)
                    edges_to_add.append((entity_id, neighbor_entity_id, similarity))
                    num_synonym_edges += 1
                    num_added += 1

                    # Mark BOTH directions as added to avoid duplicates
                    existing_entity_entity_edges.add(edge_key)
                    existing_entity_entity_edges.add(reverse_edge_key)

                # Limit max neighbors per entity (same as original)
                if num_added >= 100:
                    break

        # Batch insert all edges to SQLite and incrementally add to graph
        if edges_to_add:
            logger.info(f"Saving {len(edges_to_add)} directional synonymy edges to SQLite and graph...")
            # edges_to_add already contains both directions, so we save all of them
            cursor.executemany(
                'INSERT OR REPLACE INTO synonymy_edges (entity_id_1, entity_id_2, similarity) VALUES (?, ?, ?)',
                edges_to_add
            )
            self.conn.commit()

            # Incrementally add synonymy edges to graph
            # Note: Synonymy edges are UNIDIRECTIONAL in storage but we add them as undirected edges
            for entity_id_1, entity_id_2, similarity in edges_to_add:
                # Add edge to node_to_node_stats (unidirectional, as stored in SQLite)
                self.node_to_node_stats[(entity_id_1, entity_id_2)] = similarity

                # Add edge to igraph (will be undirected since graph is undirected)
                if entity_id_1 in self.node_to_idx and entity_id_2 in self.node_to_idx:
                    idx_1 = self.node_to_idx[entity_id_1]
                    idx_2 = self.node_to_idx[entity_id_2]
                    try:
                        self.graph.add_edge(idx_1, idx_2, weight=similarity)
                    except Exception as e:
                        logger.warning(f"Failed to add synonymy edge {entity_id_1} -> {entity_id_2}: {e}")

            logger.info(f"Added {num_synonym_edges} unique synonymy edges ({len(edges_to_add)} directional edges)")
        else:
            logger.info("No synonymy edges to add")

    def _rebuild_chunk_embeddings_array(self):
        """Rebuild chunk embeddings array for brute-force search"""
        if self._chunk_embeddings_array is not None:
            return  # Already built

        logger.info("Rebuilding chunk embeddings array...")

        self._chunk_ids_list = list(self.chunk_embeddings.keys())
        embeddings_list = [self.chunk_embeddings[cid] for cid in self._chunk_ids_list]

        if embeddings_list:
            self._chunk_embeddings_array = np.array(embeddings_list)
            logger.info(f"Chunk embeddings array built: {len(self._chunk_ids_list)} chunks")
        else:
            self._chunk_embeddings_array = np.array([])
            logger.warning("No chunk embeddings found")

    # ========== GraphStore Interface Implementation ==========

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build graph from a list of Chunks"""
        logger.info(f"Building index from {len(chunks)} chunks...")

        for chunk in chunks:
            self.add_chunk(chunk)
            if chunk.graph and not chunk.graph.is_empty():
                self.add_graph_data(chunk.graph, chunk.id)

        # Batch generate embeddings
        self.batch_generate_embeddings()

        # Compute and save synonymy edges to SQLite (if enabled)
        if self.add_synonymy_edges:
            self._add_synonymy_edges()

        # Build graph from SQLite (including synonymy edges if available)
        self.build_graph()

        # Rebuild chunk embeddings array
        self._rebuild_chunk_embeddings_array()

        logger.info("Index building completed")

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update existing chunks' graphs in the database"""
        logger.info(f"Updating index with {len(chunks)} chunks...")

        try:
            # Step 1: Add chunks and graph data
            logger.info("Step 1: Adding chunks and graph data...")
            for i, chunk in enumerate(chunks):
                logger.info(f"  Processing chunk {i+1}/{len(chunks)}: {chunk.id}")
                self.add_chunk(chunk)
                if chunk.graph and not chunk.graph.is_empty():
                    logger.info(f"    Adding graph data: {len(chunk.graph.entities)} entities, {len(chunk.graph.relations)} relations")
                    self.add_graph_data(chunk.graph, chunk.id)
                else:
                    logger.warning(f"    Chunk {chunk.id} has no graph data")
            logger.info("Step 1 completed: All chunks and graph data added")

            # Step 2: Batch generate embeddings
            logger.info("Step 2: Batch generating embeddings...")
            self.batch_generate_embeddings()
            logger.info("Step 2 completed: Embeddings generated")

            # Step 3: Compute and save synonymy edges (if enabled)
            # Synonymy edges are added incrementally to the graph in _add_synonymy_edges()
            if self.add_synonymy_edges:
                logger.info("Step 3: Computing synonymy edges...")
                self._add_synonymy_edges()
                logger.info("Step 3 completed: Synonymy edges added")
            else:
                logger.info("Step 3 skipped: Synonymy edges disabled")

            # Step 4: Rebuild chunk embeddings array
            logger.info("Step 4: Rebuilding chunk embeddings array...")
            self._rebuild_chunk_embeddings_array()
            logger.info("Step 4 completed: Chunk embeddings array rebuilt")

            # Step 5: Final commit to ensure all changes are persisted
            logger.info("Step 5: Committing all changes to database...")
            self.conn.commit()
            logger.info("Step 5 completed: All changes committed")

            logger.info("✅ Index update completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update index: {e}", exc_info=True)
            self.conn.rollback()  # Rollback on error
            return False

    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete chunks and their graphs by IDs"""
        if ids is None or len(ids) == 0:
            logger.warning("No chunk IDs provided for deletion")
            return False

        return self.delete_chunks(ids)

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks and clean up orphan nodes"""
        logger.info(f"Deleting {len(chunk_ids)} chunks...")

        cursor = self.conn.cursor()

        # 1. Find entities that will become orphans
        placeholders = ','.join('?' * len(chunk_ids))

        # Find entities only referenced by these chunks
        cursor.execute(f'''
            SELECT entity_id
            FROM chunk_entity_relations
            WHERE entity_id IN (
                SELECT entity_id FROM chunk_entity_relations WHERE chunk_id IN ({placeholders})
            )
            GROUP BY entity_id
            HAVING COUNT(DISTINCT chunk_id) = SUM(CASE WHEN chunk_id IN ({placeholders}) THEN 1 ELSE 0 END)
        ''', chunk_ids + chunk_ids)

        orphan_entities = [row[0] for row in cursor.fetchall()]

        # 2. Delete chunk-entity relations
        cursor.execute(f'''
            DELETE FROM chunk_entity_relations
            WHERE chunk_id IN ({placeholders})
        ''', chunk_ids)

        # 3. Delete orphan entities and their facts
        if orphan_entities:
            entity_placeholders = ','.join('?' * len(orphan_entities))

            # Find facts involving orphan entities
            cursor.execute(f'''
                SELECT fact_id FROM facts
                WHERE head IN (SELECT entity_name FROM entities WHERE entity_id IN ({entity_placeholders}))
                   OR tail IN (SELECT entity_name FROM entities WHERE entity_id IN ({entity_placeholders}))
            ''', orphan_entities + orphan_entities)
            orphan_fact_ids = [row[0] for row in cursor.fetchall()]

            # Delete facts from FAISS
            if orphan_fact_ids:
                self.fact_faiss_db.delete_index(orphan_fact_ids)
                logger.info(f"Deleted {len(orphan_fact_ids)} orphan facts from FAISS")

            # Delete facts from SQLite
            cursor.execute(f'''
                DELETE FROM facts
                WHERE head IN (SELECT entity_name FROM entities WHERE entity_id IN ({entity_placeholders}))
                   OR tail IN (SELECT entity_name FROM entities WHERE entity_id IN ({entity_placeholders}))
            ''', orphan_entities + orphan_entities)

            # Delete entities from FAISS
            self.entity_faiss_db.delete_index(orphan_entities)
            logger.info(f"Deleted {len(orphan_entities)} orphan entities from FAISS")

            # Delete entities from SQLite
            cursor.execute(f'''
                DELETE FROM entities
                WHERE entity_id IN ({entity_placeholders})
            ''', orphan_entities)

        # 4. Delete chunks from SQLite
        cursor.execute(f'''
            DELETE FROM chunks
            WHERE chunk_id IN ({placeholders})
        ''', chunk_ids)

        self.conn.commit()

        # 5. ✅ Delete from chunk_embeddings (not FAISS)
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_embeddings:
                del self.chunk_embeddings[chunk_id]

        # Mark array needs rebuild
        self._chunk_embeddings_array = None

        # 6. Rebuild graph
        self.build_graph()

        logger.info(f"Deleted {len(chunk_ids)} chunks, {len(orphan_entities)} orphan entities")
        return True

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graphs"""
        if not confirm:
            logger.warning("delete_all_index requires confirm=True")
            return False

        logger.info("Deleting all index data...")

        cursor = self.conn.cursor()

        # Delete all tables
        cursor.execute('DELETE FROM chunk_entity_relations')
        cursor.execute('DELETE FROM facts')
        cursor.execute('DELETE FROM entities')
        cursor.execute('DELETE FROM chunks')
        self.conn.commit()

        # Clear FAISS indices
        # Note: FAISS doesn't have a clear method, so we recreate the indices
        self._init_faiss_indices()

        # Clear chunk embeddings
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None

        # Clear graph
        self.graph = ig.Graph(directed=False)
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_to_node_stats = defaultdict(float)

        logger.info("All index data deleted")
        return True

    def get_by_ids(self, ids: Sequence[str]) -> List[Chunk]:
        """Retrieve chunks (including their graphs) by IDs"""
        chunks = []
        cursor = self.conn.cursor()

        for chunk_id in ids:
            cursor.execute('SELECT chunk_id, content, owner_id, metadata FROM chunks WHERE chunk_id = ?', (chunk_id,))
            result = cursor.fetchone()

            if result:
                chunk_id, content, owner_id, metadata_str = result
                metadata = json.loads(metadata_str) if metadata_str else {}

                # Get graph data
                graph_data = self._get_graph_data(chunk_id)

                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    owner_id=owner_id,
                    metadata=metadata,
                    graph=graph_data
                )
                chunks.append(chunk)

        return chunks

    def _get_graph_data(self, chunk_id: str) -> GraphData:
        """Get graph data for a specific chunk"""
        cursor = self.conn.cursor()

        # Get entities for this chunk
        cursor.execute('''
            SELECT e.entity_id, e.entity_name, e.entity_type, e.attributes
            FROM entities e
            JOIN chunk_entity_relations cer ON e.entity_id = cer.entity_id
            WHERE cer.chunk_id = ?
        ''', (chunk_id,))

        entities = []
        entity_names = set()
        for entity_id, entity_name, entity_type, attributes_str in cursor.fetchall():
            entities.append({
                'id': entity_id,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'attributes': json.loads(attributes_str) if attributes_str else {}
            })
            entity_names.add(entity_name)

        # Get relations (facts) involving these entities
        if entity_names:
            entity_names_placeholders = ','.join('?' * len(entity_names))
            cursor.execute(f'''
                SELECT head, relation, tail
                FROM facts
                WHERE head IN ({entity_names_placeholders})
                  AND tail IN ({entity_names_placeholders})
            ''', list(entity_names) + list(entity_names))

            relations = []
            for head, relation, tail in cursor.fetchall():
                relations.append([head, relation, tail])
        else:
            relations = []

        return GraphData(entities=entities, relations=relations, metadata={})

    def save_index(self, path: str, name: str = "index") -> None:
        """Persist the graph database to filesystem"""
        os.makedirs(path, exist_ok=True)

        # 1. Save graph structure
        graph_path = os.path.join(path, f"{name}_graph.pkl")
        self.graph.write_pickle(graph_path)
        logger.info(f"Saved graph to {graph_path}")

        # 2. Save node mappings
        mappings_path = os.path.join(path, f"{name}_mappings.pkl")
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'node_to_idx': self.node_to_idx,
                'idx_to_node': self.idx_to_node,
                'node_to_node_stats': dict(self.node_to_node_stats)
            }, f)
        logger.info(f"Saved mappings to {mappings_path}")

        # 3. ✅ Save chunk embeddings (use pickle, not FAISS)
        chunk_embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
        with open(chunk_embeddings_path, 'wb') as f:
            pickle.dump(self.chunk_embeddings, f)
        logger.info(f"Saved chunk embeddings to {chunk_embeddings_path}")

        # 4. Save chunk embeddings array (pre-computed numpy array)
        if self._chunk_embeddings_array is not None:
            chunk_array_path = os.path.join(path, f"{name}_chunk_embeddings_array.npy")
            np.save(chunk_array_path, self._chunk_embeddings_array)

            chunk_ids_path = os.path.join(path, f"{name}_chunk_ids_list.pkl")
            with open(chunk_ids_path, 'wb') as f:
                pickle.dump(self._chunk_ids_list, f)
            logger.info(f"Saved chunk embeddings array to {chunk_array_path}")

        # 5. ✅ SQLite auto-saves (every operation commits)
        # 6. ✅ FAISS auto-saves (in batch_generate_embeddings)

        logger.info(f"Index saved to {path}")
        logger.info(f"  Graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        logger.info(f"  Facts (FAISS Flat): {self.fact_faiss_db.index.ntotal if self.fact_faiss_db.index else 0}")
        logger.info(f"  Entities (FAISS HNSW): {self.entity_faiss_db.index.ntotal if self.entity_faiss_db.index else 0}")
        logger.info(f"  Chunks (numpy array): {len(self.chunk_embeddings)}")

    def load_index(self, path: str, name: str = "index") -> None:
        """Load persisted graph database from filesystem"""
        logger.info(f"Loading index from {path}...")

        # 1. Load graph structure
        graph_path = os.path.join(path, f"{name}_graph.pkl")
        if os.path.exists(graph_path):
            self.graph = ig.Graph.Read_Pickle(graph_path)
            logger.info(f"Loaded graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")

        # 2. Load node mappings
        mappings_path = os.path.join(path, f"{name}_mappings.pkl")
        if os.path.exists(mappings_path):
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.node_to_idx = mappings['node_to_idx']
                self.idx_to_node = mappings['idx_to_node']
                self.node_to_node_stats = defaultdict(float, mappings['node_to_node_stats'])
            logger.info(f"Loaded mappings")

        # 3. ✅ Load chunk embeddings (from pickle, not FAISS)
        chunk_embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
        if os.path.exists(chunk_embeddings_path):
            with open(chunk_embeddings_path, 'rb') as f:
                self.chunk_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self.chunk_embeddings)} chunk embeddings")

            # Try to load pre-computed array first
            chunk_array_path = os.path.join(path, f"{name}_chunk_embeddings_array.npy")
            chunk_ids_path = os.path.join(path, f"{name}_chunk_ids_list.pkl")

            if os.path.exists(chunk_array_path) and os.path.exists(chunk_ids_path):
                self._chunk_embeddings_array = np.load(chunk_array_path)
                with open(chunk_ids_path, 'rb') as f:
                    self._chunk_ids_list = pickle.load(f)
                logger.info(f"Loaded pre-computed chunk embeddings array: {len(self._chunk_ids_list)} chunks")
            else:
                # Rebuild array if not saved
                logger.info("Pre-computed array not found, rebuilding...")
                self._rebuild_chunk_embeddings_array()

        # 4. ✅ Reconnect SQLite to the correct database path
        db_path = os.path.join(path, 'metadata.db')
        if os.path.exists(db_path):
            # Close existing connection if any
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()

            # Connect to the correct database
            self.db_path = db_path
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to SQLite database: {self.db_path}")
        else:
            logger.warning(f"SQLite database not found at {db_path}")

        # 5. ✅ FAISS auto-loads (in _init_faiss_indices)

        logger.info(f"Index loaded from {path}")

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Run a query on the graph database"""
        # This is a placeholder - implement based on your query needs
        logger.warning("query() method not implemented for graph store")
        return None

    def get_graph_db_info(self) -> Dict[str, Any]:
        """Return statistics or metadata about the graph database"""
        cursor = self.conn.cursor()

        # Count chunks
        cursor.execute('SELECT COUNT(*) FROM chunks')
        chunk_count = cursor.fetchone()[0]

        # Count entities
        cursor.execute('SELECT COUNT(*) FROM entities')
        entity_count = cursor.fetchone()[0]

        # Count facts
        cursor.execute('SELECT COUNT(*) FROM facts')
        fact_count = cursor.fetchone()[0]

        # Count relations
        cursor.execute('SELECT COUNT(*) FROM chunk_entity_relations')
        relation_count = cursor.fetchone()[0]

        return {
            'type': 'pruned_hipporag_igraph',
            'storage_path': self.storage_path,
            # Standard keys (compatible with other implementations)
            'num_nodes': self.graph.vcount(),
            'num_edges': self.graph.ecount(),
            'num_chunks': chunk_count,
            'num_entities': entity_count,
            'num_facts': fact_count,
            # Additional graph store specific info
            'chunk_entity_relations': relation_count,
            'fact_index_type': 'FAISS Flat',
            'fact_index_size': self.fact_faiss_db.index.ntotal if self.fact_faiss_db.index else 0,
            'entity_index_type': 'FAISS HNSW',
            'entity_index_size': self.entity_faiss_db.index.ntotal if self.entity_faiss_db.index else 0,
            'chunk_index_type': 'numpy array',
            'chunk_index_size': len(self.chunk_embeddings),
            'synonymy_edges_enabled': self.add_synonymy_edges
        }

