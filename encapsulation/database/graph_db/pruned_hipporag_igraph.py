import os
import json
import logging
import pickle
import sqlite3
import re
from typing import List, Dict, Any, Optional, Sequence, TYPE_CHECKING
from collections import defaultdict
import numpy as np
import igraph as ig
import faiss

from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig

logger = logging.getLogger(__name__)


@shared_module
class PrunedHippoRAGIGraphStore(GraphStore):
    """
    Pruned HippoRAG Graph Store using hybrid storage.

    This graph store implements a multi-component storage system:

    1. **Facts**: FAISS Flat index for exact similarity search
       - Stores fact embeddings for dense retrieval
       - Used to find relevant facts given a query

    2. **Entities**: FAISS HNSW index for approximate nearest neighbor search
       - Stores entity embeddings
       - Used to compute synonymy edges between similar entities

    3. **Chunks**: In-memory numpy array for brute-force search
       - Stores chunk embeddings
       - Used for dense passage retrieval fallback

    4. **Metadata**: SQLite database
       - Stores chunks, entities, facts, and their relationships
       - Provides efficient querying and filtering

    5. **Graph**: igraph undirected graph
       - Represents the knowledge graph structure
       - Nodes: chunks and entities
       - Edges: chunk-entity relations, entity-entity relations (facts), synonymy edges
       - Used for Personalized PageRank during retrieval
    """

    def __init__(self, config: "PrunedHippoRAGIGraphConfig"):
        """
        Initialize the Pruned HippoRAG Graph Store.

        Args:
            config: Configuration object containing all storage parameters
        """
        super().__init__(config)

        # Initialize embedding model
        self.embedding_model = config.embedding.build()

        # Initialize undirected graph
        self.graph = ig.Graph(directed=False)

        # Initialize FAISS indices for facts and entities
        self._init_faiss_indices()

        # Initialize SQLite database for metadata
        self._init_sqlite_db()

        # In-memory chunk embeddings (not stored in FAISS)
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None

        # Graph node mappings
        self.node_to_idx = {}  # node_id -> graph_index
        self.idx_to_node = {}  # graph_index -> node_id
        self.node_to_node_stats = defaultdict(float)  # (node_id, node_id) -> edge_weight

        # Storage configuration
        self.storage_path = getattr(config, 'storage_path', './data/graph_index')
        self.index_name = getattr(config, 'index_name', 'index')

        # Synonymy edge configuration
        self.add_synonymy_edges = getattr(config, 'add_synonymy_edges', False)
        self.synonymy_edge_topk = getattr(config, 'synonymy_edge_topk', 100)
        self.synonymy_edge_sim_threshold = getattr(config, 'synonymy_edge_sim_threshold', 0.8)

        logger.info("Pruned HippoRAG graph store initialized")
        logger.info(f"  - Fact index: FAISS Flat (exact search)")
        logger.info(f"  - Entity index: FAISS HNSW (synonymy edges)")
        logger.info(f"  - Chunk index: numpy array (brute-force search)")
        logger.info(f"  - Metadata: SQLite")
        logger.info(f"  - Graph: igraph")

    def _init_faiss_indices(self):
        """
        Initialize FAISS indices for facts and entities.

        - Fact index: FAISS Flat (exact search) for fact retrieval
        - Entity index: FAISS HNSW (approximate search) for synonymy edge computation
        """
        from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig

        storage_path = getattr(self.config, 'storage_path', './data/graph_index')
        os.makedirs(storage_path, exist_ok=True)

        # Initialize fact index (FAISS Flat for exact search)
        fact_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='flat',
            metric='cosine',
            normalize_L2=True,
            index_path=os.path.join(storage_path, 'fact_index'),
            index_name='index'
        )
        self.fact_faiss_db = fact_config.build()

        # Load existing fact index if available
        fact_index_path = os.path.join(storage_path, 'fact_index')
        if os.path.exists(fact_index_path):
            try:
                self.fact_faiss_db.load_index(fact_index_path)
                logger.info(f"Loaded existing fact index: {self.fact_faiss_db.index.ntotal} facts")
            except Exception as e:
                logger.warning(f"Failed to load fact index: {e}")

        # Initialize entity index (FAISS HNSW for approximate search)
        entity_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='hnsw',
            metric='cosine',
            normalize_L2=True,
            m=getattr(self.config, 'hnsw_M', 32),
            efConstruction=getattr(self.config, 'hnsw_ef_construction', 200),
            efSearch=getattr(self.config, 'hnsw_ef_search', 100),
            index_path=os.path.join(storage_path, 'entity_index'),
            index_name='index'
        )
        self.entity_faiss_db = entity_config.build()

        # Load existing entity index if available
        entity_index_path = os.path.join(storage_path, 'entity_index')
        if os.path.exists(entity_index_path):
            try:
                self.entity_faiss_db.load_index(entity_index_path)
                logger.info(f"Loaded existing entity index: {self.entity_faiss_db.index.ntotal} entities")
            except Exception as e:
                logger.warning(f"Failed to load entity index: {e}")

        logger.info("FAISS indices initialized (fact: Flat, entity: HNSW)")

    def _init_sqlite_db(self):
        """
        Initialize SQLite database for metadata storage.

        Creates tables for:
        - chunks: Document chunks with content and metadata
        - entities: Extracted entities with names and types
        - facts: Knowledge graph triples (head, relation, tail)
        - chunk_entity_relations: Links between chunks and entities
        - synonymy_edges: Similarity-based edges between entities
        """
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

        # Facts table (knowledge graph triples)
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

        # Chunk-entity relations table
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

        # Synonymy edges table (similarity-based entity connections)
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

    def _add_chunk_no_commit(self, chunk: Chunk):
        """
        Add a chunk to the database without committing (for batch operations).

        Args:
            chunk: Chunk object to add
        """
        chunk_id = chunk.id

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

        # Mark chunk embeddings array as dirty
        if chunk_id not in self.chunk_embeddings:
            self._chunk_embeddings_array = None

        # Add chunk node to graph if not exists
        if chunk_id not in self.node_to_idx:
            vertex_idx = self.graph.vcount()
            self.graph.add_vertex(name=chunk_id, node_type='chunk')
            self.node_to_idx[chunk_id] = vertex_idx
            self.idx_to_node[vertex_idx] = chunk_id

        logger.debug(f"Added chunk {chunk_id}")

    def add_chunk(self, chunk: Chunk):
        """
        Add a chunk to the database and commit.

        Args:
            chunk: Chunk object to add
        """
        self._add_chunk_no_commit(chunk)
        self.conn.commit()

    def _add_graph_data_no_commit(self, graph_data: GraphData, chunk_id: str) -> List[str]:
        """
        Add graph data (entities and facts) for a chunk without committing.

        This method:
        1. Processes and normalizes relation triples
        2. Extracts unique entities from triples with their types
        3. Adds entities to database and graph
        4. Creates chunk-entity relations
        5. Adds facts to database

        Args:
            graph_data: GraphData object containing entities and relations
            chunk_id: ID of the chunk this graph data belongs to

        Returns:
            List of newly created entity IDs
        """
        # Build entity name to type mapping from graph.entities
        # IMPORTANT: Use text_processing() on entity names to match processed triple entities
        entity_name_to_type = {}
        for entity_dict in graph_data.entities:
            entity_name = entity_dict.get('entity_name')
            entity_type = entity_dict.get('entity_type', 'Entity')
            if entity_name:
                # Process entity name to match the processed names in triples
                processed_name = text_processing(entity_name)
                if processed_name:
                    entity_name_to_type[processed_name] = entity_type

        # Process and normalize relation triples
        processed_triples = []
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head = text_processing(relation[0])
                rel_type = text_processing(relation[1])
                tail = text_processing(relation[2])

                if head and tail:  # Only keep triples with valid head and tail
                    processed_triples.append([head, rel_type, tail])

        # Extract unique entities from triples
        triple_entities = set()
        for triple in processed_triples:
            triple_entities.add(triple[0])  # head
            triple_entities.add(triple[2])  # tail

        new_entity_ids = []
        cursor = self.conn.cursor()

        # Add entities to database and graph
        for entity_name in triple_entities:
            entity_id = compute_mdhash_id(entity_name, prefix='entity-')
            # Get entity type from mapping, default to 'Entity'
            entity_type = entity_name_to_type.get(entity_name, 'Entity')

            # Add entity node to graph if not exists
            if entity_id not in self.node_to_idx:
                cursor.execute('''
                    INSERT OR IGNORE INTO entities (entity_id, entity_name, entity_type, attributes)
                    VALUES (?, ?, ?, ?)
                ''', (entity_id, entity_name, entity_type, '{}'))

                vertex_idx = self.graph.vcount()
                self.graph.add_vertex(name=entity_id, node_type='entity', entity_name=entity_name, entity_type=entity_type)
                self.node_to_idx[entity_id] = vertex_idx
                self.idx_to_node[vertex_idx] = entity_id

                new_entity_ids.append(entity_id)

            # Create chunk-entity relation
            cursor.execute('''
                INSERT OR IGNORE INTO chunk_entity_relations (chunk_id, entity_id, weight)
                VALUES (?, ?, ?)
            ''', (chunk_id, entity_id, 2.0))

        # Initialize fact cache if needed
        if not hasattr(self, '_fact_ids_cache'):
            self._fact_ids_cache = set()

        # Add facts to database
        for head_name, relation_type, tail_name in processed_triples:
            fact_text = str((head_name, relation_type, tail_name))
            fact_id = compute_mdhash_id(fact_text, prefix='fact-')

            if fact_id not in self._fact_ids_cache:
                cursor.execute('''
                    INSERT OR IGNORE INTO facts (fact_id, head, relation, tail, text)
                    VALUES (?, ?, ?, ?, ?)
                ''', (fact_id, head_name, relation_type, tail_name, fact_text))

                self._fact_ids_cache.add(fact_id)

        logger.debug(f"Added graph data for chunk {chunk_id}: {len(triple_entities)} entities, {len(processed_triples)} facts")

        return new_entity_ids

    def add_graph_data(self, graph_data: GraphData, chunk_id: str) -> List[str]:
        """
        Add graph data for a chunk and commit.

        Args:
            graph_data: GraphData object containing entities and relations
            chunk_id: ID of the chunk this graph data belongs to

        Returns:
            List of newly created entity IDs
        """
        result = self._add_graph_data_no_commit(graph_data, chunk_id)
        self.conn.commit()
        return result

    def _add_edge_to_graph(self, from_node: str, to_node: str, weight: float):
        """
        Incrementally add an edge to the graph.

        This method enables true incremental updates without rebuilding the entire graph.
        Edges are stored bidirectionally in node_to_node_stats for symmetric weight lookup,
        but only added once to the undirected igraph.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            weight: Edge weight
        """
        # Skip self-loops
        if from_node == to_node:
            return

        # Skip if nodes don't exist
        if from_node not in self.node_to_idx or to_node not in self.node_to_idx:
            return

        from_idx = self.node_to_idx[from_node]
        to_idx = self.node_to_idx[to_node]

        # Check if edge already exists in node_to_node_stats
        edge_key_1 = (from_node, to_node)
        edge_key_2 = (to_node, from_node)

        edge_exists = edge_key_1 in self.node_to_node_stats or edge_key_2 in self.node_to_node_stats

        # Update statistics (bidirectional for symmetric lookup)
        self.node_to_node_stats[edge_key_1] += weight
        self.node_to_node_stats[edge_key_2] += weight

        # Add edge to igraph if it doesn't exist yet
        # For undirected graph, we only need to add one edge
        if not edge_exists:
            try:
                self.graph.add_edge(from_idx, to_idx, weight=weight)
            except Exception as e:
                logger.warning(f"Failed to add edge {from_node} -> {to_node}: {e}")

    def batch_generate_embeddings(self):
        """
        Batch generate embeddings for all facts, entities, and chunks.

        This method processes embeddings in the following order:
        1. Chunk embeddings: Generated and stored in memory (not in FAISS)
        2. Entity embeddings: Generated and added to FAISS HNSW index
        3. Fact embeddings: Generated and added to FAISS Flat index

        Only new items (not already in FAISS/memory) are processed.
        """
        logger.info("Batch generating embeddings...")

        cursor = self.conn.cursor()

        cursor.execute('SELECT chunk_id, content FROM chunks')
        chunks_data = cursor.fetchall()

        new_chunks = []
        new_chunk_ids = []
        for chunk_id, content in chunks_data:
            if chunk_id not in self.chunk_embeddings:
                new_chunks.append(content)
                new_chunk_ids.append(chunk_id)

        if new_chunks:
            logger.info(f"Batch generating embeddings for {len(new_chunks)} chunks...")
            # Batch generate all chunk embeddings at once
            chunk_embeddings = self.embedding_model.embed(new_chunks)

            # Store embeddings
            for chunk_id, embedding in zip(new_chunk_ids, chunk_embeddings):
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                # Normalize for cosine similarity
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                self.chunk_embeddings[chunk_id] = embedding

            # Mark array needs rebuild
            self._chunk_embeddings_array = None
            logger.info(f"Chunk embeddings generated for {len(new_chunks)} chunks")

        # Generate entity embeddings and add to FAISS HNSW
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

        # Generate fact embeddings and add to FAISS Flat
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

        logger.info("Batch embedding generation completed!")

    def build_graph(self):
        """
        Build the knowledge graph from SQLite data.

        This method constructs an undirected igraph from the database:

        **Nodes**:
        - Chunk nodes: Document chunks
        - Entity nodes: Extracted entities

        **Edges** (stored bidirectionally for symmetric weight lookup):
        - Chunk-entity edges: Links between chunks and their entities (weight = 1.0 per edge)
        - Entity-entity edges (facts): Links between entities via facts (weight = 1.0 per fact, cumulative)
        - Synonymy edges: Similarity-based links between entities (weight = similarity score)

        The graph is rebuilt from scratch using batch edge addition for performance.
        """
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
        cursor.execute('SELECT entity_id, entity_name, entity_type FROM entities')
        entities = cursor.fetchall()
        for entity_id, entity_name, entity_type in entities:
            if entity_id not in self.node_to_idx:
                vertex_idx = self.graph.vcount()
                self.graph.add_vertex(name=entity_id, node_type='entity', entity_name=entity_name, entity_type=entity_type)
                self.node_to_idx[entity_id] = vertex_idx
                self.idx_to_node[vertex_idx] = entity_id

        # Collect all edges in node_to_node_stats
        # 1. Add chunk-entity edges (BIDIRECTIONAL: chunk <-> entity)
        cursor.execute('SELECT chunk_id, entity_id, weight FROM chunk_entity_relations')
        relations = cursor.fetchall()
        for chunk_id, entity_id, weight in relations:
            # Update statistics (BIDIRECTIONAL: both directions)
            # Using += to match HippoRAG's _add_edge_weight behavior
            self.node_to_node_stats[(chunk_id, entity_id)] += 1.0
            self.node_to_node_stats[(entity_id, chunk_id)] += 1.0

        # 2. Add entity-entity edges from facts (BIDIRECTIONAL: head <-> tail)
        # Build entity name to ID mapping for fast lookup
        cursor.execute('SELECT entity_id, entity_name FROM entities')
        entity_name_to_id = {name: eid for eid, name in cursor.fetchall()}

        cursor.execute('SELECT head, tail FROM facts')
        facts = cursor.fetchall()
        for head_name, tail_name in facts:
            head_id = entity_name_to_id.get(head_name)
            tail_id = entity_name_to_id.get(tail_name)
            if head_id and tail_id and head_id != tail_id:
                # Update statistics (BIDIRECTIONAL: both directions)
                # Multiple facts between same entities will accumulate weight (co-occurrence count)
                self.node_to_node_stats[(head_id, tail_id)] += 1.0
                self.node_to_node_stats[(tail_id, head_id)] += 1.0

        # 3. Add synonymy edges (UNIDIRECTIONAL - no need for bidirectional)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synonymy_edges'")
        if cursor.fetchone():
            cursor.execute('SELECT entity_id_1, entity_id_2, similarity FROM synonymy_edges')
            synonymy_edges = cursor.fetchall()
            if synonymy_edges:
                logger.info(f"Loading {len(synonymy_edges)} unidirectional synonymy edges from SQLite...")
                for entity_id_1, entity_id_2, similarity in synonymy_edges:
                    # Synonymy edges stored UNIDIRECTIONAL (igraph undirected handles traversal)
                    self.node_to_node_stats[(entity_id_1, entity_id_2)] = similarity

        # Build graph from node_to_node_stats
        valid_edges = []
        edge_weights = []

        for (from_node, to_node), weight in self.node_to_node_stats.items():
            if from_node in self.node_to_idx and to_node in self.node_to_idx:
                from_idx = self.node_to_idx[from_node]
                to_idx = self.node_to_idx[to_node]
                valid_edges.append((from_idx, to_idx))
                edge_weights.append(weight)

        # Add edges to graph (batch operation for performance)
        if valid_edges:
            logger.info(f"Adding {len(valid_edges)} edges to igraph (batch mode)...")
            self.graph.add_edges(valid_edges)
            self.graph.es['weight'] = edge_weights

        logger.info(f"Graph built: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")

    def _add_synonymy_edges(self):
        """
        Add synonymy edges between similar entities using FAISS HNSW.

        This method:
        1. Retrieves all entities from the database
        2. Filters out short entities (<=2 alphanumeric characters)
        3. Performs batch FAISS search to find top-k similar entities
        4. Filters results by similarity threshold
        5. Excludes entity pairs already connected by facts
        6. Stores synonymy edges in SQLite database

        Synonymy edges connect entities with similar embeddings, enabling
        the graph to capture semantic relationships beyond explicit facts.
        """
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

        if len(valid_entities) > 0:
            _, first_entity_name = valid_entities[0]
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
                similarity = -float(distance)

                # Check threshold
                if similarity < self.synonymy_edge_sim_threshold:
                    break  # Distances are sorted, can break early

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

            logger.info(f"Adding {len(edges_to_add)} synonymy edges to graph (batch mode)...")

            # Prepare batch data for igraph
            valid_edges = []
            edge_weights = []

            for entity_id_1, entity_id_2, similarity in edges_to_add:
                # Add edge to node_to_node_stats (unidirectional, as stored in SQLite)
                self.node_to_node_stats[(entity_id_1, entity_id_2)] = similarity

                # Prepare edge for batch addition to igraph
                if entity_id_1 in self.node_to_idx and entity_id_2 in self.node_to_idx:
                    idx_1 = self.node_to_idx[entity_id_1]
                    idx_2 = self.node_to_idx[entity_id_2]
                    valid_edges.append((idx_1, idx_2))
                    edge_weights.append(similarity)

            # Batch add all edges at once (much faster than individual add_edge calls)
            if valid_edges:
                try:
                    self.graph.add_edges(valid_edges)
                    # Set weights for all edges
                    edge_ids = self.graph.get_eids(valid_edges)
                    self.graph.es[edge_ids]['weight'] = edge_weights
                    logger.info(f"Successfully added {len(valid_edges)} synonymy edges to graph")
                except Exception as e:
                    logger.warning(f"Failed to batch add synonymy edges: {e}")

            logger.info(f"Added {num_synonym_edges} unique synonymy edges ({len(edges_to_add)} directional edges)")
        else:
            logger.info("No synonymy edges to add")

    def _rebuild_chunk_embeddings_array(self):
        """
        Rebuild chunk embeddings array for dense passage retrieval.

        This method creates a numpy array of chunk embeddings ordered by chunk IDs,
        enabling efficient brute-force similarity search during retrieval.
        The array is cached and only rebuilt when marked as dirty.
        """
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
        """
        Build the complete graph index from a list of chunks.

        This method performs the following steps:
        1. Adds all chunks to SQLite database
        2. Extracts and adds graph data (entities and facts) to SQLite
        3. Generates embeddings for facts, entities, and chunks
        4. Builds the igraph structure from SQLite data
        5. Optionally computes synonymy edges
        6. Rebuilds chunk embeddings array for dense retrieval

        Args:
            chunks: List of Chunk objects to index
        """
        logger.info(f"Building index from {len(chunks)} chunks...")

        batch_size = 1000
        total_chunks = len(chunks)

        from tqdm import tqdm

        logger.info("Step 1: Adding chunks and graph data to SQLite...")
        for i in tqdm(range(0, total_chunks, batch_size), desc="Processing chunks"):
            batch_end = min(i + batch_size, total_chunks)
            batch = chunks[i:batch_end]

            # Process batch (SQLite only, no igraph operations)
            for chunk in batch:
                self._add_chunk_no_commit(chunk)
                if chunk.graph and not chunk.graph.is_empty():
                    self._add_graph_data_no_commit(chunk.graph, chunk.id)

            # Commit once per batch
            self.conn.commit()

        logger.info(f"All {total_chunks} chunks added to SQLite")

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
        """
        Update the graph index with new or modified chunks.

        This method performs incremental updates:
        1. Adds new chunks and graph data to SQLite
        2. Generates embeddings for new items
        3. Rebuilds the graph structure from SQLite
        4. Optionally recomputes synonymy edges
        5. Rebuilds chunk embeddings array

        Args:
            chunks: List of Chunk objects to add/update

        Returns:
            True if successful, False otherwise
        """
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
        """
        Delete chunks and their associated graph data by IDs.

        This method:
        1. Deletes chunks from SQLite (cascades to relations)
        2. Rebuilds the graph structure from remaining data
        3. Rebuilds chunk embeddings array

        Args:
            ids: List of chunk IDs to delete

        Returns:
            True if successful, False otherwise
        """
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

        # 5. Delete from chunk_embeddings (not FAISS)
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
        """
        Retrieve chunks and their associated graph data by IDs.

        Args:
            ids: Sequence of chunk IDs to retrieve

        Returns:
            List of Chunk objects with graph data in metadata
        """
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
        """
        Get graph data (entities and relations) for a specific chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            GraphData object containing entities and relations for the chunk
        """
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
        """
        Persist the graph database to filesystem.

        This method saves:
        1. Graph structure (igraph object) as pickle
        2. Node mappings (node_to_idx, idx_to_node) as pickle
        3. Chunk embeddings as pickle
        4. FAISS indices for facts and entities
        5. SQLite database (already persisted)

        Args:
            path: Directory path to save the index
            name: Base name for index files
        """
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

        # 3. Save chunk embeddings
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

        logger.info(f"Index saved to {path}")
        logger.info(f"  Graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges")
        logger.info(f"  Facts (FAISS Flat): {self.fact_faiss_db.index.ntotal if self.fact_faiss_db.index else 0}")
        logger.info(f"  Entities (FAISS HNSW): {self.entity_faiss_db.index.ntotal if self.entity_faiss_db.index else 0}")
        logger.info(f"  Chunks (numpy array): {len(self.chunk_embeddings)}")

    def load_index(self, path: str, name: str = "index") -> None:
        """
        Load persisted graph database from filesystem.

        This method loads:
        1. Graph structure (igraph object) from pickle
        2. Node mappings (node_to_idx, idx_to_node) from pickle
        3. Chunk embeddings from pickle
        4. FAISS indices for facts and entities
        5. SQLite database (already connected)

        Args:
            path: Directory path containing the index
            name: Base name for index files
        """
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

        # 3. Load chunk embeddings
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

        # 4. Reconnect SQLite to the correct database path
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

        # 5. FAISS auto-loads (in _init_faiss_indices)

        logger.info(f"Index loaded from {path}")

    def query(self, _query: str, _params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Query method (not implemented for graph store).

        Graph stores use specialized retrieval methods instead of generic queries.

        Args:
            _query: Query string (unused)
            _params: Query parameters (unused)

        Returns:
            None
        """
        logger.warning("query() method not implemented for graph store")
        return None

    def get_graph_db_info(self) -> Dict[str, Any]:
        """
        Return statistics and metadata about the graph database.

        Returns:
            Dictionary containing counts of chunks, entities, facts, relations,
            graph nodes/edges, and FAISS index sizes
        """
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
            'num_nodes': self.graph.vcount(),
            'num_edges': self.graph.ecount(),
            'num_chunks': chunk_count,
            'num_entities': entity_count,
            'num_facts': fact_count,
            'chunk_entity_relations': relation_count,
            'fact_index_type': 'FAISS Flat',
            'fact_index_size': self.fact_faiss_db.index.ntotal if self.fact_faiss_db.index else 0,
            'entity_index_type': 'FAISS HNSW',
            'entity_index_size': self.entity_faiss_db.index.ntotal if self.entity_faiss_db.index else 0,
            'chunk_index_type': 'numpy array',
            'chunk_index_size': len(self.chunk_embeddings),
            'synonymy_edges_enabled': self.add_synonymy_edges
        }

