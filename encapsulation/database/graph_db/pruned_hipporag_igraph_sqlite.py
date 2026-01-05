import os
import json
import logging
import sqlite3
from typing import List, Dict, Any, Optional

from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from encapsulation.database.utils.sqlite_threadlocal import ThreadLocalSQLiteConnection

logger = logging.getLogger(__name__)


class _PrunedHippoRAGIGraphSQLiteMixin:
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
        # NOTE(thread-safety):
        # - DeepSearch may execute concurrent retrievals in a threadpool.
        # - Use one SQLite connection per thread to avoid cross-thread usage hazards.
        # - Guard in-memory graph state via the store RWLock (read_lock/write_lock).
        self.conn = ThreadLocalSQLiteConnection(
            self.db_path,
            timeout=30.0,
            pragmas={
                "journal_mode": "WAL",
                "synchronous": "NORMAL",
                "busy_timeout": 5000,
            },
        )
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
                owner_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity_name)')
        try:
            cursor.execute('ALTER TABLE entities ADD COLUMN owner_id TEXT')
        except sqlite3.OperationalError:
            pass

        # Facts table (knowledge graph triples)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                head TEXT NOT NULL,
                relation TEXT NOT NULL,
                tail TEXT NOT NULL,
                text TEXT NOT NULL,
                owner_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_head ON facts(head)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_tail ON facts(tail)')
        try:
            cursor.execute('ALTER TABLE facts ADD COLUMN owner_id TEXT')
        except sqlite3.OperationalError:
            pass

        # Chunk-entity relations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_entity_relations (
                chunk_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                owner_id TEXT,
                PRIMARY KEY (chunk_id, entity_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ce_entity ON chunk_entity_relations(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ce_chunk ON chunk_entity_relations(chunk_id)')
        try:
            cursor.execute('ALTER TABLE chunk_entity_relations ADD COLUMN owner_id TEXT')
        except sqlite3.OperationalError:
            pass

        # Synonymy edges table (similarity-based entity connections)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonymy_edges (
                entity_id_1 TEXT NOT NULL,
                entity_id_2 TEXT NOT NULL,
                similarity REAL NOT NULL,
                owner_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_id_1, entity_id_2),
                FOREIGN KEY (entity_id_1) REFERENCES entities(entity_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id_2) REFERENCES entities(entity_id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_syn_entity1 ON synonymy_edges(entity_id_1)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_syn_entity2 ON synonymy_edges(entity_id_2)')
        try:
            cursor.execute('ALTER TABLE synonymy_edges ADD COLUMN owner_id TEXT')
        except sqlite3.OperationalError:
            pass

        self.conn.commit()
        logger.info(f"SQLite database initialized at {self.db_path}")

    def _add_chunk_no_commit(self, chunk: Chunk):
        """
        Add a chunk to the database without committing (for batch operations).

        Args:
            chunk: Chunk object to add
        """
        chunk_id = chunk.id

        metadata = dict(chunk.metadata) if chunk.metadata else {}
        owner_value = chunk.owner_id or metadata.get('owner_id')
        owner_str = self._normalize_owner_id(owner_value)
        if owner_str:
            metadata['owner_id'] = owner_str

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO chunks (chunk_id, content, owner_id, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            chunk_id,
            chunk.content,
            owner_str,
            json.dumps(metadata) if metadata else '{}'
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

    def _get_chunk_owner_id(self, chunk_id: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT owner_id FROM chunks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        return self._normalize_owner_id(row[0]) if row and row[0] else None

    def _add_graph_data_no_commit(self, graph_data: GraphData, chunk_id: str, owner_id: Optional[Any] = None) -> List[str]:
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
        owner_str = self._normalize_owner_id(owner_id)
        if owner_str is None:
            owner_str = self._get_chunk_owner_id(chunk_id)

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
            entity_id = compute_mdhash_id(entity_name, prefix='entity-', owner_id=owner_str)
            # Get entity type from mapping, default to 'Entity'
            entity_type = entity_name_to_type.get(entity_name, 'Entity')

            # Add entity node to graph if not exists
            if entity_id not in self.node_to_idx:
                cursor.execute('''
                    INSERT OR IGNORE INTO entities (entity_id, entity_name, entity_type, attributes, owner_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (entity_id, entity_name, entity_type, '{}', owner_str))

                vertex_idx = self.graph.vcount()
                self.graph.add_vertex(name=entity_id, node_type='entity', entity_name=entity_name, entity_type=entity_type)
                self.node_to_idx[entity_id] = vertex_idx
                self.idx_to_node[vertex_idx] = entity_id

                new_entity_ids.append(entity_id)

            # Create chunk-entity relation
            cursor.execute('''
                INSERT OR IGNORE INTO chunk_entity_relations (chunk_id, entity_id, weight, owner_id)
                VALUES (?, ?, ?, ?)
            ''', (chunk_id, entity_id, 2.0, owner_str))

        # Initialize fact cache if needed
        if not hasattr(self, '_fact_ids_cache'):
            # A simple in-memory dedup cache for fact IDs.
            # This is process-local and not synchronized; safe for the intended single-thread usage.
            self._fact_ids_cache = set()

        # Add facts to database
        for head_name, relation_type, tail_name in processed_triples:
            fact_text = str((head_name, relation_type, tail_name))
            fact_id = compute_mdhash_id(fact_text, prefix='fact-', owner_id=owner_str)

            if fact_id not in self._fact_ids_cache:
                cursor.execute('''
                    INSERT OR IGNORE INTO facts (fact_id, head, relation, tail, text, owner_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (fact_id, head_name, relation_type, tail_name, fact_text, owner_str))

                self._fact_ids_cache.add(fact_id)

        logger.debug(f"Added graph data for chunk {chunk_id}: {len(triple_entities)} entities, {len(processed_triples)} facts")

        return new_entity_ids

    def add_graph_data(self, graph_data: GraphData, chunk_id: str, owner_id: Optional[Any] = None) -> List[str]:
        """
        Add graph data for a chunk and commit.

        Args:
            graph_data: GraphData object containing entities and relations
            chunk_id: ID of the chunk this graph data belongs to

        Returns:
            List of newly created entity IDs
        """
        owner_str = self._normalize_owner_id(owner_id) or self._get_chunk_owner_id(chunk_id)
        result = self._add_graph_data_no_commit(graph_data, chunk_id, owner_id=owner_str)
        self.conn.commit()
        return result


