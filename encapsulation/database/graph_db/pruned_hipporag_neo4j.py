import os
import json
import logging
import pickle
import re
from typing import List, Dict, Any, Optional, Sequence, Set, Tuple, TYPE_CHECKING
import numpy as np
import igraph as ig
import faiss
import neo4j

from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from config.encapsulation.database.graph_db.pruned_hipporag_neo4j_config import PrunedHippoRAGNeo4jConfig

logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


@shared_module
class PrunedHippoRAGNeo4jStore(GraphStore):
    """
    Pruned HippoRAG Graph Store using Neo4j backend.

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

    4. **Metadata & Graph**: Neo4j database
       - Stores chunks, entities, facts, and their relationships
       - Provides native graph queries and traversal
       - Nodes: Chunk, Entity
       - Relationships: MENTIONS (chunk->entity), RELATES_TO (entity->entity), SIMILAR_TO (entity->entity)

    5. **PageRank**: Extracted subgraph to igraph for computation
       - Subgraph extracted from Neo4j based on query
       - PageRank computed in-memory using igraph
    """

    def __init__(self, config: "PrunedHippoRAGNeo4jConfig"):
        """
        Initialize the Pruned HippoRAG Graph Store with Neo4j backend.

        Args:
            config: Configuration object containing all storage parameters
        """
        super().__init__(config)

        # Initialize embedding model
        self.embedding_model = config.embedding.build()

        # Initialize Neo4j connection with notifications disabled
        self._driver = None
        try:
            self._driver: neo4j.Driver = neo4j.GraphDatabase.driver(
                config.url,
                auth=(config.username, config.password),
                notifications_min_severity="OFF"  # Disable all notifications
            )
            logger.info(f"Successfully connected to Neo4j database: {config.url}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
            raise

        self.database = config.database

        # Add conn property for compatibility with base class
        # (Neo4j doesn't use SQLite conn, but base class expects it)
        self.conn = None

        # Initialize Neo4j schema (constraints and indices)
        self._init_neo4j_schema()

        # Initialize FAISS indices for facts and entities
        self._init_faiss_indices()

        # In-memory chunk embeddings (not stored in FAISS)
        self.chunk_embeddings = {}
        self._chunk_embeddings_array = None
        self._chunk_ids_list = None

        # Chunk embeddings optimization settings
        self.use_float16_embeddings = getattr(config, 'use_float16_embeddings', True)
        self.normalize_chunk_embeddings = getattr(config, 'normalize_chunk_embeddings', True)

        # In-memory graph cache for fast neighbor lookups
        self._graph_cache: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self._cache_loaded = False

        # Entity chunk count cache (computed from graph cache)
        self._entity_chunk_count_cache: Optional[Dict[str, int]] = None

        # Storage configuration
        self.storage_path = getattr(config, 'storage_path', './data/graph_index_neo4j')
        self.index_name = getattr(config, 'index_name', 'index')

        # Synonymy edge configuration
        self.add_synonymy_edges = getattr(config, 'add_synonymy_edges', False)
        self.synonymy_edge_topk = getattr(config, 'synonymy_edge_topk', 100)
        self.synonymy_edge_sim_threshold = getattr(config, 'synonymy_edge_sim_threshold', 0.8)

        # Load graph structure into memory for fast lookups
        self._load_graph_cache()

        # Load chunk embeddings from disk if available
        self._load_chunk_embeddings()

        logger.info("Pruned HippoRAG Neo4j graph store initialized")
        logger.info(f"  - Fact index: FAISS Flat (exact search)")
        logger.info(f"  - Entity index: FAISS HNSW (synonymy edges)")
        logger.info(f"  - Chunk index: numpy array (brute-force search)")
        logger.info(f"  - Metadata & Graph: Neo4j (cached in memory)")
        logger.info(f"  - PageRank: igraph (extracted subgraph)")

    def _init_neo4j_schema(self):
        """
        Initialize Neo4j schema with constraints and indices.

        Creates:
        - Unique constraints on Chunk.chunk_id and Entity.entity_id
        - Indices on frequently queried properties
        - Note: Facts are now stored as relationships, not nodes
        """
        with self._driver.session(database=self.database) as session:
            # Chunk constraints and indices
            try:
                session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                session.run("CREATE INDEX chunk_owner IF NOT EXISTS FOR (c:Chunk) ON (c.owner_id)")
                logger.info("Created Chunk constraints and indices")
            except Exception as e:
                logger.warning(f"Failed to create Chunk constraints: {e}")

            # Entity constraints and indices
            try:
                session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)")
                logger.info("Created Entity constraints and indices")
            except Exception as e:
                logger.warning(f"Failed to create Entity constraints: {e}")

            # Note: Facts are now stored as :Fact relationships between entities
            # No separate Fact nodes needed
            logger.info("Schema initialized (Facts stored as relationships)")

    def _init_faiss_indices(self):
        """
        Initialize FAISS indices for facts and entities.

        - Fact index: FAISS Flat (exact search) for fact retrieval
        - Entity index: FAISS HNSW (approximate search) for synonymy edge computation
        """
        from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig

        storage_path = getattr(self.config, 'storage_path', './data/graph_index_neo4j')
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

    def _load_chunk_embeddings(self):
        """
        Load chunk embeddings from disk if available.

        This is called during initialization to restore chunk embeddings
        that were saved during previous sessions.
        """
        embeddings_path = os.path.join(self.storage_path, f"{self.index_name}_chunk_embeddings.pkl")
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    self.chunk_embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.chunk_embeddings)} chunk embeddings from {embeddings_path}")
                # Mark array for rebuild on first use
                self._chunk_embeddings_array = None
            except Exception as e:
                logger.warning(f"Failed to load chunk embeddings: {e}")
        else:
            logger.info(f"No existing chunk embeddings found at {embeddings_path}")

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on Neo4j.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]



    def _batch_add_chunks_and_graph_data(self, chunks: List[Chunk]) -> List[str]:
        """
        Batch add chunks and their graph data to Neo4j (OPTIMIZED).

        This method collects all data from chunks and performs batch insertions
        using UNWIND, which is much faster than individual queries.

        Args:
            chunks: List of Chunk objects to add

        Returns:
            List of newly created entity IDs
        """
        import time
        start_time = time.time()

        # Collect all data
        chunk_data = []
        entity_data = {}  # entity_id -> (entity_name, entity_type) (deduplicated)
        mention_data = []
        fact_data = []
        new_entity_ids = []

        for chunk in chunks:
            # Prepare chunk data
            # Extract owner_id from metadata if available
            owner_id = chunk.metadata.get('owner_id') if chunk.metadata else None
            chunk_data.append({
                'chunk_id': chunk.id,
                'content': chunk.content,
                'metadata': json.dumps(chunk.metadata) if chunk.metadata else '{}',
                'owner_id': owner_id
            })

            # Process graph data
            if chunk.graph and not chunk.graph.is_empty():
                # Build entity name to type mapping from graph.entities
                # IMPORTANT: Use text_processing() on entity names to match processed triple entities
                entity_name_to_type = {}
                for entity_dict in chunk.graph.entities:
                    entity_name = entity_dict.get('entity_name')
                    entity_type = entity_dict.get('entity_type', 'Entity')
                    if entity_name:
                        # Process entity name to match the processed names in triples
                        processed_name = text_processing(entity_name)
                        if processed_name:
                            entity_name_to_type[processed_name] = entity_type

                # Process and normalize relation triples
                processed_triples = []
                for relation in chunk.graph.relations:
                    if len(relation) >= 3:
                        head = text_processing(relation[0])
                        rel_type = text_processing(relation[1])
                        tail = text_processing(relation[2])

                        if head and tail:
                            processed_triples.append([head, rel_type, tail])

                # Extract unique entities from triples
                triple_entities = set()
                for triple in processed_triples:
                    triple_entities.add(triple[0])  # head
                    triple_entities.add(triple[2])  # tail

                # Collect entity data (deduplicated across all chunks)
                for entity_name in triple_entities:
                    entity_id = compute_mdhash_id(entity_name, prefix='entity-')
                    if entity_id not in entity_data:
                        # Get entity type from mapping, default to 'Entity'
                        entity_type = entity_name_to_type.get(entity_name, 'Entity')
                        entity_data[entity_id] = (entity_name, entity_type)

                    # Collect mention data
                    mention_data.append({
                        'chunk_id': chunk.id,
                        'entity_id': entity_id
                    })

                # Collect fact data
                for head_name, relation_type, tail_name in processed_triples:
                    fact_text = str((head_name, relation_type, tail_name))
                    fact_id = compute_mdhash_id(fact_text, prefix='fact-')
                    head_id = compute_mdhash_id(head_name, prefix='entity-')
                    tail_id = compute_mdhash_id(tail_name, prefix='entity-')

                    fact_data.append({
                        'fact_id': fact_id,
                        'head_id': head_id,
                        'tail_id': tail_id,
                        'head_name': head_name,
                        'relation_type': relation_type,
                        'tail_name': tail_name,
                        'fact_text': fact_text
                    })

        # Prepare entity list for batch insertion
        entity_list = [
            {'entity_id': eid, 'entity_name': name, 'entity_type': etype}
            for eid, (name, etype) in entity_data.items()
        ]

        logger.info(f"Batch data prepared: {len(chunk_data)} chunks, {len(entity_list)} entities, "
                   f"{len(mention_data)} mentions, {len(fact_data)} facts")

        # Batch insert using single transaction
        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                # 1. Batch insert chunks
                if chunk_data:
                    chunk_query = """
                    UNWIND $chunks AS chunk
                    MERGE (c:Chunk {chunk_id: chunk.chunk_id})
                    SET c.content = chunk.content,
                        c.metadata = chunk.metadata,
                        c.owner_id = chunk.owner_id,
                        c.updated_at = datetime(),
                        c.created_at = COALESCE(c.created_at, datetime())
                    """
                    tx.run(chunk_query, {'chunks': chunk_data})
                    logger.info(f"  Batch inserted {len(chunk_data)} chunks")

                # 2. Batch insert entities and track new ones
                if entity_list:
                    entity_query = """
                    UNWIND $entities AS entity
                    MERGE (e:Entity {entity_id: entity.entity_id})
                    ON CREATE SET e.entity_name = entity.entity_name,
                                  e.entity_text = entity.entity_name,
                                  e.entity_type = entity.entity_type,
                                  e.node_type = 'entity',
                                  e.attributes = '{}',
                                  e.created_at = datetime(),
                                  e.updated_at = datetime(),
                                  e.is_new = true
                    ON MATCH SET e.entity_name = entity.entity_name,
                                 e.entity_text = entity.entity_name,
                                 e.entity_type = entity.entity_type,
                                 e.updated_at = datetime(),
                                 e.is_new = false
                    RETURN e.entity_id AS entity_id, e.is_new AS is_new
                    """
                    result = tx.run(entity_query, {'entities': entity_list})
                    for record in result:
                        if record['is_new']:
                            new_entity_ids.append(record['entity_id'])
                    logger.info(f"  Batch inserted {len(entity_list)} entities ({len(new_entity_ids)} new)")

                # 3. Batch create chunk-entity relationships
                if mention_data:
                    mention_query = """
                    UNWIND $mentions AS m
                    MATCH (c:Chunk {chunk_id: m.chunk_id})
                    MATCH (e:Entity {entity_id: m.entity_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.weight = COALESCE(r.weight, 0.0) + 1.0,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(mention_query, {'mentions': mention_data})
                    logger.info(f"  Batch created {len(mention_data)} MENTIONS relationships")

                # 4. Batch create fact relationships
                if fact_data:
                    fact_query = """
                    UNWIND $facts AS f
                    MATCH (e1:Entity {entity_id: f.head_id})
                    MATCH (e2:Entity {entity_id: f.tail_id})
                    MERGE (e1)-[r:RELATES_TO {fact_id: f.fact_id}]->(e2)
                    SET r.head = f.head_name,
                        r.predicate = f.relation_type,
                        r.tail = f.tail_name,
                        r.text = f.fact_text,
                        r.weight = COALESCE(r.weight, 0.0) + 1.0,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(fact_query, {'facts': fact_data})
                    logger.info(f"  Batch created {len(fact_data)} RELATES_TO relationships")

                tx.commit()

        elapsed = time.time() - start_time
        logger.info(f"Batch insertion completed in {elapsed:.2f}s")

        return new_entity_ids

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

        # 1. Generate chunk embeddings
        chunk_query = "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.content AS content"
        chunks_data = self._execute_query(chunk_query)

        new_chunks = []
        new_chunk_ids = []
        for record in chunks_data:
            chunk_id = record['chunk_id']
            content = record['content']
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

        # 2. Generate entity embeddings and add to FAISS HNSW
        entity_query = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_name AS entity_name"
        entities = self._execute_query(entity_query)

        new_entities = []
        for record in entities:
            entity_id = record['entity_id']
            entity_name = record['entity_name']
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

        # 3. Generate fact embeddings and add to FAISS Flat
        # Facts are stored as RELATES_TO relationships between entities
        fact_query = "MATCH ()-[r:RELATES_TO]->() RETURN r.fact_id AS fact_id, r.text AS text"
        facts = self._execute_query(fact_query)

        new_facts = []
        for record in facts:
            fact_id = record['fact_id']
            fact_text = record['text']
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

    def _add_synonymy_edges(self, new_entity_ids: Optional[List[str]] = None):
        """
        Add synonymy edges between similar entities using FAISS HNSW.

        This method:
        1. Retrieves entities from Neo4j (all or only new ones for incremental update)
        2. Filters out short entities (<=2 alphanumeric characters)
        3. Performs batch FAISS search to find top-k similar entities
        4. Filters results by similarity threshold
        5. Excludes entity pairs already connected by facts
        6. Stores synonymy edges in Neo4j as SIMILAR_TO relationships

        Args:
            new_entity_ids: Optional list of new entity IDs for incremental update.
                          If None, processes all entities (full rebuild).
        """
        if not self.add_synonymy_edges:
            logger.info("Synonymy edges disabled")
            return

        from tqdm import tqdm

        # Determine if this is incremental or full rebuild
        if new_entity_ids:
            logger.info(f"Computing synonymy edges for {len(new_entity_ids)} new entities (incremental)...")
            # Get only new entities
            entity_query = """
            MATCH (e:Entity)
            WHERE e.entity_id IN $entity_ids
            RETURN e.entity_id AS entity_id, e.entity_name AS entity_name
            """
            entities = self._execute_query(entity_query, {'entity_ids': new_entity_ids})
        else:
            logger.info("Computing synonymy edges for all entities (full rebuild)...")
            # Get all entities
            entity_query = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_name AS entity_name"
            entities = self._execute_query(entity_query)

        if not entities:
            logger.warning("No entities found, skipping synonymy edge addition")
            return

        # Build entity ID to name mapping for fast lookup
        entity_id_to_name = {record['entity_id']: record['entity_name'] for record in entities}

        # Build a set to track existing entity-entity edges (fact edges only)
        existing_entity_entity_edges = set()

        # Get all RELATES_TO relationships
        relation_query = """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        RETURN e1.entity_id AS head_id, e2.entity_id AS tail_id
        """
        relations = self._execute_query(relation_query)

        for record in relations:
            head_id = record['head_id']
            tail_id = record['tail_id']
            existing_entity_entity_edges.add((head_id, tail_id))
            existing_entity_entity_edges.add((tail_id, head_id))

        logger.info(f"Built existing entity-entity edge set with {len(existing_entity_entity_edges)} directional edges")

        num_synonym_edges = 0
        edges_to_add = []  # Batch collect edges for Neo4j

        # Pre-extract and normalize all embeddings for batch search
        logger.info("Preparing embeddings for batch FAISS search...")
        valid_entities = []
        embeddings_list = []

        for record in entities:
            entity_id = record['entity_id']
            entity_name = record['entity_name']

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

                # Get neighbor name for validation (from cache, not Neo4j)
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

        # Batch insert all edges to Neo4j
        if edges_to_add:
            logger.info(f"Saving {len(edges_to_add)} directional synonymy edges to Neo4j...")

            # Use UNWIND for batch insertion
            batch_query = """
            UNWIND $edges AS edge
            MATCH (e1:Entity {entity_id: edge.entity_id_1})
            MATCH (e2:Entity {entity_id: edge.entity_id_2})
            MERGE (e1)-[r:SIMILAR_TO]-(e2)
            SET r.similarity = edge.similarity,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """

            # Prepare batch data
            batch_data = [
                {
                    'entity_id_1': e1,
                    'entity_id_2': e2,
                    'similarity': sim
                }
                for e1, e2, sim in edges_to_add
            ]

            self._execute_query(batch_query, {'edges': batch_data})

            logger.info(f"Added {num_synonym_edges} unique synonymy edges ({len(edges_to_add)} directional edges)")
        else:
            logger.info("No synonymy edges to add")

    def get_neighbors_with_weights(self, node_id: str) -> List[Tuple[str, float]]:
        """
        Get all neighbors of a node with their edge weights from Neo4j.

        Args:
            node_id: Node ID (chunk_id or entity_id)

        Returns:
            List of (neighbor_id, weight) tuples
        """
        # Optimized query: use single MATCH with OR condition
        query = """
        MATCH (n)-[r]-(neighbor)
        WHERE n.chunk_id = $node_id OR n.entity_id = $node_id
        RETURN COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight
        """

        results = self._execute_query(query, {'node_id': node_id})

        neighbors = []
        for record in results:
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0
            if neighbor_id:
                neighbors.append((neighbor_id, float(weight)))

        return neighbors

    def _build_entity_chunk_count_cache(self):
        """
        Build entity chunk count cache from graph cache.

        This computes how many chunks each entity appears in, which is used
        for normalizing entity weights during PPR.
        """
        if not self._graph_cache:
            logger.warning("Graph cache not loaded, cannot build entity chunk count cache")
            return

        logger.info("Building entity chunk count cache from graph cache...")
        import time
        start_time = time.time()

        self._entity_chunk_count_cache = {}

        for entity_id, neighbors in self._graph_cache.items():
            # Only process entity nodes
            if not entity_id.startswith("entity-"):
                continue

            # Count unique chunk neighbors (chunks don't start with "entity-")
            chunk_count = sum(1 for neighbor_id, _ in neighbors if not neighbor_id.startswith("entity-"))
            self._entity_chunk_count_cache[entity_id] = chunk_count

        elapsed = time.time() - start_time
        logger.info(f"Entity chunk count cache built: {len(self._entity_chunk_count_cache)} entities in {elapsed:.2f}s")

    def get_entity_chunk_count_from_cache(self, entity_id: str) -> int:
        """
        Get the number of chunks an entity appears in from cache.

        Args:
            entity_id: Entity ID

        Returns:
            Number of chunks the entity appears in (0 if not found)
        """
        if self._entity_chunk_count_cache is None:
            logger.warning("Entity chunk count cache not built, returning 0")
            return 0

        return self._entity_chunk_count_cache.get(entity_id, 0)

    def get_batch_entity_chunk_counts_from_cache(self, entity_ids: List[str]) -> Dict[str, int]:
        """
        Get chunk counts for multiple entities from cache.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dictionary mapping entity IDs to chunk counts
        """
        if self._entity_chunk_count_cache is None:
            logger.warning("Entity chunk count cache not built, returning empty dict")
            return {}

        return {eid: self._entity_chunk_count_cache.get(eid, 0) for eid in entity_ids}

    def _load_graph_cache(self, force_reload: bool = False):
        """
        Load entire graph structure into memory for fast neighbor lookups.
        This trades memory for speed - loads all edges once at startup.

        Args:
            force_reload: If True, force reload even if cache is already loaded
        """
        if self._cache_loaded and not force_reload:
            return

        logger.info("Loading graph structure into memory cache...")
        import time
        start_time = time.time()

        # Query all edges in the graph
        query = """
        MATCH (n)-[r]-(neighbor)
        RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
               COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight
        """

        results = self._execute_query(query)

        # Build adjacency list
        self._graph_cache = {}
        edge_count = 0
        for record in results:
            node_id = record['node_id']
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0

            if node_id and neighbor_id:
                if node_id not in self._graph_cache:
                    self._graph_cache[node_id] = []
                self._graph_cache[node_id].append((neighbor_id, float(weight)))
                edge_count += 1

        self._cache_loaded = True
        elapsed = time.time() - start_time
        logger.info(f"Graph cache loaded: {len(self._graph_cache)} nodes, {edge_count} edges in {elapsed:.2f}s")

        # Build entity chunk count cache
        self._build_entity_chunk_count_cache()

    def _update_graph_cache_incremental(self, new_chunk_ids: List[str], new_entity_ids: List[str]):
        """
        Incrementally update graph cache with new edges from new chunks and entities.

        Args:
            new_chunk_ids: List of newly added chunk IDs
            new_entity_ids: List of newly added entity IDs
        """
        if not self._cache_loaded or self._graph_cache is None:
            # Cache not loaded yet, do full load
            self._load_graph_cache()
            return

        logger.info(f"Incrementally updating graph cache for {len(new_chunk_ids)} chunks and {len(new_entity_ids)} entities...")
        import time
        start_time = time.time()

        # Query edges involving new nodes
        all_new_node_ids = new_chunk_ids + new_entity_ids
        if not all_new_node_ids:
            return

        query = """
        MATCH (n)-[r]-(neighbor)
        WHERE n.chunk_id IN $node_ids OR n.entity_id IN $node_ids
        RETURN COALESCE(n.chunk_id, n.entity_id) AS node_id,
               COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight
        """
        results = self._execute_query(query, {'node_ids': all_new_node_ids})

        # Update adjacency list
        edge_count = 0
        for record in results:
            node_id = record['node_id']
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0

            if node_id and neighbor_id:
                # Add edge from node_id to neighbor_id
                if node_id not in self._graph_cache:
                    self._graph_cache[node_id] = []
                # Check if edge already exists (avoid duplicates)
                if not any(n == neighbor_id for n, _ in self._graph_cache[node_id]):
                    self._graph_cache[node_id].append((neighbor_id, float(weight)))
                    edge_count += 1

                # Add reverse edge (since graph is undirected)
                if neighbor_id not in self._graph_cache:
                    self._graph_cache[neighbor_id] = []
                if not any(n == node_id for n, _ in self._graph_cache[neighbor_id]):
                    self._graph_cache[neighbor_id].append((node_id, float(weight)))

        elapsed = time.time() - start_time
        logger.info(f"Graph cache updated: added {edge_count} new edges in {elapsed:.2f}s")

    def get_batch_neighbors_with_weights(self, node_ids: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get neighbors for multiple nodes in a single query (batch operation).
        Uses in-memory cache if available, otherwise queries Neo4j.

        Args:
            node_ids: List of node IDs

        Returns:
            Dictionary mapping node_id to list of (neighbor_id, weight) tuples
        """
        if not node_ids:
            return {}

        # Use cache if loaded
        if self._cache_loaded and self._graph_cache is not None:
            neighbors_map = {}
            for nid in node_ids:
                neighbors_map[nid] = self._graph_cache.get(nid, [])
            return neighbors_map

        # Fallback to Neo4j query
        query = """
        UNWIND $node_ids AS nid
        MATCH (n)-[r]-(neighbor)
        WHERE n.chunk_id = nid OR n.entity_id = nid
        RETURN nid AS node_id,
               COALESCE(neighbor.chunk_id, neighbor.entity_id) AS neighbor_id,
               COALESCE(r.weight, r.similarity, 1.0) AS weight
        """

        results = self._execute_query(query, {'node_ids': node_ids})

        # Group by node_id
        neighbors_map = {nid: [] for nid in node_ids}
        for record in results:
            node_id = record['node_id']
            neighbor_id = record['neighbor_id']
            weight = record['weight'] or 1.0
            if neighbor_id and node_id in neighbors_map:
                neighbors_map[node_id].append((neighbor_id, float(weight)))

        return neighbors_map

    def extract_subgraph_from_cache(self, subgraph_node_ids: Set[str]) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str]]:
        """
        Extract a subgraph from in-memory cache and convert to igraph for PageRank computation.

        This method is much faster than extract_subgraph_for_ppr as it avoids Neo4j queries.

        Args:
            subgraph_node_ids: Set of node IDs to include in the subgraph

        Returns:
            Tuple of (igraph, node_to_idx, idx_to_node)
        """
        if not subgraph_node_ids:
            logger.warning("Empty subgraph node set")
            return ig.Graph(directed=False), {}, {}

        # Require cache to be loaded
        if not (self._cache_loaded and self._graph_cache):
            logger.error("Graph cache not loaded, cannot extract subgraph")
            return ig.Graph(directed=False), {}, {}

        logger.info(f"Extracting subgraph with {len(subgraph_node_ids)} nodes from cache...")

        # Build node mappings
        node_to_idx = {node_id: i for i, node_id in enumerate(sorted(subgraph_node_ids))}
        idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}

        # Create igraph
        graph = ig.Graph(directed=False)
        graph.add_vertices(len(node_to_idx))

        # Extract edges from cache
        edge_list = []
        edge_weights = []

        for u in subgraph_node_ids:
            neighbors = self._graph_cache.get(u, [])
            for v, w in neighbors:
                if v in node_to_idx and node_to_idx[u] < node_to_idx[v]:
                    # Only add each edge once (undirected graph)
                    edge_list.append((node_to_idx[u], node_to_idx[v]))
                    edge_weights.append(float(w))

        if edge_list:
            graph.add_edges(edge_list)
            graph.es['weight'] = edge_weights

        logger.info(f"Extracted subgraph from cache: {graph.vcount()} nodes, {graph.ecount()} edges")

        return graph, node_to_idx, idx_to_node



    def compute_ppr_push(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        alpha: float = 0.5,
        epsilon: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank using push-based algorithm on cached graph.

        This method is faster than igraph-based PPR for small to medium subgraphs
        as it avoids the overhead of constructing igraph objects.

        Args:
            subgraph_nodes: Set of node IDs in the subgraph
            reset: Reset distribution (dict mapping node_id -> probability, should sum to 1.0)
            alpha: Damping factor (teleport probability)
            epsilon: Convergence threshold

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        if not (self._cache_loaded and self._graph_cache):
            logger.warning("Graph cache not loaded, cannot use push-based PPR")
            return {}

        from encapsulation.database.utils.ppr_push import extract_subgraph_adjacency, ppr_push

        # Extract subgraph adjacency from cache
        subgraph_adj = extract_subgraph_adjacency(self._graph_cache, subgraph_nodes)

        # Run push-based PPR
        ppr_scores = ppr_push(
            adjacency=subgraph_adj,
            reset=reset,
            alpha=alpha,
            epsilon=epsilon
        )

        return ppr_scores

    def _append_chunk_embeddings(self, new_chunk_ids: List[str]):
        """
        Incrementally append new chunk embeddings to the array (OPTIMIZED).

        This method only processes new chunks instead of rebuilding the entire array,
        which is much faster for incremental updates.

        Args:
            new_chunk_ids: List of new chunk IDs to append
        """
        if not new_chunk_ids:
            logger.info("No new chunks to append")
            return

        import time
        start_time = time.time()

        logger.info(f"Appending {len(new_chunk_ids)} new chunk embeddings...")

        new_embeddings_list = []
        new_chunk_ids_ordered = []

        for cid in sorted(new_chunk_ids):  # Sort for consistency
            if cid not in self.chunk_embeddings:
                logger.warning(f"Chunk {cid} not found in chunk_embeddings, skipping")
                continue

            emb = self.chunk_embeddings[cid]

            # Ensure it's a numpy array
            if isinstance(emb, list):
                emb = np.array(emb)
            elif not isinstance(emb, np.ndarray):
                logger.warning(f"Chunk {cid} has invalid embedding type: {type(emb)}, skipping")
                continue

            # Check shape consistency with existing array
            if self._chunk_embeddings_array is not None and len(self._chunk_embeddings_array) > 0:
                expected_shape = (self._chunk_embeddings_array.shape[1],)
                if emb.shape != expected_shape:
                    logger.warning(f"Chunk {cid} has shape {emb.shape}, expected {expected_shape}, skipping")
                    continue

            # Normalize if enabled (for cosine similarity)
            if self.normalize_chunk_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

            new_embeddings_list.append(emb)
            new_chunk_ids_ordered.append(cid)

        if new_embeddings_list:
            # Convert to array with optional float16
            if self.use_float16_embeddings:
                new_array = np.array(new_embeddings_list, dtype=np.float16)
            else:
                new_array = np.array(new_embeddings_list, dtype=np.float32)

            # Append to existing array or create new one
            if self._chunk_embeddings_array is not None and len(self._chunk_embeddings_array) > 0:
                self._chunk_embeddings_array = np.vstack([self._chunk_embeddings_array, new_array])
                self._chunk_ids_list.extend(new_chunk_ids_ordered)
            else:
                self._chunk_embeddings_array = new_array
                self._chunk_ids_list = new_chunk_ids_ordered

            elapsed = time.time() - start_time
            dtype_str = "float16" if self.use_float16_embeddings else "float32"
            logger.info(f"Appended {len(new_embeddings_list)} chunk embeddings ({dtype_str}) in {elapsed:.3f}s, "
                       f"total: {len(self._chunk_ids_list)} chunks, "
                       f"memory: {self._chunk_embeddings_array.nbytes / 1024 / 1024:.2f} MB")
        else:
            logger.warning("No valid new chunk embeddings to append")

    def _rebuild_chunk_embeddings_array(self):
        """
        Rebuild chunk embeddings array for dense passage retrieval.

        This method creates a numpy array of chunk embeddings ordered by chunk IDs,
        enabling efficient brute-force similarity search during retrieval.
        The array is cached and only rebuilt when marked as dirty.

        Optimizations:
        - Uses float16 to reduce memory usage (if enabled)
        - Normalizes embeddings for cosine similarity (if enabled)
        """
        if self._chunk_embeddings_array is not None:
            return  # Already built

        logger.info("Rebuilding chunk embeddings array...")

        self._chunk_ids_list = list(self.chunk_embeddings.keys())
        embeddings_list = []

        for i, cid in enumerate(self._chunk_ids_list):
            emb = self.chunk_embeddings[cid]
            # Ensure it's a numpy array
            if isinstance(emb, list):
                emb = np.array(emb)
            elif not isinstance(emb, np.ndarray):
                logger.warning(f"Chunk {cid} has invalid embedding type: {type(emb)}")
                continue

            # Check shape
            if len(embeddings_list) > 0 and emb.shape != embeddings_list[0].shape:
                logger.error(f"Chunk {cid} (index {i}) has shape {emb.shape}, expected {embeddings_list[0].shape}")
                logger.error(f"  First chunk ID: {self._chunk_ids_list[0]}, shape: {embeddings_list[0].shape}")
                logger.error(f"  Current chunk ID: {cid}, shape: {emb.shape}")
                continue

            # Normalize if enabled (for cosine similarity)
            if self.normalize_chunk_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

            embeddings_list.append(emb)

        if embeddings_list:
            try:
                # Build array with optional float16 conversion
                if self.use_float16_embeddings:
                    self._chunk_embeddings_array = np.array(embeddings_list, dtype=np.float16)
                    logger.info(f"Chunk embeddings array built (float16): {len(self._chunk_ids_list)} chunks, "
                               f"memory: {self._chunk_embeddings_array.nbytes / 1024 / 1024:.2f} MB")
                else:
                    self._chunk_embeddings_array = np.array(embeddings_list)
                    logger.info(f"Chunk embeddings array built (float32): {len(self._chunk_ids_list)} chunks, "
                               f"memory: {self._chunk_embeddings_array.nbytes / 1024 / 1024:.2f} MB")
            except ValueError as e:
                logger.error(f"Failed to build chunk embeddings array: {e}")
                logger.error(f"  Total chunks: {len(self._chunk_ids_list)}")
                logger.error(f"  Valid embeddings: {len(embeddings_list)}")
                if embeddings_list:
                    logger.error(f"  First embedding shape: {embeddings_list[0].shape}")
                    logger.error(f"  Last embedding shape: {embeddings_list[-1].shape}")
                raise
        else:
            self._chunk_embeddings_array = np.array([])
            logger.warning("No chunk embeddings found")

    # ========== GraphStore Interface Implementation ==========

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build the complete graph index from a list of chunks.

        This method performs the following steps:
        1. Adds all chunks to Neo4j database
        2. Extracts and adds graph data (entities and facts) to Neo4j
        3. Generates embeddings for facts, entities, and chunks
        4. Optionally computes synonymy edges
        5. Rebuilds chunk embeddings array for dense retrieval

        Args:
            chunks: List of Chunk objects to index
        """
        logger.info(f"Building index from {len(chunks)} chunks...")

        batch_size = 1000
        total_chunks = len(chunks)

        from tqdm import tqdm

        logger.info("Step 1: Adding chunks and graph data to Neo4j...")
        for i in tqdm(range(0, total_chunks, batch_size), desc="Processing chunks"):
            batch_end = min(i + batch_size, total_chunks)
            batch = chunks[i:batch_end]

            # Batch insert chunks and graph data using optimized method
            self._batch_add_chunks_and_graph_data(batch)

        logger.info(f"All {total_chunks} chunks added to Neo4j")

        # Batch generate embeddings
        self.batch_generate_embeddings()

        # Compute and save synonymy edges to Neo4j (if enabled)
        if self.add_synonymy_edges:
            self._add_synonymy_edges()

        # Rebuild chunk embeddings array
        self._rebuild_chunk_embeddings_array()

        logger.info("Index building completed")

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """
        Update the graph index with new or modified chunks (incremental update).

        This method performs incremental updates:
        1. Adds new chunks and graph data to Neo4j (BATCH OPTIMIZED)
        2. Generates embeddings for new items only
        3. Incrementally computes synonymy edges for new entities only
        4. Incrementally updates graph cache
        5. Incrementally appends chunk embeddings to array (OPTIMIZED)

        Args:
            chunks: List of Chunk objects to add/update

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating index with {len(chunks)} chunks (incremental)...")

        try:
            # Track new chunk IDs and entity IDs for incremental updates
            new_chunk_ids = []
            new_entity_ids = []

            # Step 1: Batch add chunks and graph data (OPTIMIZED)
            logger.info("Step 1: Batch adding chunks and graph data...")
            new_entity_ids = self._batch_add_chunks_and_graph_data(chunks)
            new_chunk_ids = [chunk.id for chunk in chunks]
            logger.info("Step 1 completed: All chunks and graph data added")

            # Step 2: Batch generate embeddings (only for new items)
            logger.info("Step 2: Batch generating embeddings for new items...")
            self.batch_generate_embeddings()
            logger.info("Step 2 completed: Embeddings generated")

            # Step 3: Incrementally compute synonymy edges (only for new entities)
            if self.add_synonymy_edges:
                if new_entity_ids:
                    logger.info(f"Step 3: Computing synonymy edges for {len(new_entity_ids)} new entities (incremental)...")
                    self._add_synonymy_edges(new_entity_ids=new_entity_ids)
                    logger.info("Step 3 completed: Synonymy edges added incrementally")
                else:
                    logger.info("Step 3 skipped: No new entities to process")
            else:
                logger.info("Step 3 skipped: Synonymy edges disabled")

            # Step 4: Incrementally update graph cache
            logger.info("Step 4: Incrementally updating graph cache...")
            self._update_graph_cache_incremental(new_chunk_ids, new_entity_ids)
            logger.info("Step 4 completed: Graph cache updated incrementally")

            # Step 5: Incrementally append chunk embeddings (OPTIMIZED)
            logger.info("Step 5: Incrementally appending chunk embeddings...")
            self._append_chunk_embeddings(new_chunk_ids)
            logger.info("Step 5 completed: Chunk embeddings appended")

            logger.info("✅ Index update completed successfully (incremental)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update index: {e}", exc_info=True)
            return False

    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """
        Delete chunks and their associated graph data by IDs.

        This method:
        1. Deletes chunks from Neo4j (cascades to relations)
        2. Deletes orphan entities and facts
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

        try:
            # 1. Find entities that will become orphans
            orphan_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})-[:MENTIONS]->(e:Entity)
            WITH e, collect(DISTINCT chunk_id) AS deleted_chunks
            MATCH (e)<-[:MENTIONS]-(all_c:Chunk)
            WITH e, deleted_chunks, collect(DISTINCT all_c.chunk_id) AS all_chunks
            WHERE size(all_chunks) = size(deleted_chunks)
              AND all(dc IN deleted_chunks WHERE dc IN all_chunks)
            RETURN e.entity_id AS entity_id
            """

            orphan_results = self._execute_query(orphan_query, {'chunk_ids': chunk_ids})
            orphan_entities = [record['entity_id'] for record in orphan_results]

            # 2. Delete orphan entities and their facts
            if orphan_entities:
                # Find facts involving orphan entities
                fact_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})
                MATCH (f:Fact)
                WHERE f.head = e.entity_name OR f.tail = e.entity_name
                RETURN DISTINCT f.fact_id AS fact_id
                """

                fact_results = self._execute_query(fact_query, {'entity_ids': orphan_entities})
                orphan_fact_ids = [record['fact_id'] for record in fact_results]

                # Delete facts from FAISS
                if orphan_fact_ids:
                    self.fact_faiss_db.delete_index(orphan_fact_ids)
                    logger.info(f"Deleted {len(orphan_fact_ids)} orphan facts from FAISS")

                # Delete entities from FAISS
                self.entity_faiss_db.delete_index(orphan_entities)
                logger.info(f"Deleted {len(orphan_entities)} orphan entities from FAISS")

                # Delete facts from Neo4j
                delete_facts_query = """
                UNWIND $fact_ids AS fact_id
                MATCH (f:Fact {fact_id: fact_id})
                DELETE f
                """
                self._execute_query(delete_facts_query, {'fact_ids': orphan_fact_ids})

                # Delete entities from Neo4j
                delete_entities_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})
                DETACH DELETE e
                """
                self._execute_query(delete_entities_query, {'entity_ids': orphan_entities})

            # 3. Delete chunks from Neo4j (DETACH DELETE removes all relationships)
            delete_chunks_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            DETACH DELETE c
            """
            self._execute_query(delete_chunks_query, {'chunk_ids': chunk_ids})

            # 4. Delete from chunk_embeddings (not FAISS)
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_embeddings:
                    del self.chunk_embeddings[chunk_id]

            # Mark array needs rebuild
            self._chunk_embeddings_array = None

            logger.info(f"Deleted {len(chunk_ids)} chunks, {len(orphan_entities)} orphan entities")
            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}", exc_info=True)
            return False

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graphs"""
        if not confirm:
            logger.warning("delete_all_index requires confirm=True")
            return False

        logger.info("Deleting all index data...")

        try:
            # Delete all nodes and relationships from Neo4j
            delete_query = """
            MATCH (n)
            WHERE n:Chunk OR n:Entity OR n:Fact
            DETACH DELETE n
            """
            self._execute_query(delete_query)

            # Clear FAISS indices
            # Note: FAISS doesn't have a clear method, so we recreate the indices
            self._init_faiss_indices()

            # Clear chunk embeddings
            self.chunk_embeddings = {}
            self._chunk_embeddings_array = None
            self._chunk_ids_list = None

            logger.info("All index data deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete all index: {e}", exc_info=True)
            return False

    def get_by_ids(self, ids: Sequence[str]) -> List[Chunk]:
        """
        Retrieve chunks and their associated graph data by IDs.

        Args:
            ids: Sequence of chunk IDs to retrieve

        Returns:
            List of Chunk objects with graph data
        """
        chunks = []

        for chunk_id in ids:
            # Get chunk data
            chunk_query = """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN c.chunk_id AS chunk_id, c.content AS content,
                   c.owner_id AS owner_id, c.metadata AS metadata
            """

            result = self._execute_query(chunk_query, {'chunk_id': chunk_id})

            if result:
                record = result[0]
                content = record['content']
                owner_id = record['owner_id']
                metadata = json.loads(record['metadata']) if record['metadata'] else {}

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
        # Get entities for this chunk
        entity_query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN e.entity_id AS entity_id, e.entity_name AS entity_name,
               e.entity_type AS entity_type, e.attributes AS attributes
        """

        entity_results = self._execute_query(entity_query, {'chunk_id': chunk_id})

        entities = []
        entity_names = set()
        for record in entity_results:
            entity_id = record['entity_id']
            entity_name = record['entity_name']
            entity_type = record['entity_type']
            attributes_str = record['attributes']

            entities.append({
                'id': entity_id,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'attributes': json.loads(attributes_str) if attributes_str else {}
            })
            entity_names.add(entity_name)

        # Get relations (facts) from :Fact relationships between entities
        relations = []
        if entity_names:
            relation_query = """
            MATCH (e1:Entity)-[r:Fact]->(e2:Entity)
            WHERE e1.entity_name IN $entity_names AND e2.entity_name IN $entity_names
            RETURN r.head AS head, r.relation AS relation, r.tail AS tail
            """

            relation_results = self._execute_query(relation_query, {'entity_names': list(entity_names)})

            for record in relation_results:
                relations.append([record['head'], record['relation'], record['tail']])

        return GraphData(entities=entities, relations=relations, metadata={})

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

        Args:
            path: Directory path to load the index from
            name: Base name for index files
        """
        # 1. Load chunk embeddings
        embeddings_path = os.path.join(path, f"{name}_chunk_embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.chunk_embeddings = pickle.load(f)
            logger.info(f"Loaded chunk embeddings from {embeddings_path}")
            self._chunk_embeddings_array = None  # Mark for rebuild
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

        logger.info(f"Index loaded from {path}")
        logger.info("Note: Neo4j data is loaded automatically from the database")

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run a Cypher query on the Neo4j database.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            Query results
        """
        return self._execute_query(query, params)

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
        fact_count_query = "MATCH ()-[r:Fact]->() RETURN count(r) AS count"
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
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")

