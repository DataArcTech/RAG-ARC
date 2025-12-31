import os
import json
import logging
import pickle
from typing import List, Dict, Any, Optional, Sequence

import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)

from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jIndexingMixin:
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
                    loaded = pickle.load(f)
                with self.write_lock():
                    self.chunk_embeddings = loaded
                    # Mark array for rebuild on first use
                    self._chunk_embeddings_array = None
                logger.info(f"Loaded {len(self.chunk_embeddings)} chunk embeddings from {embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to load chunk embeddings: {e}")
        else:
            logger.info(f"No existing chunk embeddings found at {embeddings_path}")

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
        entity_data: Dict[str, Dict[str, Any]] = {}  # entity_id -> entity payload
        mention_data = []
        fact_data = []
        new_entity_ids = []

        for chunk in chunks:
            # Prepare chunk data
            metadata = dict(chunk.metadata) if chunk.metadata else {}

            owner_source = chunk.owner_id or metadata.get('owner_id')
            owner_str = self._normalize_owner_id(owner_source)
            if owner_str:
                metadata['owner_id'] = owner_str
            db_owner_id = self._owner_key(owner_str)

            chunk_data.append({
                'chunk_id': chunk.id,
                'content': chunk.content,
                'metadata': json.dumps(metadata) if metadata else '{}',
                'owner_id': db_owner_id
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
                    entity_id = compute_mdhash_id(entity_name, prefix='entity-', owner_id=owner_str)
                    if entity_id not in entity_data:
                        # Get entity type from mapping, default to 'Entity'
                        entity_type = entity_name_to_type.get(entity_name, 'Entity')
                        entity_data[entity_id] = {
                            'entity_id': entity_id,
                            'entity_name': entity_name,
                            'entity_type': entity_type,
                            'owner_id': db_owner_id
                        }

                    # Collect mention data
                    mention_data.append({
                        'chunk_id': chunk.id,
                        'entity_id': entity_id,
                        'owner_id': db_owner_id
                    })

                # Collect fact data
                for head_name, relation_type, tail_name in processed_triples:
                    fact_text = str((head_name, relation_type, tail_name))
                    fact_id = compute_mdhash_id(fact_text, prefix='fact-', owner_id=owner_str)
                    head_id = compute_mdhash_id(head_name, prefix='entity-', owner_id=owner_str)
                    tail_id = compute_mdhash_id(tail_name, prefix='entity-', owner_id=owner_str)

                    fact_data.append({
                        'fact_id': fact_id,
                        'head_id': head_id,
                        'tail_id': tail_id,
                        'head_name': head_name,
                        'relation_type': relation_type,
                        'tail_name': tail_name,
                        'fact_text': fact_text,
                        'owner_id': db_owner_id
                    })

        # Prepare entity list for batch insertion
        entity_list = list(entity_data.values())

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
                                  e.owner_id = entity.owner_id,
                                  e.created_at = datetime(),
                                  e.updated_at = datetime(),
                                  e.is_new = true
                    ON MATCH SET e.entity_name = entity.entity_name,
                                 e.entity_text = entity.entity_name,
                                 e.entity_type = entity.entity_type,
                                 e.owner_id = entity.owner_id,
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
                    MATCH (c:Chunk {chunk_id: m.chunk_id, owner_id: m.owner_id})
                    MATCH (e:Entity {entity_id: m.entity_id, owner_id: m.owner_id})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.weight = COALESCE(r.weight, 0.0) + 1.0,
                        r.owner_id = m.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(mention_query, {'mentions': mention_data})
                    logger.info(f"  Batch created {len(mention_data)} MENTIONS relationships")

                # 4. Batch create fact relationships
                if fact_data:
                    fact_query = """
                    UNWIND $facts AS f
                    MATCH (e1:Entity {entity_id: f.head_id, owner_id: f.owner_id})
                    MATCH (e2:Entity {entity_id: f.tail_id, owner_id: f.owner_id})
                    MERGE (e1)-[r:RELATES_TO {fact_id: f.fact_id}]->(e2)
                    SET r.head = f.head_name,
                        r.predicate = f.relation_type,
                        r.tail = f.tail_name,
                        r.text = f.fact_text,
                        r.owner_id = f.owner_id,
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

        with self.read_lock():
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
            with self.write_lock():
                if self._chunk_embeddings_array is not None and len(self._chunk_embeddings_array) > 0:
                    self._chunk_embeddings_array = np.vstack([self._chunk_embeddings_array, new_array])
                    if self._chunk_ids_list is None:
                        self._chunk_ids_list = []
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
        with self.read_lock():
            if self._chunk_embeddings_array is not None:
                return  # Already built
            chunk_ids = list(self.chunk_embeddings.keys())

        logger.info("Rebuilding chunk embeddings array...")

        kept_chunk_ids: list[str] = []
        embeddings_list: list[np.ndarray] = []

        with self.read_lock():
            for i, cid in enumerate(chunk_ids):
                emb = self.chunk_embeddings.get(cid)
                if emb is None:
                    continue
                if isinstance(emb, list):
                    emb = np.array(emb)
                elif not isinstance(emb, np.ndarray):
                    logger.warning(f"Chunk {cid} has invalid embedding type: {type(emb)}")
                    continue

                if embeddings_list and emb.shape != embeddings_list[0].shape:
                    logger.error(f"Chunk {cid} (index {i}) has shape {emb.shape}, expected {embeddings_list[0].shape}")
                    logger.error(f"  First chunk ID: {kept_chunk_ids[0] if kept_chunk_ids else 'N/A'}, shape: {embeddings_list[0].shape}")
                    logger.error(f"  Current chunk ID: {cid}, shape: {emb.shape}")
                    continue

                if self.normalize_chunk_embeddings:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                kept_chunk_ids.append(cid)
                embeddings_list.append(emb)

        if not embeddings_list:
            new_array = np.array([])
            logger.warning("No chunk embeddings found")
        else:
            try:
                if self.use_float16_embeddings:
                    new_array = np.array(embeddings_list, dtype=np.float16)
                else:
                    new_array = np.array(embeddings_list, dtype=np.float32)
            except ValueError as e:
                logger.error(f"Failed to build chunk embeddings array: {e}")
                logger.error(f"  Total chunks: {len(chunk_ids)}")
                logger.error(f"  Valid embeddings: {len(embeddings_list)}")
                if embeddings_list:
                    logger.error(f"  First embedding shape: {embeddings_list[0].shape}")
                    logger.error(f"  Last embedding shape: {embeddings_list[-1].shape}")
                raise

        with self.write_lock():
            if self._chunk_embeddings_array is not None:
                return
            self._chunk_ids_list = kept_chunk_ids
            self._chunk_embeddings_array = new_array

        if len(new_array) > 0 and isinstance(new_array, np.ndarray):
            dtype_str = "float16" if self.use_float16_embeddings else "float32"
            logger.info(
                f"Chunk embeddings array built ({dtype_str}): {len(kept_chunk_ids)} chunks, "
                f"memory: {new_array.nbytes / 1024 / 1024:.2f} MB"
            )

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
            self.batch_generate_embeddings(chunk_ids=new_chunk_ids, entity_ids=new_entity_ids)
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

            # Step 6: Increment cache version to notify retrievers
            with self.write_lock():
                self._cache_version += 1
                cache_version = self._cache_version
            
            logger.info(f"✅ Index update completed successfully (incremental, cache_version={cache_version})")
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
        """Delete chunks and clean up orphan nodes.
        
        This method:
        1. Finds entities that will become orphans after chunk deletion
        2. Finds facts (RELATES_TO relationships) involving orphan entities
        3. Deletes orphan facts from FAISS (soft-delete)
        4. Deletes orphan entities from FAISS (soft-delete)
        5. Deletes orphan entities and their relationships from Neo4j
        6. Deletes chunks from Neo4j
        7. Updates in-memory caches (chunk_embeddings, graph_cache)
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
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
            RETURN e.entity_id AS entity_id, e.entity_name AS entity_name
            """

            orphan_results = self._execute_query(orphan_query, {'chunk_ids': chunk_ids})
            orphan_entities = [record['entity_id'] for record in orphan_results]
            orphan_entity_names = [record['entity_name'] for record in orphan_results]

            orphan_fact_ids = []
            
            # 2. Delete orphan entities and their facts
            if orphan_entities:
                # Find facts (RELATES_TO relationships) involving orphan entities
                # Facts are stored as RELATES_TO relationships with fact_id property
                fact_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})-[r:RELATES_TO]-()
                RETURN DISTINCT r.fact_id AS fact_id
                """

                fact_results = self._execute_query(fact_query, {'entity_ids': orphan_entities})
                orphan_fact_ids = [record['fact_id'] for record in fact_results if record['fact_id']]

                # Delete facts from FAISS (soft-delete)
                if orphan_fact_ids:
                    self.fact_faiss_db.delete_index(orphan_fact_ids)
                    logger.info(f"Soft-deleted {len(orphan_fact_ids)} orphan facts from FAISS")

                # Delete entities from FAISS (soft-delete)
                self.entity_faiss_db.delete_index(orphan_entities)
                logger.info(f"Soft-deleted {len(orphan_entities)} orphan entities from FAISS")

                # Delete entities from Neo4j (DETACH DELETE removes all relationships including RELATES_TO)
                delete_entities_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})
                DETACH DELETE e
                """
                self._execute_query(delete_entities_query, {'entity_ids': orphan_entities})
                logger.info(f"Deleted {len(orphan_entities)} orphan entities from Neo4j")

            # 3. Delete chunks from Neo4j (DETACH DELETE removes all relationships)
            delete_chunks_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            DETACH DELETE c
            """
            self._execute_query(delete_chunks_query, {'chunk_ids': chunk_ids})
            logger.info(f"Deleted {len(chunk_ids)} chunks from Neo4j")

            # 4. Delete from chunk_embeddings
            with self.write_lock():
                for chunk_id in chunk_ids:
                    if chunk_id in self.chunk_embeddings:
                        del self.chunk_embeddings[chunk_id]

            # 5. Invalidate chunk embeddings array (mark for rebuild)
            with self.write_lock():
                self._chunk_embeddings_array = None
                self._chunk_ids_list = None

            # 6. Update graph cache and entity count cache
            self._invalidate_graph_cache_for_deleted_nodes(chunk_ids, orphan_entities)

            # 7. Increment cache version to notify retrievers
            with self.write_lock():
                self._cache_version += 1
                cache_version = self._cache_version
            
            logger.info(f"✅ Deleted {len(chunk_ids)} chunks, {len(orphan_entities)} orphan entities, "
                       f"{len(orphan_fact_ids)} orphan facts (cache_version={cache_version})")
            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}", exc_info=True)
            return False
    
    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graphs.
        
        This method completely clears all data:
        1. Deletes all nodes and relationships from Neo4j
        2. Reinitializes FAISS indices (clears all vectors)
        3. Clears all in-memory caches
        
        Args:
            confirm: Must be True to confirm the operation
            
        Returns:
            True if successful, False otherwise
        """
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

            with self.write_lock():
                # Clear chunk embeddings
                self.chunk_embeddings = {}
                self._chunk_embeddings_array = None
                self._chunk_ids_list = None

                # Clear graph cache
                self._graph_cache = {}
                self._cache_loaded = True  # Mark as loaded (empty cache is valid)

                # Clear entity chunk count cache
                self._entity_chunk_count_cache = {}

                # Increment cache version
                self._cache_version += 1
                cache_version = self._cache_version

            logger.info(f"✅ All index data deleted (cache_version={cache_version})")
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

