"""GraphStore indexing CRUD operations for PrunedHippoRAG Neo4j store."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from encapsulation.data_model.schema import Chunk, GraphData

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jIndexingOpsMixin:
    # ========== GraphStore Interface Implementation ==========

    def _cleanup_relates_to_source_chunk_ids(self, *, chunk_ids: List[str], owner_ids: List[Optional[str]]) -> int:
        """
        Remove deleted chunk_ids from `(:Entity)-[:RELATES_TO].source_chunk_ids` provenance lists.

        Why this exists
        ---------------
        `RELATES_TO.source_chunk_ids` is append-only during ingest (facts accumulate multiple evidences).
        When chunks are deleted/reindexed, if we don't clean these lists, we accumulate dangling provenance
        that can break downstream graph rebuild/analysis (and can bias graph scoring).

        Notes
        -----
        - This does NOT delete the `RELATES_TO` relationship itself; it only updates provenance.
        - Owner scoping uses the same normalization as the store (None -> OWNER_GLOBAL_KEY).
        """
        deleted = [str(cid) for cid in (chunk_ids or []) if str(cid).strip()]
        if not deleted:
            return 0

        owners = sorted({self._restore_owner_id(oid) for oid in (owner_ids or [])})
        if not owners:
            # Fallback: owner unknown; still safe to clean globally by matching chunk ids.
            owners = [None]

        updated_total = 0
        for owner in owners:
            owner_key = self._owner_key(owner)
            cleanup_query = """
            MATCH ()-[r:RELATES_TO]-()
            WHERE coalesce(r.owner_id, $owner_key) = $owner_key
              AND any(cid IN $deleted_chunk_ids WHERE cid IN coalesce(r.source_chunk_ids, []))
            WITH r, [cid IN coalesce(r.source_chunk_ids, []) WHERE NOT cid IN $deleted_chunk_ids] AS filtered
            SET r.source_chunk_ids = filtered
            RETURN count(r) AS updated
            """
            try:
                rows = self._execute_query(
                    cleanup_query,
                    {"owner_key": owner_key, "deleted_chunk_ids": deleted},
                )
                if rows and isinstance(rows, list):
                    for rec in rows:
                        try:
                            updated_total += int(rec.get("updated") or 0)
                        except Exception:
                            continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to cleanup RELATES_TO.source_chunk_ids (owner=%s chunks=%s): %s",
                    owner_key,
                    len(deleted),
                    exc,
                )
        return updated_total

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

        batch_size = int(getattr(getattr(self, "config", None), "neo4j_ingest_chunk_batch_size", 1000) or 1000)
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

    def _update_index_impl(self, chunks: List[Chunk], *, collect_stats: bool) -> tuple[bool, Dict[str, Any]]:
        stats: Dict[str, Any] = {"timings_s": {}, "counts": {}} if collect_stats else {}
        t_total0 = time.perf_counter()

        logger.info(f"Updating index with {len(chunks)} chunks (incremental)...")

        try:
            # Track new chunk IDs and entity IDs for incremental updates
            new_chunk_ids: List[str] = []
            new_entity_ids: List[str] = []

            # Step 1: Batch add chunks and graph data (OPTIMIZED)
            logger.info("Step 1: Batch adding chunks and graph data...")
            t0 = time.perf_counter()
            ingest_batch_size = int(getattr(getattr(self, "config", None), "neo4j_ingest_chunk_batch_size", 1000) or 1000)
            new_entity_ids = []
            for i in range(0, len(chunks), ingest_batch_size):
                batch = chunks[i : i + ingest_batch_size]
                new_entity_ids.extend(self._batch_add_chunks_and_graph_data(batch) or [])
            # Deduplicate while preserving stability for downstream embedding selection.
            if new_entity_ids:
                seen = set()
                deduped: List[str] = []
                for eid in new_entity_ids:
                    s = str(eid or "").strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    deduped.append(s)
                new_entity_ids = deduped
            # Normalize IDs to strings: chunk.id can be UUID-like objects depending on upstream pipelines,
            # but Neo4j stores `Chunk.chunk_id` as a string.
            new_chunk_ids = [str(chunk.id).strip() for chunk in chunks if str(getattr(chunk, "id", "") or "").strip()]
            if collect_stats:
                stats["timings_s"]["step1_add_chunks_graph"] = time.perf_counter() - t0
                stats["counts"]["input_chunks"] = int(len(chunks))
                stats["counts"]["new_chunks"] = int(len(new_chunk_ids))
                stats["counts"]["new_entities"] = int(len(new_entity_ids or []))
            logger.info("Step 1 completed: All chunks and graph data added")

            # Step 2: Batch generate embeddings (only for new items)
            logger.info("Step 2: Batch generating embeddings for new items...")
            t0 = time.perf_counter()
            embed_summary = self.batch_generate_embeddings(chunk_ids=new_chunk_ids, entity_ids=new_entity_ids)
            if collect_stats:
                stats["timings_s"]["step2_embeddings"] = time.perf_counter() - t0
                stats["embeddings"] = embed_summary
            logger.info("Step 2 completed: Embeddings generated")

            # Step 3: Incrementally compute synonymy edges (only for new entities)
            t0 = time.perf_counter()
            if self.add_synonymy_edges:
                if new_entity_ids:
                    logger.info(
                        f"Step 3: Computing synonymy edges for {len(new_entity_ids)} new entities (incremental)..."
                    )
                    self._add_synonymy_edges(new_entity_ids=new_entity_ids)
                    logger.info("Step 3 completed: Synonymy edges added incrementally")
                else:
                    logger.info("Step 3 skipped: No new entities to process")
            else:
                logger.info("Step 3 skipped: Synonymy edges disabled")
            if collect_stats:
                stats["timings_s"]["step3_synonymy_edges"] = time.perf_counter() - t0

            # Step 4: Incrementally update graph cache
            logger.info("Step 4: Incrementally updating graph cache...")
            t0 = time.perf_counter()
            self._update_graph_cache_incremental(new_chunk_ids, new_entity_ids)
            if collect_stats:
                stats["timings_s"]["step4_update_graph_cache"] = time.perf_counter() - t0
            logger.info("Step 4 completed: Graph cache updated incrementally")

            # Step 5: Incrementally append chunk embeddings (OPTIMIZED)
            logger.info("Step 5: Incrementally appending chunk embeddings...")
            t0 = time.perf_counter()
            self._append_chunk_embeddings(new_chunk_ids)
            if collect_stats:
                stats["timings_s"]["step5_append_chunk_embeddings"] = time.perf_counter() - t0
            logger.info("Step 5 completed: Chunk embeddings appended")

            # Step 6: Increment cache version to notify retrievers
            with self.write_lock():
                self._cache_version += 1
                cache_version = self._cache_version

            if collect_stats:
                stats["cache_version"] = int(cache_version)
                stats["timings_s"]["total"] = time.perf_counter() - t_total0

            logger.info(f"✅ Index update completed successfully (incremental, cache_version={cache_version})")
            return True, stats

        except Exception as e:
            logger.error(f"❌ Failed to update index: {e}", exc_info=True)
            if collect_stats:
                stats["error"] = str(e)
                stats["timings_s"]["total"] = time.perf_counter() - t_total0
            return False, stats

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
        ok, _stats = self._update_index_impl(chunks, collect_stats=False)
        return ok

    def update_index_with_stats(self, chunks: List[Chunk]) -> tuple[bool, Dict[str, Any]]:
        """Like update_index(), but returns a JSON-serializable stats dict for profiling/benchmarking."""
        return self._update_index_impl(chunks, collect_stats=True)

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
            # Capture owner_ids before deleting Chunk nodes so we can owner-scope provenance cleanup.
            owner_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            RETURN DISTINCT c.owner_id AS owner_id
            """
            owner_rows = self._execute_query(owner_query, {"chunk_ids": chunk_ids}) or []
            owners_for_cleanup: List[Optional[str]] = []
            for row in owner_rows:
                if isinstance(row, dict):
                    owners_for_cleanup.append(row.get("owner_id"))

            orphan_entities: List[str] = []
            orphan_fact_ids: List[str] = []

            # 1. Find entities that will become orphans
            orphan_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})-[:MENTIONS]->(e:Entity)
            WITH e, collect(DISTINCT chunk_id) AS deleted_chunks
            MATCH (e)<-[:MENTIONS]-(all_c:Chunk)
            WITH e, deleted_chunks, collect(DISTINCT all_c.chunk_id) AS all_chunks
            WHERE size(all_chunks) = size(deleted_chunks)
              AND all(dc IN deleted_chunks WHERE dc IN all_chunks)
            RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id
            """

            orphan_results = self._execute_query(orphan_query, {'chunk_ids': chunk_ids})
            orphan_by_owner: Dict[Optional[str], List[str]] = {}
            for record in orphan_results:
                owner = self._restore_owner_id(record.get("owner_id"))
                orphan_by_owner.setdefault(owner, []).append(record["entity_id"])

            # 2. Delete orphan entities and their facts
            if orphan_by_owner:
                # Find facts (RELATES_TO relationships) involving orphan entities
                # Facts are stored as RELATES_TO relationships with fact_id property
                fact_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})-[r:RELATES_TO]-()
                RETURN DISTINCT r.fact_id AS fact_id, r.owner_id AS owner_id
                """

                orphan_entity_set: set[str] = set()
                for ids in orphan_by_owner.values():
                    for entity_id in ids:
                        if entity_id:
                            orphan_entity_set.add(str(entity_id))
                orphan_entities = sorted(orphan_entity_set)

                fact_results = self._execute_query(fact_query, {'entity_ids': orphan_entities})
                orphan_facts_by_owner: Dict[Optional[str], List[str]] = {}
                for record in fact_results:
                    fid = record.get("fact_id")
                    if not fid:
                        continue
                    owner = self._restore_owner_id(record.get("owner_id"))
                    fid_str = str(fid)
                    orphan_facts_by_owner.setdefault(owner, []).append(fid_str)
                    orphan_fact_ids.append(fid_str)

                # Delete facts from FAISS (soft-delete)
                for owner, fact_ids in orphan_facts_by_owner.items():
                    if not fact_ids:
                        continue
                    fact_db = self.get_fact_faiss_db(owner)
                    fact_db.delete_index(fact_ids)
                    fact_index_path = self._faiss_owner_scoped_dir("fact", owner_id=owner)
                    fact_db.save_index(fact_index_path, "index")
                    logger.info("Soft-deleted %s orphan facts from FAISS (owner=%s)", len(fact_ids), owner)

                # Delete entities from FAISS (soft-delete)
                for owner, entity_ids in orphan_by_owner.items():
                    if not entity_ids:
                        continue
                    entity_db = self.get_entity_faiss_db(owner)
                    entity_db.delete_index(entity_ids)
                    entity_index_path = self._faiss_owner_scoped_dir("entity", owner_id=owner)
                    entity_db.save_index(entity_index_path, "index")
                    logger.info("Soft-deleted %s orphan entities from FAISS (owner=%s)", len(entity_ids), owner)

                # Delete entities from Neo4j (DETACH DELETE removes all relationships including RELATES_TO)
                delete_entities_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {entity_id: entity_id})
                DETACH DELETE e
                """
                self._execute_query(delete_entities_query, {'entity_ids': orphan_entities})
                logger.info("Deleted %s orphan entities from Neo4j", len(orphan_entities))

            # 3. Delete chunks from Neo4j (DETACH DELETE removes all relationships)
            delete_chunks_query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            DETACH DELETE c
            """
            self._execute_query(delete_chunks_query, {'chunk_ids': chunk_ids})
            logger.info(f"Deleted {len(chunk_ids)} chunks from Neo4j")

            # 3.5 Cleanup dangling provenance in facts (RELATES_TO.source_chunk_ids).
            # This must run after deletions, otherwise deleted chunk_ids will remain referenced forever.
            updated = self._cleanup_relates_to_source_chunk_ids(chunk_ids=chunk_ids, owner_ids=owners_for_cleanup)
            if updated:
                logger.info("Cleaned RELATES_TO.source_chunk_ids: updated=%s (deleted_chunks=%s)", updated, len(chunk_ids))

            # 4. Delete from chunk_embeddings
            with self.write_lock():
                for chunk_id in chunk_ids:
                    if chunk_id in self.chunk_embeddings:
                        del self.chunk_embeddings[chunk_id]
                    try:
                        owner = self._chunk_embedding_owner_by_chunk_id.pop(str(chunk_id), None)
                        if owner in self._chunk_ids_by_owner:
                            self._chunk_ids_by_owner.get(owner, set()).discard(str(chunk_id))
                        self._chunk_embeddings_dirty_owners.add(owner)
                    except Exception:
                        pass

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

            logger.info(
                f"✅ Deleted {len(chunk_ids)} chunks, {len(orphan_entities)} orphan entities, "
                f"{len(orphan_fact_ids)} orphan facts (cache_version={cache_version})"
            )
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
                   c.owner_id AS owner_id, c.metadata AS metadata,
                   c.source_file_id AS source_file_id
            """

            result = self._execute_query(chunk_query, {'chunk_id': chunk_id})

            if result:
                record = result[0]
                content = record['content']
                owner_id = record['owner_id']
                metadata = json.loads(record['metadata']) if record['metadata'] else {}
                source_file_id = record.get("source_file_id")

                # Normalize file provenance so callers can filter deleted files consistently.
                if isinstance(metadata, dict) and source_file_id and "source_file_id" not in metadata:
                    metadata["source_file_id"] = source_file_id

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

        # Get relations (facts) from :RELATES_TO relationships between entities
        relations = []
        if entity_names:
            relation_query = """
            MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
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
