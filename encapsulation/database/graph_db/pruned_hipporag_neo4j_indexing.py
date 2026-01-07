import os
import json
import logging
from typing import List, Dict, Any, Optional, Sequence

from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.graph_db.pruned_hipporag_neo4j_chunk_embeddings import _PrunedHippoRAGNeo4jChunkEmbeddingsMixin
from encapsulation.database.utils.fact_provenance import upsert_fact_occurrence
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing
from core.knowledge_graph.schema import normalize_relation_token

logger = logging.getLogger(__name__)


def _coerce_fact_provenance_max_source_chunks(raw: Any, *, default: int = 50, max_value: int = 1000) -> int:
    """
    Coerce `fact_provenance_max_source_chunks` into a safe integer.

    Note:
    - `0` is a valid value meaning "disable source_chunk_ids storage", so do NOT use `or default`.
    """
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(default)
    return max(0, min(int(max_value), value))

class _PrunedHippoRAGNeo4jIndexingMixin(_PrunedHippoRAGNeo4jChunkEmbeddingsMixin):
    def _init_faiss_indices(self):
        """
        Initialize FAISS indices for facts and entities.

        - Fact index: FAISS Flat (exact search) for fact retrieval
        - Entity index: FAISS HNSW (approximate search) for synonymy edge computation
        """
        from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig

        storage_path = self.config.storage_path
        index_name = self.config.index_name
        os.makedirs(storage_path, exist_ok=True)

        # Initialize fact index (FAISS Flat for exact search)
        fact_config = FaissVectorDBConfig(
            embedding_config=self.config.embedding,
            index_type='flat',
            metric='cosine',
            normalize_L2=True,
            index_path=os.path.join(storage_path, 'fact_index'),
            index_name=index_name,
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
            m=self.config.hnsw_M,
            efConstruction=self.config.hnsw_ef_construction,
            efSearch=self.config.hnsw_ef_search,
            index_path=os.path.join(storage_path, 'entity_index'),
            index_name=index_name,
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
        fact_data_by_id: Dict[str, Dict[str, Any]] = {}
        new_entity_ids = []

        # HippoRAG chunk-triples contract: keep only triples whose endpoints are extracted named entities.
        # (Precision-first; endpoints outside extracted entities are dropped.)

        stats_by_owner: Dict[str, Dict[str, int]] = {}
        cfg = getattr(self, "config", None)
        max_source_chunks = _coerce_fact_provenance_max_source_chunks(
            getattr(cfg, "fact_provenance_max_source_chunks", None),
            default=50,
            max_value=1000,
        )

        for chunk in chunks:
            # Prepare chunk data
            metadata = dict(chunk.metadata) if chunk.metadata else {}

            owner_source = chunk.owner_id or metadata.get('owner_id')
            owner_str = self._normalize_owner_id(owner_source)
            if owner_str:
                metadata['owner_id'] = owner_str
            db_owner_id = self._owner_key(owner_str)
            owner_stats = stats_by_owner.setdefault(
                db_owner_id,
                {
                    "triples_total": 0,
                    "triples_kept": 0,
                    "triples_dropped_endpoints": 0,
                    "triples_dropped_ambiguous_endpoints": 0,
                    "triples_dropped_schema": 0,
                },
            )

            # Extract source_file_id from metadata for independent storage
            source_file_id = metadata.get("source_file_id")
            
            chunk_data.append({
                'chunk_id': chunk.id,
                'content': chunk.content,
                'metadata': json.dumps(metadata) if metadata else '{}',
                'owner_id': db_owner_id,
                'source_file_id': source_file_id  # Store as independent property for filtering
            })

            # Process graph data
            if chunk.graph and not chunk.graph.is_empty():
                domain_raw = chunk.domain or metadata.get("domain") or "default"
                domain = str(domain_raw).strip() or "default"
                schema = getattr(self, "kg_schema", None)
                domain_schema = schema.for_domain(domain) if schema else None
                schema_version = getattr(schema, "version", None) or "unmanaged"

                # Build entity mapping from NER outputs.
                # - `entity_name_normalized` is used for stable matching + hashing.
                # - `entity_name` keeps the display value for explainability/exports.
                entity_name_to_type_keys: dict[str, set[str]] = {}
                entity_key_to_type_display: dict[tuple[str, str], str] = {}
                entity_key_to_display: dict[tuple[str, str], str] = {}
                entity_key_to_canonical: dict[tuple[str, str], str] = {}
                for entity_dict in chunk.graph.entities:
                    raw_name = entity_dict.get("entity_name")
                    if not raw_name:
                        continue
                    display_name = str(raw_name).strip()
                    if not display_name:
                        continue

                    normalized_name = text_processing(display_name)
                    if not normalized_name:
                        continue

                    entity_type_display = str(entity_dict.get("entity_type", "Entity") or "Entity").strip() or "Entity"
                    entity_type_key = text_processing(entity_type_display) or "entity"

                    entity_name_to_type_keys.setdefault(normalized_name, set()).add(entity_type_key)
                    entity_key = (normalized_name, entity_type_key)
                    entity_key_to_type_display.setdefault(entity_key, entity_type_display)
                    entity_key_to_display.setdefault(entity_key, display_name)

                    canonical = None
                    if domain_schema is not None:
                        canonical = domain_schema.canonicalize_entity_name(display_name)
                    canonical_name = canonical or normalized_name
                    entity_key_to_canonical[entity_key] = canonical_name

                # Collect entity nodes + chunk mentions from NER outputs (not only triple endpoints).
                # This keeps the entity layer usable even when relation extraction is sparse or schema-rejected.
                mention_keys: set[tuple[str, str]] = set()
                for (normalized_name, entity_type_key), entity_type_display in entity_key_to_type_display.items():
                    entity_id = compute_mdhash_id(
                        f"{normalized_name}|{entity_type_key}",
                        prefix="entity-",
                        owner_id=owner_str,
                    )
                    display_name = entity_key_to_display.get((normalized_name, entity_type_key)) or normalized_name
                    if entity_id not in entity_data:
                        canonical_name = entity_key_to_canonical.get((normalized_name, entity_type_key)) or normalized_name
                        entity_data[entity_id] = {
                            "entity_id": entity_id,
                            "entity_name": display_name,
                            "entity_name_normalized": normalized_name,
                            "entity_canonical_name": canonical_name,
                            # Canonical key keeps type to avoid collapsing same-name different-type entities in COUNT DISTINCT.
                            "entity_canonical_key": f"{canonical_name}|{entity_type_key}",
                            "entity_type": entity_type_display or "Entity",
                            "entity_type_key": entity_type_key,
                            "owner_id": db_owner_id,
                        }
                    mention_key = (chunk.id, entity_id)
                    if mention_key not in mention_keys:
                        mention_keys.add(mention_key)
                        mention_data.append({
                            'chunk_id': chunk.id,
                            'entity_id': entity_id,
                            'owner_id': db_owner_id
                        })

                # Process and normalize relation triples (schema-governed predicate normalization)
                processed_triples: list[tuple[str, str, str, str, str]] = []
                for relation in chunk.graph.relations:
                    if len(relation) >= 3:
                        head = text_processing(relation[0])
                        tail = text_processing(relation[2])

                        if not head or not tail:
                            continue
                        owner_stats["triples_total"] += 1

                        # Enforce HippoRAG chunk-triples endpoint constraint (precision-first):
                        # triples must use extracted named entities as endpoints.
                        if head not in entity_name_to_type_keys or tail not in entity_name_to_type_keys:
                            owner_stats["triples_dropped_endpoints"] += 1
                            continue
                        head_types = entity_name_to_type_keys.get(head) or set()
                        tail_types = entity_name_to_type_keys.get(tail) or set()
                        if len(head_types) != 1 or len(tail_types) != 1:
                            # Without entity linking, multiple types for the same surface form is ambiguous.
                            # Drop the triple to avoid mixing semantics across types (precision-first).
                            owner_stats["triples_dropped_ambiguous_endpoints"] += 1
                            continue
                        head_type_key = next(iter(head_types))
                        tail_type_key = next(iter(tail_types))
                        head_id = compute_mdhash_id(f"{head}|{head_type_key}", prefix="entity-", owner_id=owner_str)
                        tail_id = compute_mdhash_id(f"{tail}|{tail_type_key}", prefix="entity-", owner_id=owner_str)
                        head_display = entity_key_to_display.get((head, head_type_key)) or head
                        tail_display = entity_key_to_display.get((tail, tail_type_key)) or tail

                        raw_predicate = relation[1]
                        if domain_schema is not None:
                            normalized_predicate = domain_schema.normalize_predicate(str(raw_predicate))
                        else:
                            # Keep predicate token shape stable even without schema governance.
                            normalized_predicate = normalize_relation_token(str(raw_predicate))
                        if not normalized_predicate:
                            owner_stats["triples_dropped_schema"] += 1
                            continue

                        processed_triples.append((head_id, head_display, normalized_predicate, tail_id, tail_display))
                        owner_stats["triples_kept"] += 1

                # Collect fact data (aggregated with provenance)
                for head_id, head_name, relation_type, tail_id, tail_name in processed_triples:
                    # When schema governance is not configured, preserve legacy direction semantics.
                    # When schema is present, only relations declared as direction-sensitive keep direction;
                    # all others are treated as undirected for storage/provenance canonicalization.
                    direction_sensitive = True if domain_schema is None else bool(domain_schema.is_direction_sensitive(relation_type))
                    upsert_fact_occurrence(
                        fact_data_by_id,
                        head_id=head_id,
                        head_name=head_name,
                        relation_type=relation_type,
                        tail_id=tail_id,
                        tail_name=tail_name,
                        chunk_id=chunk.id,
                        owner_id=owner_str,
                        db_owner_id=db_owner_id,
                        schema_version=schema_version,
                        domain=domain,
                        direction_sensitive=direction_sensitive,
                        max_source_chunks=max_source_chunks,
                    )

        # Prepare entity list for batch insertion
        entity_list = list(entity_data.values())
        fact_data = list(fact_data_by_id.values())

        # Phase 2: queryable canonicalization layer (canonical keys + aliases).
        canonical_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        canonical_links: List[Dict[str, Any]] = []
        alias_nodes_by_id: Dict[str, Dict[str, Any]] = {}
        alias_links: List[Dict[str, Any]] = []
        for entity in entity_list:
            owner_key = str(entity.get("owner_id") or "").strip()
            entity_id = str(entity.get("entity_id") or "").strip()
            canonical_key = str(entity.get("entity_canonical_key") or "").strip()
            canonical_name = str(entity.get("entity_canonical_name") or "").strip()
            entity_type_key = str(entity.get("entity_type_key") or "").strip() or "entity"
            alias_text_normalized = str(entity.get("entity_name_normalized") or "").strip()
            alias_text_display = str(entity.get("entity_name") or "").strip()

            if not owner_key or not entity_id or not canonical_key or not canonical_name:
                continue

            canonical_id = compute_mdhash_id(canonical_key, prefix="canonical-", owner_id=owner_key)
            canonical_nodes_by_id.setdefault(
                canonical_id,
                {
                    "canonical_id": canonical_id,
                    "canonical_key": canonical_key,
                    "canonical_name": canonical_name,
                    "entity_type_key": entity_type_key,
                    "owner_id": owner_key,
                },
            )
            canonical_links.append({"entity_id": entity_id, "canonical_id": canonical_id, "owner_id": owner_key})

            # Only create alias nodes when the surface form differs from the canonicalized key.
            if alias_text_normalized and alias_text_normalized != canonical_name:
                alias_id = compute_mdhash_id(f"{alias_text_normalized}|{canonical_id}", prefix="alias-", owner_id=owner_key)
                alias_nodes_by_id.setdefault(
                    alias_id,
                    {
                        "alias_id": alias_id,
                        "alias_text_normalized": alias_text_normalized,
                        "alias_text": alias_text_display or alias_text_normalized,
                        "owner_id": owner_key,
                    },
                )
                alias_links.append({"alias_id": alias_id, "canonical_id": canonical_id, "owner_id": owner_key})

        canonical_nodes = list(canonical_nodes_by_id.values())
        alias_nodes = list(alias_nodes_by_id.values())

        kg_ingest_meta_payloads: List[Dict[str, Any]] = []
        for owner_id, counters in stats_by_owner.items():
            total = int(counters.get("triples_total", 0))
            dropped_endpoints = int(counters.get("triples_dropped_endpoints", 0))
            dropped_ambiguous = int(counters.get("triples_dropped_ambiguous_endpoints", 0))
            drop_ratio = (float(dropped_endpoints) / float(total)) if total else 0.0
            payload = {
                "owner_id": owner_id,
                "triples_total": total,
                "triples_kept": int(counters.get("triples_kept", 0)),
                "triples_dropped_endpoints": dropped_endpoints,
                "triples_dropped_ambiguous_endpoints": dropped_ambiguous,
                "triples_dropped_schema": int(counters.get("triples_dropped_schema", 0)),
                "endpoint_drop_ratio": float(drop_ratio),
                "fact_provenance_max_source_chunks": int(max_source_chunks),
            }
            kg_ingest_meta_payloads.append(payload)
            logger.info(
                "KG ingest stats (owner=%s): total=%s kept=%s dropped_endpoints=%s dropped_schema=%s endpoint_drop_ratio=%.4f",
                owner_id,
                payload["triples_total"],
                payload["triples_kept"],
                payload["triples_dropped_endpoints"],
                payload["triples_dropped_schema"],
                payload["endpoint_drop_ratio"],
            )

        logger.info(
            f"Batch data prepared: {len(chunk_data)} chunks, {len(entity_list)} entities, "
            f"{len(mention_data)} mentions, {len(fact_data)} facts"
        )

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
                        c.source_file_id = chunk.source_file_id,
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
                                  e.entity_name_normalized = entity.entity_name_normalized,
                                  e.entity_canonical_name = entity.entity_canonical_name,
                                  e.entity_canonical_key = entity.entity_canonical_key,
                                  e.entity_type = entity.entity_type,
                                  e.entity_type_key = entity.entity_type_key,
                                  e.node_type = 'entity',
                                  e.attributes = '{}',
                                  e.owner_id = entity.owner_id,
                                  e.created_at = datetime(),
                                  e.updated_at = datetime(),
                                  e.is_new = true
                    ON MATCH SET e.entity_name = entity.entity_name,
                                 e.entity_text = entity.entity_name,
                                 e.entity_name_normalized = entity.entity_name_normalized,
                                 e.entity_canonical_name = entity.entity_canonical_name,
                                 e.entity_canonical_key = entity.entity_canonical_key,
                                 e.entity_type = entity.entity_type,
                                 e.entity_type_key = entity.entity_type_key,
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

                # 2.5 Canonicalization layer: canonical key nodes + canonical/alias relationships.
                if canonical_nodes:
                    canonical_query = """
                    UNWIND $canonicals AS c
                    MERGE (n:EntityCanonical {canonical_id: c.canonical_id})
                    ON CREATE SET n.owner_id = c.owner_id,
                                  n.canonical_key = c.canonical_key,
                                  n.canonical_name = c.canonical_name,
                                  n.entity_type_key = c.entity_type_key,
                                  n.created_at = datetime(),
                                  n.updated_at = datetime()
                    ON MATCH SET  n.owner_id = c.owner_id,
                                  n.canonical_key = c.canonical_key,
                                  n.canonical_name = c.canonical_name,
                                  n.entity_type_key = c.entity_type_key,
                                  n.updated_at = datetime()
                    """
                    tx.run(canonical_query, {"canonicals": canonical_nodes})
                    logger.info("  Upserted %s EntityCanonical nodes", len(canonical_nodes))

                if canonical_links:
                    canonical_rel_query = """
                    UNWIND $links AS link
                    MATCH (e:Entity {entity_id: link.entity_id, owner_id: link.owner_id})
                    MATCH (c:EntityCanonical {canonical_id: link.canonical_id})
                    MERGE (e)-[r:CANONICAL_OF]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(canonical_rel_query, {"links": canonical_links})
                    logger.info("  Upserted %s CANONICAL_OF relationships", len(canonical_links))

                if alias_nodes:
                    alias_query = """
                    UNWIND $aliases AS a
                    MERGE (n:EntityAlias {alias_id: a.alias_id})
                    ON CREATE SET n.owner_id = a.owner_id,
                                  n.alias_text_normalized = a.alias_text_normalized,
                                  n.alias_text = a.alias_text,
                                  n.created_at = datetime(),
                                  n.updated_at = datetime()
                    ON MATCH SET  n.owner_id = a.owner_id,
                                  n.alias_text_normalized = a.alias_text_normalized,
                                  n.alias_text = a.alias_text,
                                  n.updated_at = datetime()
                    """
                    tx.run(alias_query, {"aliases": alias_nodes})
                    logger.info("  Upserted %s EntityAlias nodes", len(alias_nodes))

                if alias_links:
                    alias_rel_query = """
                    UNWIND $links AS link
                    MATCH (a:EntityAlias {alias_id: link.alias_id, owner_id: link.owner_id})
                    MATCH (c:EntityCanonical {canonical_id: link.canonical_id})
                    MERGE (a)-[r:ALIAS_OF]->(c)
                    SET r.owner_id = link.owner_id,
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(alias_rel_query, {"links": alias_links})
                    logger.info("  Upserted %s ALIAS_OF relationships", len(alias_links))

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
                    WITH e1, e2, r, f,
                         CASE
                             WHEN r.source_chunk_ids IS NULL THEN f.source_chunk_ids
                             ELSE r.source_chunk_ids + [cid IN f.source_chunk_ids WHERE NOT cid IN r.source_chunk_ids]
                         END AS merged_chunk_ids
                    SET r.head = f.head_name,
                        r.predicate = f.relation_type,
                        r.tail = f.tail_name,
                        r.text = f.fact_text,
                        r.owner_id = f.owner_id,
                        r.schema_version = f.schema_version,
                        r.domain = f.domain,
                        r.occurrences = COALESCE(r.occurrences, 0) + COALESCE(toInteger(f.occurrences), 1),
                        r.source_chunk_ids = merged_chunk_ids[..$max_source_chunks],
                        r.source_chunk_ids_truncated = COALESCE(r.source_chunk_ids_truncated, false)
                            OR COALESCE(f.source_chunk_ids_truncated, false)
                            OR size(merged_chunk_ids) > $max_source_chunks,
                        r.weight = COALESCE(r.weight, 0.0) + COALESCE(toFloat(f.occurrences), 1.0),
                        r.updated_at = datetime(),
                        r.created_at = COALESCE(r.created_at, datetime())
                    """
                    tx.run(fact_query, {"facts": fact_data, "max_source_chunks": int(max_source_chunks)})
                    logger.info(f"  Batch created {len(fact_data)} RELATES_TO relationships")

                # 5. Persist per-owner ingest stats (avoids process-global state and survives concurrency).
                if kg_ingest_meta_payloads:
                    meta_query = """
                    UNWIND $metas AS meta
                    MERGE (m:KGIngestMeta {owner_id: meta.owner_id})
                    SET m.triples_total = meta.triples_total,
                        m.triples_kept = meta.triples_kept,
                        m.triples_dropped_endpoints = meta.triples_dropped_endpoints,
                        m.triples_dropped_ambiguous_endpoints = meta.triples_dropped_ambiguous_endpoints,
                        m.triples_dropped_schema = meta.triples_dropped_schema,
                        m.endpoint_drop_ratio = meta.endpoint_drop_ratio,
                        m.fact_provenance_max_source_chunks = meta.fact_provenance_max_source_chunks,
                        m.updated_at = datetime(),
                        m.created_at = COALESCE(m.created_at, datetime())
                    """
                    tx.run(meta_query, {"metas": kg_ingest_meta_payloads})
                    logger.info("  Updated %s KGIngestMeta rows", len(kg_ingest_meta_payloads))

                tx.commit()

        elapsed = time.time() - start_time
        logger.info(f"Batch insertion completed in {elapsed:.2f}s")

        return new_entity_ids

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
