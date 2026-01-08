import os
import logging
import re
import random
import time
from typing import List, Dict, Any, Optional, Sequence, Set, Tuple

import numpy as np

import faiss

from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jEmbeddingsMixin:
    def _graph_embedding_settings(self) -> Dict[str, float]:
        config_batch_size = getattr(getattr(self.embedding_model, "config", None), "request_batch_size", None)
        batch_size = int(os.getenv("GRAPH_INDEX_EMBED_BATCH_SIZE", str(config_batch_size or 64)))
        max_retries = int(os.getenv("GRAPH_INDEX_EMBED_MAX_RETRIES", "6"))
        backoff_base = float(os.getenv("GRAPH_INDEX_EMBED_BACKOFF_BASE_SECONDS", "1.0"))
        backoff_cap = float(os.getenv("GRAPH_INDEX_EMBED_BACKOFF_MAX_SECONDS", "30.0"))
        sleep_between_batches = float(os.getenv("GRAPH_INDEX_EMBED_SLEEP_BETWEEN_BATCHES_SECONDS", "0.0"))
        return {
            "batch_size": float(max(1, batch_size)),
            "max_retries": float(max(0, max_retries)),
            "backoff_base": float(max(0.0, backoff_base)),
            "backoff_cap": float(max(0.1, backoff_cap)),
            "sleep_between_batches": float(max(0.0, sleep_between_batches)),
        }

    @staticmethod
    def _looks_like_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
            or "ratelimit" in msg
        )

    def _embed_texts_resilient(self, texts: List[str], *, purpose: str) -> List[List[float]]:
        """
        Resilient embedding wrapper for graph indexing.

        - Batches inputs to avoid 429 amplification
        - Retries on rate limit errors with jittered backoff
        - If a batch fails, splits down to per-item to isolate bad inputs / backend quirks
        """
        settings = self._graph_embedding_settings()
        batch_size = int(settings["batch_size"])
        max_retries = int(settings["max_retries"])
        backoff_base = float(settings["backoff_base"])
        backoff_cap = float(settings["backoff_cap"])
        sleep_between_batches = float(settings["sleep_between_batches"])

        if not texts:
            return []

        normalized: List[str] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"{purpose} embedding input must be str (index={idx}, type={type(text).__name__})")
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"{purpose} embedding input cannot be empty (index={idx})")
            normalized.append(stripped.replace("\n", " "))

        embeddings: List[List[float]] = []
        failures: List[Dict[str, Any]] = []

        def _embed_batch(batch: List[str]) -> List[List[float]]:
            out = self.embedding_model.embed(batch)
            if isinstance(out, list) and out and isinstance(out[0], (int, float)):
                return [out]  # type: ignore[list-item]
            return out  # type: ignore[return-value]

        for start in range(0, len(normalized), batch_size):
            batch = normalized[start:start + batch_size]
            attempt = 0
            while True:
                try:
                    batch_embeddings = _embed_batch(batch)
                    if len(batch_embeddings) != len(batch):
                        raise RuntimeError(
                            f"{purpose} embeddings size mismatch: got {len(batch_embeddings)} for {len(batch)} inputs"
                        )
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as exc:  # noqa: BLE001
                    if self._looks_like_rate_limit(exc) and attempt < max_retries:
                        delay = min(backoff_cap, backoff_base * (2 ** attempt)) * (0.8 + 0.4 * random.random())
                        logger.warning(
                            "Graph index embedding rate-limited (%s). Retrying in %.2fs (attempt %s/%s, batch=%s)",
                            purpose,
                            delay,
                            attempt + 1,
                            max_retries,
                            len(batch),
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue

                    # If a batch fails, split to isolate bad inputs / incompatible endpoints.
                    if len(batch) > 1:
                        logger.warning(
                            "Graph index embedding batch failed (%s, batch=%s). Splitting to per-item. error=%s",
                            purpose,
                            len(batch),
                            str(exc),
                        )
                        for item in batch:
                            try:
                                item_embedding = _embed_batch([item])
                                if len(item_embedding) != 1:
                                    raise RuntimeError(
                                        f"{purpose} embeddings size mismatch: got {len(item_embedding)} for 1 input"
                                    )
                                embeddings.extend(item_embedding)
                            except Exception as item_exc:  # noqa: BLE001
                                failures.append({"purpose": purpose, "error": str(item_exc)})
                        break

                    failures.append({"purpose": purpose, "error": str(exc)})
                    break

            if sleep_between_batches > 0:
                time.sleep(sleep_between_batches)

        if failures:
            raise RuntimeError(
                f"Graph index embedding failed ({purpose}): {len(failures)} failures; first_error={failures[0]['error']}"
            )

        return embeddings

    def batch_generate_embeddings(
        self,
        *,
        chunk_ids: Optional[Sequence[str]] = None,
        entity_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch generate embeddings for all facts, entities, and chunks.

        This method processes embeddings in the following order:
        1. Chunk embeddings: Generated and stored in memory (not in FAISS)
        2. Entity embeddings: Generated and added to FAISS HNSW index
        3. Fact embeddings: Generated and added to FAISS Flat index

        Only new items (not already in FAISS/memory) are processed.
        """
        logger.info("Batch generating embeddings...")
        summary: Dict[str, Any] = {"chunks": {}, "entities": {}, "facts": {}}

        # 1. Generate chunk embeddings
        chunk_params: Dict[str, Any] = {}
        if chunk_ids:
            chunk_query = "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids RETURN c.chunk_id AS chunk_id, c.content AS content"
            chunk_params = {"chunk_ids": list(chunk_ids)}
        else:
            chunk_query = "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, c.content AS content"
        chunks_data = self._execute_query(chunk_query, chunk_params or None)

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
            chunk_embeddings = self._embed_texts_resilient(new_chunks, purpose="chunk")

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
            summary["chunks"] = {"attempted": len(new_chunks), "embedded": len(new_chunk_ids)}
        else:
            summary["chunks"] = {"attempted": 0, "embedded": 0}

        # 2. Generate entity embeddings and add to FAISS HNSW
        entity_params: Dict[str, Any] = {}
        if entity_ids:
            entity_query = (
                "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids "
                "RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
            )
            entity_params = {"entity_ids": list(entity_ids)}
        else:
            entity_query = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
        entities = self._execute_query(entity_query, entity_params or None)

        new_entities = []
        for record in entities:
            entity_id = record['entity_id']
            entity_name = record['entity_name']
            entity_owner = self._restore_owner_id(record.get('owner_id'))
            # Check if already in FAISS
            if entity_id not in self.entity_faiss_db.docstore:
                metadata = {'type': 'entity'}
                if entity_owner:
                    metadata['owner_id'] = entity_owner
                new_entities.append(Chunk(
                    id=entity_id,
                    content=entity_name,
                    owner_id=entity_owner,
                    metadata=metadata
                ))

        if new_entities:
            logger.info(f"Adding {len(new_entities)} entities to FAISS HNSW...")
            entity_texts = [chunk.content for chunk in new_entities]
            entity_embeddings = self._embed_texts_resilient(entity_texts, purpose="entity")

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
            summary["entities"] = {"attempted": len(new_entities), "embedded": len(new_entities)}
        else:
            summary["entities"] = {"attempted": 0, "embedded": 0}

        # 3. Generate fact embeddings and add to FAISS Flat
        # Facts are stored as RELATES_TO relationships between entities
        fact_query = """
        MATCH (h:Entity)-[r:RELATES_TO]->(t:Entity)
        RETURN r.fact_id AS fact_id,
               r.text AS text,
               r.predicate AS predicate,
               r.owner_id AS owner_id,
               r.source_chunk_ids AS source_chunk_ids,
               r.source_chunk_ids_truncated AS source_chunk_ids_truncated,
               h.entity_id AS head_id,
               t.entity_id AS tail_id,
               h.entity_name AS head_name,
               t.entity_name AS tail_name,
               h.entity_type AS head_type,
               t.entity_type AS tail_type
        """
        facts = self._execute_query(fact_query)

        new_facts = []
        updated_fact_meta = 0
        for record in facts:
            fact_id = record['fact_id']
            fact_text = record['text']
            fact_owner = self._restore_owner_id(record.get('owner_id'))
            if fact_id not in self.fact_faiss_db.docstore:
                metadata = {'type': 'fact'}
                if fact_owner:
                    metadata['owner_id'] = fact_owner
                # Preserve structured endpoints for downstream retrieval (avoids re-parsing text and
                # allows representing same-name different-type entities).
                for key in (
                    "head_id",
                    "tail_id",
                    "head_name",
                    "tail_name",
                    "head_type",
                    "tail_type",
                    "predicate",
                ):
                    if record.get(key) is not None:
                        metadata[key] = record.get(key)
                from encapsulation.database.utils.fact_provenance import merge_provenance_into_fact_metadata

                merge_provenance_into_fact_metadata(
                    metadata,
                    source_chunk_ids=record.get("source_chunk_ids"),
                    source_chunk_ids_truncated=record.get("source_chunk_ids_truncated"),
                )
                new_facts.append(Chunk(
                    id=fact_id,
                    content=fact_text,
                    owner_id=fact_owner,
                    metadata=metadata
                ))
            else:
                chunk = self.fact_faiss_db.docstore.get(fact_id)
                if not chunk:
                    continue
                meta = getattr(chunk, "metadata", None)
                if not isinstance(meta, dict):
                    continue
                from encapsulation.database.utils.fact_provenance import merge_provenance_into_fact_metadata

                if merge_provenance_into_fact_metadata(
                    meta,
                    source_chunk_ids=record.get("source_chunk_ids"),
                    source_chunk_ids_truncated=record.get("source_chunk_ids_truncated"),
                ):
                    updated_fact_meta += 1

        if new_facts:
            logger.info(f"Adding {len(new_facts)} facts to FAISS Flat...")
            fact_texts = [chunk.content for chunk in new_facts]
            fact_embeddings = self._embed_texts_resilient(fact_texts, purpose="fact")
            for chunk, embedding in zip(new_facts, fact_embeddings):
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                chunk.metadata["embedding"] = embedding
            self.fact_faiss_db.update_index(new_facts)
            fact_index_path = os.path.join(self.storage_path, 'fact_index')
            self.fact_faiss_db.save_index(fact_index_path, 'index')
            logger.info(f"Saved fact index to {fact_index_path}")
            summary["facts"] = {"attempted": len(new_facts), "embedded": len(new_facts)}
        else:
            summary["facts"] = {"attempted": 0, "embedded": 0}
            fact_index_path = os.path.join(self.storage_path, 'fact_index')
            if updated_fact_meta > 0:
                self.fact_faiss_db.save_index(fact_index_path, 'index')
                logger.info("Backfilled fact provenance for %s existing facts; saved index to %s", updated_fact_meta, fact_index_path)

        logger.info("Batch embedding generation completed!")
        return summary

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
            RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id
            """
            entities = self._execute_query(entity_query, {'entity_ids': new_entity_ids})
        else:
            logger.info("Computing synonymy edges for all entities (full rebuild)...")
            # Get all entities
            entity_query = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.entity_name AS entity_name, e.owner_id AS owner_id"
            entities = self._execute_query(entity_query)

        if not entities:
            logger.warning("No entities found, skipping synonymy edge addition")
            return

        # Build entity metadata mapping for fast lookup
        entity_id_to_info = {
            record['entity_id']: {
                'name': record['entity_name'],
                'owner_id': self._restore_owner_id(record.get('owner_id'))
            }
            for record in entities
        }

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
            entity_owner = entity_id_to_info.get(entity_id, {}).get('owner_id')
            owner_key = self._owner_key(entity_owner)

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

            valid_entities.append((entity_id, entity_name, entity_owner, owner_key))
            embeddings_list.append(embedding)

        if not valid_entities:
            logger.warning("No valid entities for synonymy edge computation")
            return

        # Batch normalize embeddings
        embeddings_array = np.array(embeddings_list).astype(np.float32)
        if self.entity_faiss_db.config.normalize_L2 or self.entity_faiss_db.config.metric == "cosine":
            from core.utils.faiss_lock import FAISS_LOCK
            with FAISS_LOCK:
                faiss.normalize_L2(embeddings_array)

        logger.info(f"Prepared {len(valid_entities)} valid entities for synonymy edge computation")

        # Batch FAISS search
        logger.info("Performing batch FAISS search...")
        k = min(self.synonymy_edge_topk, self.entity_faiss_db.index.ntotal)
        from core.utils.faiss_lock import FAISS_LOCK
        with FAISS_LOCK:
            distances_batch, indices_batch = self.entity_faiss_db.index.search(embeddings_array, k)
        logger.info("Batch FAISS search completed")

        # Process results
        logger.info("Processing search results...")

        for i, ((entity_id, entity_name, entity_owner, owner_key), distances, indices) in enumerate(tqdm(
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
                neighbor_info = entity_id_to_info.get(neighbor_entity_id)
                if not neighbor_info:
                    continue

                neighbor_owner = neighbor_info.get('owner_id')
                if neighbor_owner != entity_owner:
                    continue

                neighbor_name = neighbor_info['name']

                # FAISS with metric='cosine' returns NEGATIVE inner product
                similarity = -float(distance)

                # Check threshold
                if similarity < self.synonymy_edge_sim_threshold:
                    break  # Distances are sorted, can break early

                edge_key = (entity_id, neighbor_entity_id)
                reverse_edge_key = (neighbor_entity_id, entity_id)

                if edge_key not in existing_entity_entity_edges and reverse_edge_key not in existing_entity_entity_edges:
                    # Add UNIDIRECTIONAL edge (only one direction to avoid duplication)
                    edges_to_add.append((entity_id, neighbor_entity_id, similarity, owner_key))
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
                r.owner_id = edge.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """

            # Prepare batch data
            batch_data = [
                {
                    'entity_id_1': e1,
                    'entity_id_2': e2,
                    'similarity': sim,
                    'owner_id': owner_id
                }
                for e1, e2, sim, owner_id in edges_to_add
            ]

            self._execute_query(batch_query, {'edges': batch_data})

            logger.info(f"Added {num_synonym_edges} unique synonymy edges ({len(edges_to_add)} directional edges)")
        else:
            logger.info("No synonymy edges to add")
