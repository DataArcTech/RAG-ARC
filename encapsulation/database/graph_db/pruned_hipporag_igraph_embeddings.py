import os
import logging
import re

import numpy as np
import faiss

from encapsulation.data_model.schema import Chunk
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing

logger = logging.getLogger(__name__)


class _PrunedHippoRAGIGraphEmbeddingsMixin:
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
        cursor.execute('SELECT entity_id, entity_name, owner_id FROM entities')
        entities = cursor.fetchall()

        new_entities = []
        for entity_id, entity_name, entity_owner in entities:
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
        cursor.execute('SELECT fact_id, text, owner_id FROM facts')
        facts = cursor.fetchall()

        new_facts = []
        for fact_id, fact_text, fact_owner in facts:
            if fact_id not in self.fact_faiss_db.docstore:
                metadata = {'type': 'fact'}
                if fact_owner:
                    metadata['owner_id'] = fact_owner
                new_facts.append(Chunk(
                    id=fact_id,
                    content=fact_text,
                    owner_id=fact_owner,
                    metadata=metadata
                ))

        if new_facts:
            logger.info(f"Adding {len(new_facts)} facts to FAISS Flat...")
            self.fact_faiss_db.update_index(new_facts)
            fact_index_path = os.path.join(self.storage_path, 'fact_index')
            self.fact_faiss_db.save_index(fact_index_path, 'index')
            logger.info(f"Saved fact index to {fact_index_path}")

        logger.info("Batch embedding generation completed!")

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
        cursor.execute('SELECT entity_id, entity_name, owner_id FROM entities')
        entities = cursor.fetchall()

        if not entities:
            logger.warning("No entities found, skipping synonymy edge addition")
            return

        # Build entity metadata mapping for fast lookup
        entity_id_to_info = {
            eid: {'name': name, 'owner_id': owner}
            for eid, name, owner in entities
        }

        # Build a set to track existing entity-entity edges (fact edges only)
        # We only check entity-entity edges to avoid duplicates, not chunk-entity edges
        existing_entity_entity_edges = set()

        # Add fact edges (entity-entity edges)
        cursor.execute('SELECT head, tail, owner_id FROM facts')
        for head_name, tail_name, fact_owner in cursor.fetchall():
            owner_str = self._normalize_owner_id(fact_owner)
            head_id = compute_mdhash_id(text_processing(head_name), prefix='entity-', owner_id=owner_str)
            tail_id = compute_mdhash_id(text_processing(tail_name), prefix='entity-', owner_id=owner_str)
            existing_entity_entity_edges.add((head_id, tail_id))
            existing_entity_entity_edges.add((tail_id, head_id))

        logger.info(f"Built existing entity-entity edge set with {len(existing_entity_entity_edges)} directional edges")

        num_synonym_edges = 0
        edges_to_add = []  # Batch collect edges for SQLite

        # Pre-extract and normalize all embeddings for batch search
        logger.info("Preparing embeddings for batch FAISS search...")
        valid_entities = []
        embeddings_list = []

        for entity_id, entity_name, entity_owner in entities:
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

            valid_entities.append((entity_id, entity_name, entity_owner))
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

        if len(valid_entities) > 0:
            _, first_entity_name, _ = valid_entities[0]
            first_distances = distances_batch[0]
            first_indices = indices_batch[0]
            logger.info(f"DEBUG: First entity '{first_entity_name}' top-5 neighbors:")
            for j in range(min(5, len(first_distances))):
                if first_indices[j] != -1 and first_indices[j] in self.entity_faiss_db.index_to_docstore_id:
                    neighbor_id = self.entity_faiss_db.index_to_docstore_id[first_indices[j]]
                    neighbor_name = entity_id_to_info.get(neighbor_id, {}).get('name', "Unknown")
                    logger.info(f"  {j+1}. {neighbor_name}: distance={first_distances[j]:.4f}")

        for i, ((entity_id, entity_name, entity_owner), distances, indices) in enumerate(tqdm(
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
                neighbor_info = entity_id_to_info.get(neighbor_entity_id)
                if not neighbor_info:
                    continue

                neighbor_owner = neighbor_info.get('owner_id')
                if neighbor_owner != entity_owner:
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
                    edges_to_add.append((entity_id, neighbor_entity_id, similarity, entity_owner))
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
                'INSERT OR REPLACE INTO synonymy_edges (entity_id_1, entity_id_2, similarity, owner_id) VALUES (?, ?, ?, ?)',
                edges_to_add
            )
            self.conn.commit()

            logger.info(f"Adding {len(edges_to_add)} synonymy edges to graph (batch mode)...")

            # Prepare batch data for igraph
            valid_edges = []
            edge_weights = []

            for entity_id_1, entity_id_2, similarity, _ in edges_to_add:
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


