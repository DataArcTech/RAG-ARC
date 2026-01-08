import logging
import uuid
from typing import List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jFactsMixin:
    def _get_fact_scores_faiss(
        self,
        query: str,
        owner_id: Optional[uuid.UUID] = None,
        *,
        query_doc_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        scores, fact_ids = super()._get_fact_scores_faiss(query, owner_id=owner_id)

        if owner_id is None or scores is None or len(scores) == 0:
            return scores, fact_ids

        owner_str = self._owner_to_str(owner_id)
        docstore = self.graph_store.fact_faiss_db.docstore

        filtered_scores = []
        filtered_ids = []

        for idx, fact_id in enumerate(fact_ids):
            chunk = docstore.get(fact_id)
            if not chunk:
                continue

            fact_owner = getattr(chunk, "owner_id", None)
            if fact_owner is None and chunk.metadata:
                fact_owner = chunk.metadata.get("owner_id")

            if fact_owner is None:
                continue

            if str(fact_owner) == owner_str:
                score_value = scores[idx] if isinstance(scores, np.ndarray) else scores[idx]
                filtered_scores.append(float(score_value))
                filtered_ids.append(fact_id)

        if not filtered_scores:
            return np.array([]), []

        dtype = scores.dtype if isinstance(scores, np.ndarray) else np.float32
        filtered_scores_arr = np.array(filtered_scores, dtype=dtype)

        cfg_obj = getattr(self, "config", None)
        if cfg_obj is None:
            return filtered_scores_arr, filtered_ids

        if not bool(getattr(cfg_obj, "fact_groundability_enabled", False)):
            return filtered_scores_arr, filtered_ids

        from core.retrieval.graph_retrieveal.fact_groundability import FactGroundabilityConfig, apply_groundability

        dense_top_k = int(getattr(cfg_obj, "fact_groundability_dense_top_k"))
        dense_top_k = max(1, dense_top_k)
        dense_scores = query_doc_scores if query_doc_scores is not None else self._dense_passage_retrieval_scores(query)
        top_indices = np.argsort(dense_scores)[-dense_top_k:][::-1]
        dense_top_chunk_ids = {
            self.passage_node_keys[i] for i in top_indices if 0 <= int(i) < len(self.passage_node_keys)
        }

        cfg = FactGroundabilityConfig(
            enabled=True,
            mode=str(getattr(cfg_obj, "fact_groundability_mode")),
            dense_top_k=dense_top_k,
            min_overlap_count=int(getattr(cfg_obj, "fact_groundability_min_overlap_count")),
            min_overlap_ratio=float(getattr(cfg_obj, "fact_groundability_min_overlap_ratio")),
            soft_min_weight=float(getattr(cfg_obj, "fact_groundability_soft_min_weight")),
            soft_gamma=float(getattr(cfg_obj, "fact_groundability_soft_gamma")),
            keep_missing_provenance=bool(getattr(cfg_obj, "fact_groundability_keep_missing_provenance")),
            missing_provenance_weight=float(getattr(cfg_obj, "fact_groundability_missing_provenance_weight")),
        )
        new_scores, new_fact_ids, meta = apply_groundability(
            cfg=cfg,
            scores=filtered_scores_arr,
            fact_ids=filtered_ids,
            docstore=self.graph_store.fact_faiss_db.docstore,
            dense_top_chunk_ids=dense_top_chunk_ids,
        )
        logger.info(
            "Fact groundability applied=%s mode=%s kept=%s dropped=%s missing=%s",
            meta.get("applied"),
            meta.get("mode"),
            meta.get("kept"),
            meta.get("dropped"),
            meta.get("missing_provenance"),
        )
        return new_scores, new_fact_ids

    def _get_facts_by_indices(
        self,
        indices: List[int],
        fact_ids: List[str],
        owner_id: Optional[uuid.UUID] = None,
    ) -> List[Tuple]:
        """
        Retrieve fact triples from FAISS docstore (no database query needed).

        Args:
            indices: List of indices into fact_ids
            fact_ids: List of fact IDs

        Returns:
            List of fact tuples:
            - Legacy: (head, relation, tail, owner)
            - Extended (when available): (head, relation, tail, owner, head_id, tail_id, head_type, tail_type)
        """
        import ast

        facts: List[Tuple] = []
        owner_str = self._owner_to_str(owner_id)

        for idx in indices:
            if idx < len(fact_ids):
                fact_id = fact_ids[idx]
                # Retrieve fact from FAISS docstore (contains full Chunk with fact content)
                if fact_id in self.graph_store.fact_faiss_db.docstore:
                    chunk = self.graph_store.fact_faiss_db.docstore[fact_id]

                    fact_owner = getattr(chunk, "owner_id", None)
                    if fact_owner is None and chunk.metadata:
                        fact_owner = chunk.metadata.get("owner_id")

                    if owner_str and str(fact_owner) != owner_str:
                        continue

                    meta = chunk.metadata if isinstance(getattr(chunk, "metadata", None), dict) else {}
                    head_id = meta.get("head_id")
                    tail_id = meta.get("tail_id")
                    head_type = meta.get("head_type")
                    tail_type = meta.get("tail_type")

                    predicate_meta = meta.get("predicate")
                    head_meta = meta.get("head_name")
                    tail_meta = meta.get("tail_name")
                    if head_meta and predicate_meta and tail_meta:
                        facts.append((str(head_meta), str(predicate_meta), str(tail_meta), fact_owner, head_id, tail_id, head_type, tail_type))
                        continue

                    fact_content = chunk.content
                    parsed_fact: Optional[Tuple[str, str, str]] = None

                    if isinstance(fact_content, tuple) and len(fact_content) == 3:
                        parsed_fact = (fact_content[0], fact_content[1], fact_content[2])
                    elif isinstance(fact_content, str):
                        # Try to parse as Python literal (tuple string representation)
                        try:
                            parsed = ast.literal_eval(fact_content)
                            if isinstance(parsed, tuple) and len(parsed) == 3:
                                parsed_fact = (parsed[0], parsed[1], parsed[2])
                            else:
                                parts = fact_content.split(" | ")
                                if len(parts) == 3:
                                    parsed_fact = (parts[0], parts[1], parts[2])
                        except Exception:
                            parts = fact_content.split(" | ")
                            if len(parts) == 3:
                                parsed_fact = (parts[0], parts[1], parts[2])
                    elif isinstance(fact_content, (list, np.ndarray)) and len(fact_content) >= 3:
                        parsed_fact = (fact_content[0], fact_content[1], fact_content[2])

                    if parsed_fact:
                        facts.append((parsed_fact[0], parsed_fact[1], parsed_fact[2], fact_owner, head_id, tail_id, head_type, tail_type))

        return facts

    def _extract_entity_ids_from_facts(self, facts: List[Tuple]) -> Set[str]:
        """
        Extract unique entity IDs from fact triples using Neo4j.

        Args:
            facts: List of fact triples (head, relation, tail)

        Returns:
            Set of entity IDs appearing in the facts
        """
        entity_ids = set()

        for fact in facts:
            head_id = fact[4] if len(fact) > 4 else None
            tail_id = fact[5] if len(fact) > 5 else None
            if head_id and tail_id:
                entity_ids.add(str(head_id))
                entity_ids.add(str(tail_id))
                continue

            # Backward-compatible fallback for older fact payloads that do not carry endpoint IDs.
            from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id
            from core.utils.text_processing import text_processing

            head_name = fact[0]
            tail_name = fact[2]
            fact_owner = fact[3] if len(fact) > 3 else None
            owner_str = self._owner_to_str(fact_owner)
            head_normalized = text_processing(head_name)
            tail_normalized = text_processing(tail_name)
            entity_ids.add(compute_mdhash_id(head_normalized, prefix="entity-", owner_id=owner_str))
            entity_ids.add(compute_mdhash_id(tail_normalized, prefix="entity-", owner_id=owner_str))

        return entity_ids

    def _dense_passage_retrieval_scores(self, query: str) -> np.ndarray:
        """
        Compute dense retrieval scores for all passages using dot product similarity.

        Optimized version using pre-computed passage embeddings array.
        Supports float16 embeddings and normalized embeddings for cosine similarity.

        Args:
            query: Query string

        Returns:
            Array of similarity scores for all passages
        """
        query_embedding = self._get_query_embedding(query)

        # Normalize query embedding if chunk embeddings are normalized
        if self.graph_store.normalize_chunk_embeddings:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        if self.passage_embeddings_array.size == 0:
            logger.warning("No passage embeddings available")
            return np.zeros(len(self.passage_node_keys))

        # Fast dot product using numpy
        # If embeddings are float16, convert query to float16 for consistency
        if self.passage_embeddings_array.dtype == np.float16:
            query_embedding = query_embedding.astype(np.float16)

        doc_scores = np.dot(self.passage_embeddings_array, query_embedding)

        # Convert back to float32 for downstream processing
        if doc_scores.dtype == np.float16:
            doc_scores = doc_scores.astype(np.float32)

        return doc_scores

    def _compute_entity_relevance_scores(
        self,
        seed_entity_ids: Set[str],
        top_k_facts: List[Tuple],
        query_fact_scores: np.ndarray,
        top_k_fact_indices: List[int],
        owner_id: Optional[uuid.UUID] = None,
    ) -> dict:
        """
        Compute relevance scores for entities based on their associated fact scores.

        This is used for query-aware pruning: entities with higher relevance scores
        will have more neighbors retained during graph expansion.

        Args:
            seed_entity_ids: Set of seed entity IDs
            top_k_facts: List of top-k fact triples
            query_fact_scores: Scores for all retrieved facts
            top_k_fact_indices: Indices of top-k facts

        Returns:
            Dictionary mapping entity IDs to relevance scores (not graph indices)
        """
        from collections import defaultdict

        # Collect fact scores for each entity
        entity_to_fact_scores = defaultdict(list)

        for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
            head_id = fact[4] if len(fact) > 4 else None
            tail_id = fact[5] if len(fact) > 5 else None
            if not head_id or not tail_id:
                # Backward-compatible fallback: older fact payloads do not have endpoint IDs.
                from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id
                from core.utils.text_processing import text_processing

                fact_owner = self._owner_to_str(fact[3] if len(fact) > 3 else None)
                head_name = text_processing(fact[0])
                tail_name = text_processing(fact[2])
                head_id = compute_mdhash_id(head_name, prefix="entity-", owner_id=fact_owner)
                tail_id = compute_mdhash_id(tail_name, prefix="entity-", owner_id=fact_owner)

            fact_score = float(query_fact_scores[fact_idx]) if query_fact_scores.ndim > 0 else float(query_fact_scores)

            entity_to_fact_scores[head_id].append(fact_score)
            entity_to_fact_scores[tail_id].append(fact_score)

        # Compute average relevance score for each entity
        entity_relevance_scores = {}

        for entity_id in seed_entity_ids:
            if entity_id in entity_to_fact_scores:
                avg_score = float(np.mean(entity_to_fact_scores[entity_id]))
                # In Neo4j version, we use entity_id directly instead of graph index
                entity_relevance_scores[entity_id] = avg_score

        if entity_relevance_scores:
            logger.info(
                f"[Query-Aware] Entity relevance scores: min={min(entity_relevance_scores.values()):.3f}, "
                f"max={max(entity_relevance_scores.values()):.3f}, "
                f"avg={np.mean(list(entity_relevance_scores.values())):.3f}"
            )

        return entity_relevance_scores
