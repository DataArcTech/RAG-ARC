import logging
import uuid
from collections import defaultdict
from typing import List, Optional, Set, Tuple

import numpy as np

from encapsulation.database.utils.pruned_hipporag_utils import compute_entity_id, normalize_entity_text
from core.prompts.graph_retrieval import FACT_RERANK_RETRY_USER_PROMPT, FACT_RERANK_USER_PROMPT
from core.retrieval.graph_retrieveal.llm_selection import parse_ranked_choice_indices
from core.utils.owner_guard import normalize_owner_id

logger = logging.getLogger(__name__)


class _PrunedHippoRAGFactsMixin:
    def _llm_rerank_chat(self, messages: list[dict], *, enforce_json_object: bool) -> str:
        """
        Call the rerank LLM with best-effort structured output enforcement.

        - For OpenAI-compatible providers/models that support it, use `response_format=json_object`.
        - If unsupported, retry without `response_format` but keep temperature low.
        """
        kwargs: dict = {"temperature": 0.0, "max_tokens": 256}
        if enforce_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            return str(self.llm_client.chat(messages, **kwargs) or "")
        except Exception as exc:  # noqa: BLE001
            if enforce_json_object:
                logger.info("LLM rerank: response_format not supported; retrying without it. error=%s", exc)
                try:
                    kwargs.pop("response_format", None)
                    return str(self.llm_client.chat(messages, **kwargs) or "")
                except Exception:  # noqa: BLE001
                    logger.warning("LLM rerank: retry without response_format failed", exc_info=True)
                    raise
            raise

    def _get_fact_scores_faiss(self, query: str, owner_id: Optional[uuid.UUID] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve relevant facts using FAISS dense retrieval.

        Args:
            query: Query string
            owner_id: Optional owner ID for tenant-aware filtering (unused in base implementation)

        Returns:
            Tuple of (normalized_scores, fact_ids)
        """
        query_embedding = self._get_query_embedding(query)

        try:
            fact_db = None
            get_db = getattr(getattr(self, "graph_store", None), "get_fact_faiss_db", None)
            if callable(get_db):
                fact_db = get_db(owner_id)
            else:
                fact_db = getattr(self.graph_store, "fact_faiss_db", None)
            if fact_db is None:
                logger.warning("Fact FAISS DB is not available")
                return np.array([]), []

            # Check if fact index exists and is initialized
            if fact_db.index is None:
                logger.warning("Fact FAISS index is not initialized")
                return np.array([]), []

            total_facts = fact_db.index.ntotal
            if total_facts == 0:
                logger.warning("No facts in FAISS index")
                return np.array([]), []

            query_vector = query_embedding.reshape(1, -1).astype(np.float32)

            # Normalize query vector for cosine similarity
            if fact_db.config.metric == "cosine" or fact_db.config.normalize_L2:
                from core.utils.faiss_lock import FAISS_LOCK
                import faiss

                with FAISS_LOCK:
                    faiss.normalize_L2(query_vector)

            desired_owner_hits = int(getattr(self.config, "fact_retrieval_top_k", 0) or 0)
            desired_owner_hits = max(1, desired_owner_hits)

            def _search(k: int) -> tuple[np.ndarray, np.ndarray]:
                from core.utils.faiss_lock import FAISS_LOCK

                with FAISS_LOCK:
                    scores_out, indices_out = fact_db.index.search(query_vector, k)
                return scores_out[0], indices_out[0]

            def _collect(scores_1d: np.ndarray, indices_1d: np.ndarray) -> tuple[np.ndarray, list[str]]:
                fact_ids_local: list[str] = []
                valid_scores_local: list[float] = []
                for idx, score in zip(indices_1d, scores_1d):
                    if idx >= 0 and idx in fact_db.index_to_docstore_id:
                        fact_id = fact_db.index_to_docstore_id[idx]
                        if fact_id not in fact_db.deleted_ids:
                            fact_ids_local.append(fact_id)
                            valid_scores_local.append(float(score))
                if not valid_scores_local:
                    return np.array([]), []
                scores_arr = self._min_max_normalize(np.array(valid_scores_local))
                return scores_arr, fact_ids_local

            # Retrieve top-k facts (with buffer for filtering). When owner filtering is enabled,
            # adaptively increase k so the top-k isn't dominated by other tenants' facts.
            k = min(total_facts, max(desired_owner_hits, self.config.fact_retrieval_top_k * 10))
            owner_str = self._owner_to_str(owner_id)
            docstore = getattr(fact_db, "docstore", {})

            while True:
                scores_1d, indices_1d = _search(k)
                query_fact_scores, fact_ids = _collect(scores_1d, indices_1d)

                if owner_id is None or len(query_fact_scores) == 0:
                    return query_fact_scores, fact_ids

                filtered_scores: list[float] = []
                filtered_ids: list[str] = []
                for score, fact_id in zip(query_fact_scores.tolist(), fact_ids):
                    chunk = docstore.get(fact_id)
                    if not chunk:
                        continue
                    fact_owner = getattr(chunk, "owner_id", None)
                    if fact_owner is None and chunk.metadata:
                        fact_owner = chunk.metadata.get("owner_id")
                    if fact_owner is None:
                        continue
                    if str(fact_owner) == owner_str:
                        filtered_scores.append(float(score))
                        filtered_ids.append(fact_id)

                # Keep increasing k until we either have enough owner-scoped hits or we've exhausted the index.
                if len(filtered_ids) >= desired_owner_hits or k >= total_facts:
                    if not filtered_scores:
                        return np.array([]), []
                    dtype = query_fact_scores.dtype if isinstance(query_fact_scores, np.ndarray) else np.float32
                    return np.array(filtered_scores, dtype=dtype), filtered_ids

                k = min(total_facts, max(k * 2, desired_owner_hits))

        except Exception as e:
            logger.error(f"FAISS fact retrieval failed: {e}")
            return np.array([]), []

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query string."""
        embedding = self.embedding_model.embed(query)
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        return embedding

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _get_facts_by_indices(
        self,
        indices: List[int],
        fact_ids: List[str],
        owner_id: Optional[uuid.UUID] = None,
    ) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Retrieve fact triples from database by their indices.

        Args:
            indices: List of indices into fact_ids
            fact_ids: List of fact IDs

        Returns:
            List of fact triples (head, relation, tail)
        """
        facts = []
        cursor = self.graph_store.conn.cursor()

        for idx in indices:
            if idx < len(fact_ids):
                fact_id = fact_ids[idx]
                cursor.execute(
                    "SELECT head, relation, tail, owner_id FROM facts WHERE fact_id = ?",
                    (fact_id,),
                )
                row = cursor.fetchone()
                if row:
                    facts.append((row[0], row[1], row[2], row[3]))

        return facts

    def _extract_entity_ids_from_facts(self, facts: List[Tuple[str, str, str, Optional[str]]]) -> Set[str]:
        """
        Extract unique entity IDs from fact triples.

        Args:
            facts: List of fact triples (head, relation, tail)

        Returns:
            Set of entity IDs appearing in the facts
        """
        entity_ids: Set[str] = set()

        for head_name, _, tail_name, fact_owner in facts:
            owner_scope = normalize_owner_id(fact_owner)
            head_id = compute_entity_id(normalize_entity_text(head_name), owner_id=owner_scope)
            tail_id = compute_entity_id(normalize_entity_text(tail_name), owner_id=owner_scope)

            if head_id in self.graph_store.node_to_idx:
                entity_ids.add(head_id)
            if tail_id in self.graph_store.node_to_idx:
                entity_ids.add(tail_id)

        return entity_ids

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
            Dictionary mapping entity graph indices to relevance scores
        """
        # Collect fact scores for each entity
        entity_to_fact_scores = defaultdict(list)

        for fact_idx, fact in zip(top_k_fact_indices, top_k_facts):
            fact_owner = normalize_owner_id(fact[3])
            head_id = compute_entity_id(normalize_entity_text(fact[0]), owner_id=fact_owner)
            tail_id = compute_entity_id(normalize_entity_text(fact[2]), owner_id=fact_owner)

            fact_score = float(query_fact_scores[fact_idx]) if query_fact_scores.ndim > 0 else float(query_fact_scores)

            entity_to_fact_scores[head_id].append(fact_score)
            entity_to_fact_scores[tail_id].append(fact_score)

        # Compute average relevance score for each entity
        entity_relevance_scores = {}
        node_to_idx = self.graph_store.node_to_idx

        for entity_id in seed_entity_ids:
            if entity_id in entity_to_fact_scores:
                avg_score = float(np.mean(entity_to_fact_scores[entity_id]))

                if entity_id in node_to_idx:
                    entity_idx = node_to_idx[entity_id]
                    entity_relevance_scores[entity_idx] = avg_score

        if entity_relevance_scores:
            logger.info(
                f"[Query-Aware] Entity relevance scores: min={min(entity_relevance_scores.values()):.3f}, "
                f"max={max(entity_relevance_scores.values()):.3f}, "
                f"avg={np.mean(list(entity_relevance_scores.values())):.3f}"
            )

        return entity_relevance_scores

    def _rerank_facts(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        fact_ids: List[str],
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Rerank facts using LLM to improve relevance.

        Args:
            query: Query string
            query_fact_scores: Scores for all retrieved facts
            fact_ids: List of fact IDs

        Returns:
            Tuple of (reranked_facts, reranked_fact_indices)
        """
        link_top_k = self.config.fact_retrieval_top_k

        # Get top-k candidate facts by score
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        candidate_facts = self._get_facts_by_indices(candidate_fact_indices, fact_ids, owner_id=owner_id)

        try:
            # Use LLM to rerank and filter facts
            top_k_facts, top_k_fact_indices = self._llm_rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=self.config.max_facts_after_reranking,
            )
            logger.info(f"LLM reranked {len(candidate_facts)} facts to {len(top_k_facts)}")
            return top_k_facts, top_k_fact_indices
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using top facts by score")
            max_facts = min(self.config.max_facts_after_reranking, len(candidate_facts))
            return candidate_facts[:max_facts], candidate_fact_indices[:max_facts]

    def _llm_rerank_filter(
        self,
        query: str,
        candidate_facts: List[Tuple],
        candidate_fact_indices: List[int],
        len_after_rerank: int = 5,
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Use LLM to select the most relevant facts from candidates.

        Args:
            query: Query string
            candidate_facts: List of candidate fact triples
            candidate_fact_indices: Indices of candidate facts
            len_after_rerank: Number of facts to select

        Returns:
            Tuple of (selected_facts, selected_fact_indices)
        """
        # Format facts for LLM prompt
        facts_text = "\n".join([f"{i+1}. {head} - {relation} - {tail}" for i, (head, relation, tail, *_) in enumerate(candidate_facts)])

        attempts = [
            (FACT_RERANK_USER_PROMPT, True),
            (FACT_RERANK_RETRY_USER_PROMPT, False),
        ]

        last_response = ""
        for template, enforce_json_object in attempts:
            prompt = template.format(query=query, facts_text=facts_text, k=int(len_after_rerank))
            messages = [{"role": "user", "content": prompt}]
            last_response = self._llm_rerank_chat(messages, enforce_json_object=enforce_json_object)

            selected_indices = parse_ranked_choice_indices(
                last_response,
                candidate_count=len(candidate_facts),
                k=int(len_after_rerank),
                one_based=True,
                allow_fallback=False,
            )
            if selected_indices:
                top_k_facts = [candidate_facts[i] for i in selected_indices[:len_after_rerank]]
                top_k_fact_indices = [candidate_fact_indices[i] for i in selected_indices[:len_after_rerank]]
                return top_k_facts, top_k_fact_indices

        logger.warning("Failed to parse LLM rerank response after retries; falling back. response=%r", last_response)
        max_facts = min(len_after_rerank, len(candidate_facts))
        return candidate_facts[:max_facts], candidate_fact_indices[:max_facts]
