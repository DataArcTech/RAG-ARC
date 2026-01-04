import logging
import uuid
from typing import Dict, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGNeo4jPPRMixin:
    def _run_ppr_with_weights(
        self,
        node_weights: dict,
        damping: float = 0.5,
        subgraph_nodes: Set[str] = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run Personalized PageRank with weighted reset probabilities.

        This method supports two backends:
        - 'push': Fast push-based PPR on cached graph (default, recommended)
        - 'igraph': Traditional igraph-based PPR (fallback)

        Args:
            node_weights: Dict mapping node IDs to weights (used as reset probabilities)
            damping: Damping factor for PageRank (probability of random jump)
            subgraph_nodes: Optional set of subgraph node IDs

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores, pagerank_scores_dict)
        """
        if subgraph_nodes is None or len(subgraph_nodes) == 0:
            logger.warning("No subgraph nodes provided")
            return np.array([]), np.array([]), {}

        # Normalize reset probabilities
        reset_sum = sum(node_weights.values())
        if reset_sum == 0:
            logger.warning("All reset probabilities are zero")
            return np.array([]), np.array([]), {}

        normalized_reset = {nid: w / reset_sum for nid, w in node_weights.items()}

        directionality = self._directionality_config()
        directed_mode = str(getattr(self, "ppr_directed_mode", "auto") or "auto").strip().lower()
        # Direction-aware PPR requires relation metadata (predicate) to decide whether to preserve direction.
        # In 'auto' mode we enable it whenever the KG schema declares direction semantics:
        # - blacklist policy: direction-sensitive by default (except explicit undirected list)
        # - whitelist policy: enable only when explicit directed predicates exist
        use_directed = (
            directed_mode == "on"
            or (
                directed_mode == "auto"
                and bool(directionality.get("schema_loaded"))
                and (
                    directionality.get("direction_policy") == "blacklist"
                    or bool(directionality.get("directed_relations"))
                )
            )
        )

        # Choose PPR backend
        if use_directed:
            # Direction-aware PPR requires relation metadata; run via igraph+Neo4j edge extraction.
            effective_policy = str(directionality.get("direction_policy") or "whitelist")
            if directed_mode == "on":
                # Forced-on should preserve direction even when schema is missing/incomplete.
                # Treat predicates as direction-sensitive by default (blacklist with empty undirected set).
                effective_policy = "blacklist"
            pagerank_scores_dict = self._run_ppr_igraph(
                subgraph_nodes,
                normalized_reset,
                damping,
                owner_id=owner_id,
                direction_policy=effective_policy,
                directed_relations=set(directionality.get("directed_relations") or set()),
                direction_insensitive_relations=set(directionality.get("direction_insensitive_relations") or set()),
            )
        elif self.ppr_backend == "push":
            pagerank_scores_dict = self._run_ppr_push(subgraph_nodes, normalized_reset, damping, owner_id=owner_id)
        else:
            pagerank_scores_dict = self._run_ppr_igraph(subgraph_nodes, normalized_reset, damping, owner_id=owner_id)

        if not pagerank_scores_dict:
            logger.warning("PPR returned no scores")
            return np.array([]), np.array([]), {}

        # Extract and sort passage scores
        doc_scores = []
        doc_ids = []
        for i, chunk_id in enumerate(self.passage_node_keys):
            if chunk_id in pagerank_scores_dict:
                doc_scores.append(pagerank_scores_dict[chunk_id])
                doc_ids.append(i)

        doc_scores = np.array(doc_scores)
        doc_ids = np.array(doc_ids)

        sorted_indices = np.argsort(doc_scores)[::-1]
        sorted_doc_ids = doc_ids[sorted_indices]
        sorted_doc_scores = doc_scores[sorted_indices]

        return sorted_doc_ids, sorted_doc_scores, pagerank_scores_dict

    def _run_ppr_push(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        damping: float,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, float]:
        """
        Run PPR using push-based algorithm on cached graph.

        Args:
            subgraph_nodes: Set of node IDs in subgraph
            reset: Normalized reset distribution
            damping: Damping factor (alpha)

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        try:
            ppr_scores = self.graph_store.compute_ppr_push(
                subgraph_nodes=subgraph_nodes,
                reset=reset,
                alpha=damping,
                epsilon=1e-6,
                owner_id=self._owner_to_str(owner_id),
            )
            logger.info(f"Push-based PPR completed: {len(ppr_scores)} nodes with non-zero scores")
            return ppr_scores
        except Exception as e:
            logger.error(f"Push-based PPR failed: {e}, falling back to igraph")
            return self._run_ppr_igraph(subgraph_nodes, reset, damping, owner_id=owner_id)

    def _run_ppr_igraph(
        self,
        subgraph_nodes: Set[str],
        reset: Dict[str, float],
        damping: float,
        owner_id: Optional[uuid.UUID] = None,
        direction_policy: str = "whitelist",
        directed_relations: Optional[Set[str]] = None,
        direction_insensitive_relations: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Run PPR using igraph (fallback method).

        Args:
            subgraph_nodes: Set of node IDs in subgraph
            reset: Normalized reset distribution
            damping: Damping factor

        Returns:
            Dictionary mapping node_id -> PageRank score
        """
        # Extract subgraph:
        # - default path uses the undirected cache for speed
        # - direction-aware PPR extracts edges (with predicates) from Neo4j to preserve direction semantics
        if direction_policy == "blacklist" or (directed_relations and direction_policy == "whitelist"):
            extract_cached = getattr(self.graph_store, "extract_subgraph_from_cache_for_ppr_directed", None)
            if callable(extract_cached):
                graph, _, idx_to_node = extract_cached(
                    subgraph_nodes,
                    owner_id=self._owner_to_str(owner_id),
                    direction_policy=direction_policy,
                    directed_relations=set(directed_relations or set()),
                    direction_insensitive_relations=set(direction_insensitive_relations or set()),
                )
            else:
                graph, _, idx_to_node = self.graph_store.extract_subgraph_from_neo4j_for_ppr(
                    subgraph_nodes,
                    owner_id=self._owner_to_str(owner_id),
                    direction_policy=direction_policy,
                    directed_relations=set(directed_relations or set()),
                    direction_insensitive_relations=set(direction_insensitive_relations or set()),
                )
        else:
            graph, _, idx_to_node = self.graph_store.extract_subgraph_from_cache(
                subgraph_nodes,
                owner_id=self._owner_to_str(owner_id),
            )

        if graph.vcount() == 0:
            logger.warning("Empty subgraph extracted")
            return {}

        logger.info(f"Using igraph PPR: {graph.vcount()} nodes, {graph.ecount()} edges")

        # Build reset probabilities for subgraph nodes
        subgraph_reset = [reset.get(idx_to_node[i], 0.0) for i in range(graph.vcount())]

        try:
            # Run PPR on subgraph
            subgraph_pagerank = graph.personalized_pagerank(
                damping=damping,
                directed=bool(getattr(graph, "is_directed", lambda: False)()),
                weights="weight",
                reset=subgraph_reset,
                implementation="prpack",
            )

            # Map subgraph scores back to node IDs
            pagerank_scores_dict = {}
            for i, score in enumerate(subgraph_pagerank):
                node_id = idx_to_node[i]
                pagerank_scores_dict[node_id] = score

            return pagerank_scores_dict

        except Exception as e:
            logger.error(f"igraph PPR failed: {e}")
            return {}

    def _directionality_config(self) -> dict:
        graph_store = getattr(self, "graph_store", None)
        schema = getattr(graph_store, "kg_schema", None)
        if schema is None:
            return {"schema_loaded": False, "direction_policy": "whitelist", "directed_relations": set(), "direction_insensitive_relations": set()}

        policy = "whitelist"
        policy_getter = getattr(schema, "direction_policy_all", None)
        if callable(policy_getter):
            try:
                policy = str(policy_getter() or "whitelist").strip().lower() or "whitelist"
            except Exception:
                policy = "whitelist"

        directed_relations: Set[str] = set()
        get_directed = getattr(schema, "direction_sensitive_relations_all", None)
        if callable(get_directed):
            try:
                directed_relations = set(get_directed() or set())
            except Exception:
                directed_relations = set()

        direction_insensitive_relations: Set[str] = set()
        get_undirected = getattr(schema, "direction_insensitive_relations_all", None)
        if callable(get_undirected):
            try:
                direction_insensitive_relations = set(get_undirected() or set())
            except Exception:
                direction_insensitive_relations = set()

        return {
            "schema_loaded": True,
            "direction_policy": policy,
            "directed_relations": directed_relations,
            "direction_insensitive_relations": direction_insensitive_relations,
        }
