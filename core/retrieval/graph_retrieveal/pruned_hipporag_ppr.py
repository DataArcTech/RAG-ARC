import logging
import uuid
from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _PrunedHippoRAGPPRMixin:
    def _run_ppr_with_weights(
        self,
        node_weights: np.ndarray,
        damping: float = 0.5,
        subgraph_nodes: List[int] = None,
        owner_id: Optional[uuid.UUID] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Personalized PageRank with weighted reset probabilities.

        When subgraph_nodes is provided, PPR runs only on the induced subgraph.
        When subgraph_nodes is None, PPR runs on the full graph.

        Args:
            node_weights: Weight for each node (used as reset probabilities)
            damping: Damping factor for PageRank (probability of random jump)
            subgraph_nodes: Optional list of subgraph node indices

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores, pagerank_scores)
        """
        # Clean up weights (remove NaN and negative values)
        node_weights = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)

        read_lock = getattr(self.graph_store, "read_lock", None)
        lock_ctx = self.graph_store.read_lock() if callable(read_lock) else nullcontext()
        with lock_ctx:
            if subgraph_nodes is not None and len(subgraph_nodes) > 0:
                # Use induced subgraph
                subgraph_size = len(subgraph_nodes)
                total_size = len(self.graph_store.node_to_idx)
                logger.info(
                    f"Using induced subgraph PPR: {subgraph_size}/{total_size} nodes ({100*subgraph_size/total_size:.1f}%)"
                )

                subgraph = self.graph_store.graph.induced_subgraph(subgraph_nodes)

                # Extract reset probabilities for subgraph nodes
                subgraph_reset = [node_weights[node_idx] for node_idx in subgraph_nodes]

                # Normalize reset probabilities
                reset_sum = sum(subgraph_reset)
                if reset_sum > 0:
                    subgraph_reset = [r / reset_sum for r in subgraph_reset]
                else:
                    logger.warning("All reset probabilities are zero")
                    return np.array([]), np.array([]), np.array([])

                try:
                    # Run PPR on subgraph
                    subgraph_pagerank = subgraph.personalized_pagerank(
                        damping=damping,
                        directed=False,
                        weights="weight",
                        reset=subgraph_reset,
                        implementation="prpack",
                    )

                    # Map subgraph scores back to full graph
                    pagerank_scores = [0.0] * len(self.graph_store.node_to_idx)
                    for i, node_idx in enumerate(subgraph_nodes):
                        pagerank_scores[node_idx] = subgraph_pagerank[i]

                except Exception as e:
                    logger.error(f"Subgraph PPR failed: {e}")
                    return np.array([]), np.array([]), np.array([])
            else:
                # No subgraph specified, use full graph
                logger.info(f"Using full graph PPR: {len(self.graph_store.node_to_idx)} nodes")

                # Normalize reset probabilities
                reset_sum = node_weights.sum()
                if reset_sum > 0:
                    reset_prob = node_weights / reset_sum
                else:
                    logger.warning("All reset probabilities are zero")
                    return np.array([]), np.array([]), np.array([])

                try:
                    # Run PPR on full graph
                    pagerank_scores = self.graph_store.graph.personalized_pagerank(
                        damping=damping,
                        directed=False,
                        weights="weight",
                        reset=reset_prob.tolist(),
                        implementation="prpack",
                    )

                except Exception as e:
                    logger.error(f"Full graph PPR failed: {e}")
                    return np.array([]), np.array([]), np.array([])

        # Extract and sort passage scores
        pagerank_array = np.array(pagerank_scores)
        doc_scores = pagerank_array[self.passage_node_idxs]

        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]

        return sorted_doc_ids, sorted_doc_scores, pagerank_array

