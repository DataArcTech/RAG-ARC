import logging
import math
from typing import Any, Dict, List

import numpy as np

from core.retrieval.graph_retrieveal.constants import DEFAULT_PPR_RELATION_WEIGHTS
from core.retrieval.graph_retrieveal.models import PPRResult, SubgraphResult

logger = logging.getLogger(__name__)


def compute_personalized_pagerank(
    *,
    config: Any,
    subgraph: SubgraphResult,
    entity_candidates: List[Dict[str, Any]],
    relation_weights: Dict[str, float] | None = None,
) -> PPRResult:
    """Compute Personalized PageRank for entities in the given subgraph."""
    if not subgraph.nodes:
        return PPRResult()

    weights = relation_weights or DEFAULT_PPR_RELATION_WEIGHTS

    entity_ids = list(subgraph.nodes.keys())
    n = len(entity_ids)
    id_to_idx = {entity_id: i for i, entity_id in enumerate(entity_ids)}

    adj_matrix = np.zeros((n, n))
    for from_id, relation_type, to_id in subgraph.edges:
        if from_id not in id_to_idx or to_id not in id_to_idx:
            continue
        i, j = id_to_idx[from_id], id_to_idx[to_id]
        weight = float(weights.get(str(relation_type), 1.0))
        adj_matrix[i][j] = weight
        adj_matrix[j][i] = weight

    transition_matrix = np.zeros((n, n))
    for i in range(n):
        row_sum = float(np.sum(adj_matrix[i]))
        if row_sum > 0:
            transition_matrix[i] = adj_matrix[i] / row_sum
        else:
            transition_matrix[i] = np.ones(n) / n

    personalization_vector = _compute_personalization_vector(
        config=config,
        entity_ids=entity_ids,
        entity_candidates=entity_candidates,
        subgraph=subgraph,
    )
    scores_vec = _run_ppr_algorithm(
        config=config,
        transition_matrix=transition_matrix,
        personalization_vector=personalization_vector,
    )

    scores = {entity_id: float(scores_vec[i]) for i, entity_id in enumerate(entity_ids)}
    max_score = max(scores.values()) if scores else 1.0
    normalized_scores = {entity_id: (score / max_score if max_score else 0.0) for entity_id, score in scores.items()}
    return PPRResult(scores=scores, normalized_scores=normalized_scores)


def _compute_personalization_vector(
    *,
    config: Any,
    entity_ids: List[str],
    entity_candidates: List[Dict[str, Any]],
    subgraph: SubgraphResult,
) -> np.ndarray:
    n = len(entity_ids)
    personalization = np.zeros(n)
    candidate_similarities = {str(c.get("entity_id")): float(c.get("similarity") or 0.0) for c in entity_candidates}

    total_score = 0.0
    for i, entity_id in enumerate(entity_ids):
        sim_e = candidate_similarities.get(entity_id, 0.0)
        triple_boost = _compute_triple_boost(entity_id=entity_id, subgraph=subgraph)
        score = float(config.beta1) * sim_e + float(config.beta2) * triple_boost
        personalization[i] = score
        total_score += score

    if total_score > 0:
        return personalization / total_score
    return np.ones(n) / n


def _compute_triple_boost(*, entity_id: str, subgraph: SubgraphResult) -> float:
    relation_count = 0
    for from_id, _, to_id in subgraph.edges:
        if from_id == entity_id or to_id == entity_id:
            relation_count += 1
    return math.log(relation_count + 1)


def _run_ppr_algorithm(*, config: Any, transition_matrix: np.ndarray, personalization_vector: np.ndarray) -> np.ndarray:
    ppr_scores = personalization_vector.copy()
    damping = float(config.damping_factor)
    tolerance = float(config.tolerance)
    max_iterations = int(config.max_iterations)

    for iteration in range(max_iterations):
        old_scores = ppr_scores.copy()
        ppr_scores = (1 - damping) * personalization_vector + damping * np.dot(transition_matrix.T, ppr_scores)
        diff = float(np.linalg.norm(ppr_scores - old_scores))
        if diff < tolerance:
            logger.info("PPR converged after %s iterations", iteration + 1)
            break

    return ppr_scores


__all__ = ["compute_personalized_pagerank"]

