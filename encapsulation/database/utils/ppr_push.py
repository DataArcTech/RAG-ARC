"""
Push-based Personalized PageRank implementation.

This module provides a fast, memory-efficient implementation of Personalized PageRank
using the push-based algorithm (also known as forward push or residual push).

The push-based algorithm is particularly efficient for:
- Small to medium-sized subgraphs (after pruning)
- Sparse reset distributions (few seed nodes)
- Approximate PPR with controllable error bounds

Compared to igraph's implementation:
- Lower memory overhead (no need to construct igraph object)
- Faster for small subgraphs (direct computation on adjacency list)
- More predictable latency (no external library overhead)
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional

logger = logging.getLogger(__name__)


def ppr_push(
    adjacency: Dict[str, List[Tuple[str, float]]],
    reset: Dict[str, float],
    alpha: float = 0.5,
    epsilon: float = 1e-6,
    max_iterations: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute Personalized PageRank using push-based algorithm.
    
    The algorithm maintains two values for each node:
    - p[v]: current PageRank estimate
    - r[v]: residual (mass to be pushed to neighbors)
    
    At each step, we select a node u with r[u] > epsilon and push its residual
    to its neighbors. The algorithm terminates when all residuals are below epsilon.
    
    Args:
        adjacency: Graph adjacency list. Dict mapping node_id -> [(neighbor_id, weight), ...]
                   Weights represent edge strengths (e.g., co-occurrence counts, similarity).
        reset: Reset distribution. Dict mapping node_id -> reset_probability.
               Should be normalized (sum to 1.0).
        alpha: Damping factor (teleport probability). Default 0.5.
               - Higher alpha = more exploration (more weight on graph structure)
               - Lower alpha = more exploitation (more weight on reset distribution)
        epsilon: Convergence threshold. Nodes with residual < epsilon are not pushed.
                 Default 1e-6. Smaller values = more accurate but slower.
        max_iterations: Maximum number of push operations. None = no limit.
                        Used to prevent infinite loops on degenerate graphs.
    
    Returns:
        Dictionary mapping node_id -> PageRank score.
        Only nodes with non-zero scores are included.
    
    Algorithm:
        1. Initialize: r[v] = reset[v] for all v, p[v] = 0
        2. While exists u with r[u] > epsilon:
            a. p[u] += (1 - alpha) * r[u]
            b. For each neighbor v of u:
                r[v] += alpha * r[u] * w(u,v) / sum_w(u)
            c. r[u] = 0
        3. Return p
    
    Time complexity: O(|E| / epsilon) in worst case, but typically much faster
    Space complexity: O(|V|) for p and r dictionaries
    """
    # Initialize PageRank estimates and residuals
    p = defaultdict(float)  # PageRank estimates
    r = defaultdict(float)  # Residuals
    
    # Initialize residuals with reset distribution
    for node_id, reset_prob in reset.items():
        r[node_id] = reset_prob
    
    # Queue of nodes with residual > epsilon
    # Use deque for efficient FIFO operations
    queue = deque([node_id for node_id, res in r.items() if res > epsilon])
    
    iterations = 0
    
    while queue:
        # Check max iterations
        if max_iterations is not None and iterations >= max_iterations:
            logger.warning(f"PPR push reached max iterations ({max_iterations}), stopping early")
            break
        
        # Pop node with residual to push
        u = queue.popleft()
        
        # Skip if residual already below threshold (may have been updated)
        if r[u] <= epsilon:
            continue
        
        # Get current residual
        ru = r[u]
        
        # Add to PageRank estimate (keep (1-alpha) of residual)
        p[u] += (1 - alpha) * ru
        
        # Clear residual
        r[u] = 0.0
        
        # Get neighbors
        neighbors = adjacency.get(u, [])
        
        if not neighbors:
            # No neighbors, residual is absorbed
            continue
        
        # Compute weighted degree (sum of edge weights)
        weighted_degree = sum(w for _, w in neighbors)
        
        if weighted_degree == 0:
            # All edge weights are zero, skip
            continue
        
        # Push residual to neighbors (distribute alpha * ru proportionally)
        mass_to_push = alpha * ru
        
        for v, edge_weight in neighbors:
            # Proportion of mass to push to this neighbor
            push_amount = mass_to_push * (edge_weight / weighted_degree)
            
            # Update neighbor's residual
            r[v] += push_amount
            
            # Add to queue if residual exceeds threshold
            if r[v] > epsilon and v not in queue:
                queue.append(v)
        
        iterations += 1
    
    # Convert defaultdict to regular dict and filter out zeros
    result = {node_id: score for node_id, score in p.items() if score > 0}
    
    logger.debug(f"PPR push completed in {iterations} iterations, {len(result)} nodes with non-zero scores")
    
    return result


def ppr_push_weighted(
    adjacency: Dict[str, List[Tuple[str, float]]],
    reset: Dict[str, float],
    alpha: float = 0.5,
    epsilon: float = 1e-6,
    max_iterations: Optional[int] = None,
    normalize_by_degree: bool = True
) -> Dict[str, float]:
    """
    Compute Personalized PageRank with optional degree normalization.
    
    This is a wrapper around ppr_push that optionally normalizes edge weights
    by node degree before running PPR. This can help balance the influence of
    high-degree and low-degree nodes.
    
    Args:
        adjacency: Graph adjacency list
        reset: Reset distribution (should sum to 1.0)
        alpha: Damping factor
        epsilon: Convergence threshold
        max_iterations: Maximum iterations
        normalize_by_degree: If True, normalize edge weights by source node degree
    
    Returns:
        Dictionary mapping node_id -> PageRank score
    """
    if not normalize_by_degree:
        return ppr_push(adjacency, reset, alpha, epsilon, max_iterations)
    
    # Normalize edge weights by degree
    normalized_adj = {}
    for u, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree > 0:
            normalized_adj[u] = [(v, w / degree) for v, w in neighbors]
        else:
            normalized_adj[u] = neighbors
    
    return ppr_push(normalized_adj, reset, alpha, epsilon, max_iterations)


def extract_subgraph_adjacency(
    full_adjacency: Dict[str, List[Tuple[str, float]]],
    subgraph_nodes: Set[str]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract a subgraph from full adjacency list.
    
    Args:
        full_adjacency: Full graph adjacency list
        subgraph_nodes: Set of node IDs to include in subgraph
    
    Returns:
        Subgraph adjacency list (only edges within subgraph_nodes)
    """
    subgraph_adj = {}
    
    for u in subgraph_nodes:
        if u not in full_adjacency:
            continue
        
        # Filter neighbors to only include nodes in subgraph
        subgraph_neighbors = [
            (v, w) for v, w in full_adjacency[u]
            if v in subgraph_nodes
        ]
        
        if subgraph_neighbors:
            subgraph_adj[u] = subgraph_neighbors
    
    return subgraph_adj

