"""Graph indexing embedding robustness knobs (centralized).

These env-driven knobs control how graph indexing (Neo4j HippoRAG store) handles
embedding failures while building chunk/entity/fact embeddings.

Why:
- The graph indexer runs in a multi-indexer pipeline (FAISS/BM25/Neo4j). If the
  graph indexer aborts on a single bad chunk, the other indexers may still
  succeed, leaving the graph index partially updated and retrieval unstable.
- We must keep failures observable and avoid silent degradation. The policy
  below allows tolerating a small number of failures while failing fast when a
  large fraction fails (e.g., embedding backend outage).
"""
import logging
import os

logger = logging.getLogger(__name__)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    token = str(raw).strip() if raw is not None else ""
    return token or default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    token = str(raw).strip() if raw is not None else ""
    if not token:
        return float(default)
    try:
        return float(token)
    except ValueError:
        logger.warning("Invalid float for %s (%s); falling back to %s", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    token = str(raw).strip() if raw is not None else ""
    if not token:
        return int(default)
    try:
        return int(token)
    except ValueError:
        logger.warning("Invalid int for %s (%s); falling back to %s", name, raw, default)
        return int(default)


# How to handle per-item embedding failures inside a batch:
# - "raise": fail the whole indexing run (current strict behavior).
# - "zero": keep indexing, but fill failed items with zero vectors (still logged).
GRAPH_INDEX_EMBED_FAILURE_POLICY = _env_str("GRAPH_INDEX_EMBED_FAILURE_POLICY", "zero").lower()

# Safety valves: even in "zero" mode, fail fast when failures are too many.
# (prevents silently producing a mostly-zero embedding index when the backend is down)
GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO = max(0.0, min(1.0, _env_float("GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO", 0.05)))
GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT = max(0, _env_int("GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT", 50))

