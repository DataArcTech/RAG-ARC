from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class CandidateResult:
    """Candidate recall result."""

    entity_candidates: List[Dict[str, Any]] = field(default_factory=list)  # [(entity_id, entity_name, similarity)]
    chunk_candidates: List[Dict[str, Any]] = field(default_factory=list)  # [(chunk_id, content, similarity)]


@dataclass
class SubgraphResult:
    """Subgraph construction result."""

    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # entity_id -> entity_data
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # [(from_id, relation_type, to_id)]
    seed_entities: List[str] = field(default_factory=list)


@dataclass
class PPRResult:
    """Personalized PageRank result."""

    scores: Dict[str, float] = field(default_factory=dict)  # entity_id -> ppr_score
    normalized_scores: Dict[str, float] = field(default_factory=dict)  # entity_id -> normalized_score


@dataclass
class ChunkScore:
    """Chunk scoring result."""

    chunk_id: str
    content: str
    graph_score: float
    embedding_score: float
    final_score: float
    mentioned_entities: List[str] = field(default_factory=list)

