"""Factory helpers for configuring EntityResolver from centralized defaults."""
from config.core.deepsearch.tool_defaults import (
    ENTITY_RESOLUTION_ALIAS_SCORE_BONUS,
    ENTITY_RESOLUTION_ENABLE_ALIAS,
    ENTITY_RESOLUTION_ENABLE_CHUNK_VALIDATION,
    ENTITY_RESOLUTION_ENABLE_EMBEDDING_FALLBACK,
    ENTITY_RESOLUTION_ENABLE_TOKEN_OVERLAP,
    ENTITY_RESOLUTION_FAISS_MIN_SIMILARITY,
    ENTITY_RESOLUTION_FAISS_TOP_K,
    ENTITY_RESOLUTION_REQUIRE_MIN_EDGE_COUNT,
    ENTITY_RESOLUTION_REQUIRE_MIN_MENTION_COUNT,
    ENTITY_RESOLUTION_SCORE_WEIGHT_CHAR_RATIO,
    ENTITY_RESOLUTION_SCORE_WEIGHT_TOKEN_F1,
    ENTITY_RESOLUTION_VALIDATE_EDGES_FIRST,
)

from .resolver import EntityResolver


def build_default_entity_resolver(
    *,
    enabled: bool = True,
    candidate_limit: int = 12,
    min_token_len: int = 3,
    min_token_hits: int = 2,
    auto_score_min: float = 0.86,
    auto_score_margin: float = 0.06,
) -> EntityResolver:
    """Build a resolver configured via `config/core/deepsearch/tool_defaults.py`."""

    return EntityResolver(
        enabled=bool(enabled),
        candidate_limit=int(candidate_limit),
        min_token_len=int(min_token_len),
        min_token_hits=int(min_token_hits),
        auto_score_min=float(auto_score_min),
        auto_score_margin=float(auto_score_margin),
        enable_alias=bool(ENTITY_RESOLUTION_ENABLE_ALIAS),
        enable_token_overlap=bool(ENTITY_RESOLUTION_ENABLE_TOKEN_OVERLAP),
        enable_embedding_fallback=bool(ENTITY_RESOLUTION_ENABLE_EMBEDDING_FALLBACK),
        faiss_top_k=int(ENTITY_RESOLUTION_FAISS_TOP_K),
        faiss_min_similarity=ENTITY_RESOLUTION_FAISS_MIN_SIMILARITY,
        validate_edges_first=bool(ENTITY_RESOLUTION_VALIDATE_EDGES_FIRST),
        require_min_edge_count=int(ENTITY_RESOLUTION_REQUIRE_MIN_EDGE_COUNT),
        enable_chunk_validation=bool(ENTITY_RESOLUTION_ENABLE_CHUNK_VALIDATION),
        require_min_mention_count=int(ENTITY_RESOLUTION_REQUIRE_MIN_MENTION_COUNT),
        score_weight_token_f1=float(ENTITY_RESOLUTION_SCORE_WEIGHT_TOKEN_F1),
        score_weight_char_ratio=float(ENTITY_RESOLUTION_SCORE_WEIGHT_CHAR_RATIO),
        alias_score_bonus=float(ENTITY_RESOLUTION_ALIAS_SCORE_BONUS),
    )

