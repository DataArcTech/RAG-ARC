"""Env-driven toggles for intent-aware multi-turn RAG behaviors.

Keep feature flags centralized and easy to gray-rollout.
"""
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


def rag_intent_routing_enabled() -> bool:
    """Enable intent-aware rewrite/routing (default: disabled until validated)."""
    return _env_bool("RAG_INTENT_ROUTING_ENABLED", False)


def rag_evidence_consistency_enabled() -> bool:
    """Enable evidence consistency filtering by anchors (default: disabled until validated)."""
    return _env_bool("RAG_EVIDENCE_CONSISTENCY_ENABLED", False)


def rag_rewrite_history_user_only() -> bool:
    """When True, only user turns are passed into the rewrite context (reduces assistant poisoning)."""
    return _env_bool("RAG_REWRITE_HISTORY_USER_ONLY", True)


def rag_rewrite_history_most_recent_first() -> bool:
    """When True, rewrite history context is ordered from most recent to oldest."""
    return _env_bool("RAG_REWRITE_HISTORY_MOST_RECENT_FIRST", True)


def rag_evidence_min_keep(default: int = 5) -> int:
    raw = os.getenv("RAG_EVIDENCE_MIN_KEEP")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return max(int(str(raw).strip()), 0)
    except ValueError:
        return int(default)


def rag_anchor_filename_fallback_enabled() -> bool:
    """Enable a deterministic fallback that derives shorter anchor terms for filename matching."""
    return _env_bool("RAG_ANCHOR_FILENAME_FALLBACK_ENABLED", True)


def rag_anchor_filename_fallback_min_len(default: int = 2) -> int:
    raw = os.getenv("RAG_ANCHOR_FILENAME_FALLBACK_MIN_LEN")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return max(int(str(raw).strip()), 1)
    except ValueError:
        return int(default)


def rag_anchor_filename_fallback_max_len(default: int = 4) -> int:
    raw = os.getenv("RAG_ANCHOR_FILENAME_FALLBACK_MAX_LEN")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return max(int(str(raw).strip()), 1)
    except ValueError:
        return int(default)


def rag_anchor_filename_fallback_max_file_fraction(default: float = 0.3) -> float:
    """Max fraction of distinct filenames that a fallback token may match (avoid overly generic tokens)."""
    val = _env_float("RAG_ANCHOR_FILENAME_FALLBACK_MAX_FILE_FRACTION", default)
    if val < 0:
        return 0.0
    if val > 1:
        return 1.0
    return float(val)


def rag_anchor_filename_fallback_max_terms(default: int = 3) -> int:
    raw = os.getenv("RAG_ANCHOR_FILENAME_FALLBACK_MAX_TERMS")
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return max(int(str(raw).strip()), 0)
    except ValueError:
        return int(default)


__all__ = [
    "rag_intent_routing_enabled",
    "rag_evidence_consistency_enabled",
    "rag_rewrite_history_user_only",
    "rag_rewrite_history_most_recent_first",
    "rag_evidence_min_keep",
    "rag_anchor_filename_fallback_enabled",
    "rag_anchor_filename_fallback_min_len",
    "rag_anchor_filename_fallback_max_len",
    "rag_anchor_filename_fallback_max_file_fraction",
    "rag_anchor_filename_fallback_max_terms",
]
