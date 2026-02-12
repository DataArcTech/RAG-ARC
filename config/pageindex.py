"""Env-driven configuration for PageIndex section tree + routing indexes."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


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
    logger.warning("Invalid boolean for %s (%s); falling back to %s", name, raw, default)
    return default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except ValueError:
            logger.warning("Invalid integer for %s (%s); falling back to %s", name, raw, default)
            value = int(default)
    if minimum is not None:
        return max(value, int(minimum))
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except ValueError:
            logger.warning("Invalid float for %s (%s); falling back to %s", name, raw, default)
            value = float(default)
    if minimum is not None:
        value = max(value, float(minimum))
    if maximum is not None:
        value = min(value, float(maximum))
    return value


def _env_optional_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    token = str(raw or "").strip()
    return token or None


def pageindex_enabled() -> bool:
    return _env_bool("PAGEINDEX_ENABLED", True)


def section_index_enabled() -> bool:
    return _env_bool("SECTION_INDEX_ENABLED", True)


def section_summary_enabled() -> bool:
    return _env_bool("SECTION_SUMMARY_ENABLED", True)


def section_summary_model_name() -> Optional[str]:
    return _env_optional_str("SECTION_SUMMARY_MODEL")


def section_summary_max_tokens(default: int = 800) -> int:
    return _env_int("SECTION_SUMMARY_MAX_TOKENS", default, minimum=0)


def section_summary_top_k(default: int = 5) -> int:
    return _env_int("SECTION_SUMMARY_TOP_K", default, minimum=0)


def section_summary_max_concurrency(default: int = 6) -> int:
    return _env_int("SECTION_SUMMARY_MAX_CONCURRENCY", default, minimum=1)


def section_summary_leaf_chunk_max_chars(default: int = 1200) -> int:
    return _env_int("SECTION_SUMMARY_LEAF_CHUNK_MAX_CHARS", default, minimum=0)


def section_summary_retry_attempts(default: int = 2) -> int:
    return _env_int("SECTION_SUMMARY_RETRY_ATTEMPTS", default, minimum=1)


def section_top_k(default: int = 6) -> int:
    return _env_int("SECTION_TOP_K", default, minimum=0)


def section_retrieval_candidates_k(default: int = 20) -> int:
    return _env_int("SECTION_RETRIEVE_CANDIDATES_K", default, minimum=0)


def section_rrf_k(default: int = 60) -> int:
    return _env_int("SECTION_RRF_K", default, minimum=1)


def section_min_keep(default: int = 5) -> int:
    return _env_int("SECTION_MIN_KEEP", default, minimum=0)


def section_score_weight(default: float = 0.1) -> float:
    return _env_float("SECTION_SCORE_WEIGHT", default, minimum=0.0)


def section_faiss_index_path(default: str = "./data/section_faiss_index") -> str:
    return str(os.getenv("SECTION_FAISS_INDEX_PATH", default))

def section_faiss_index_type() -> Optional[str]:
    """Override index_type for PageIndex section FAISS index only (default: flat)."""
    token = str(os.getenv("SECTION_FAISS_INDEX_TYPE", "") or "").strip().lower()
    if not token:
        return None
    if token in {"flat", "ivf", "hnsw"}:
        return token
    logger.warning("Invalid SECTION_FAISS_INDEX_TYPE=%s; expected flat/ivf/hnsw. Ignoring.", token)
    return None

def section_faiss_two_stage_enabled() -> Optional[bool]:
    """Override two-stage retrieval for PageIndex section FAISS index only (default: inherit base FAISS config)."""
    token = str(os.getenv("SECTION_FAISS_TWO_STAGE_ENABLED", "") or "").strip().lower()
    if not token:
        return None
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning(
        "Invalid SECTION_FAISS_TWO_STAGE_ENABLED=%s; expected true/false. Ignoring.",
        token,
    )
    return None


def section_faiss_two_stage_prefetch_k() -> Optional[int]:
    """Override two-stage prefetch_k for PageIndex section FAISS index only (default: inherit base FAISS config)."""
    token = str(os.getenv("SECTION_FAISS_TWO_STAGE_PREFETCH_K", "") or "").strip()
    if not token:
        return None
    try:
        value = int(token)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Invalid SECTION_FAISS_TWO_STAGE_PREFETCH_K=%s; expected int. Ignoring.",
            token,
        )
        return None
    if value < 1:
        logger.warning(
            "Invalid SECTION_FAISS_TWO_STAGE_PREFETCH_K=%s; expected >=1. Ignoring.",
            token,
        )
        return None
    return value


def section_bm25_index_path(default: str = "./data/section_bm25_index") -> str:
    return str(os.getenv("SECTION_BM25_INDEX_PATH", default))


def doc_routing_enabled() -> bool:
    # Doc routing is handled by DeepSearch `search.file` (relevant-file routing).
    return False


def doc_desc_model_name() -> Optional[str]:
    # Reserved for legacy configs (doc descriptions are not part of PageIndex outputs).
    return None


def doc_desc_max_tokens(default: int = 400) -> int:
    return 0

def doc_profile_enabled() -> bool:
    # Reserved for legacy configs (doc profiles are not part of PageIndex outputs).
    return False


def doc_profile_model_name() -> Optional[str]:
    return None


def doc_profile_max_tokens(default: int = 450) -> int:
    return 0


def doc_profile_max_list_items(default: int = 12) -> int:
    return 0


def doc_top_k(default: int = 5) -> int:
    return 0


def doc_retrieval_candidates_k(default: int = 10) -> int:
    return 0


def doc_rrf_k(default: int = 60) -> int:
    return 0


def doc_routing_faiss_index_path(default: str = "./data/doc_routing_faiss_index") -> str:
    return default


def doc_routing_bm25_index_path(default: str = "./data/doc_routing_bm25_index") -> str:
    return default


def section_level_conflict_ratio(default: float = 0.4) -> float:
    return _env_float("SECTION_LEVEL_CONFLICT_RATIO", default, minimum=0.0, maximum=1.0)


def section_level_force_flat_if_uniform() -> bool:
    return _env_bool("SECTION_LEVEL_FORCE_FLAT_IF_UNIFORM", True)


def section_level_max(default: int = 6) -> int:
    return _env_int("SECTION_LEVEL_MAX", default, minimum=1)


def section_numbering_enabled() -> bool:
    return _env_bool("SECTION_NUMBERING_ENABLED", True)


def section_numbering_max_level(default: int = 6) -> int:
    return _env_int("SECTION_NUMBERING_MAX_LEVEL", default, minimum=1)


def pageindex_tree_filename(default: str = "pageindex_tree.json") -> str:
    return str(os.getenv("PAGEINDEX_TREE_FILENAME", default))


def pageindex_nodes_filename(default: str = "pageindex_nodes.jsonl") -> str:
    return str(os.getenv("PAGEINDEX_NODES_FILENAME", default))


def pageindex_doc_filename(default: str = "pageindex_doc.json") -> str:
    # Reserved for legacy layouts (PageIndex produces section tree/nodes artifacts).
    return default


def section_path_delimiter(default: str = " > ") -> str:
    return str(os.getenv("SECTION_PATH_DELIMITER", default))


def section_chunk_match_snippet_chars(default: int = 200) -> int:
    return _env_int("SECTION_CHUNK_MATCH_SNIPPET_CHARS", default, minimum=0)


def section_page_match_snippet_chars(default: int = 160) -> int:
    return _env_int("SECTION_PAGE_MATCH_SNIPPET_CHARS", default, minimum=0)


def section_page_match_max_pages(default: int = 20) -> int:
    return _env_int("SECTION_PAGE_MATCH_MAX_PAGES", default, minimum=0)


def strict_page_chunking_mode(default: str = "auto") -> str:
    """Control strict page-aware chunking behavior.

    - off: disable strict page chunking
    - auto: enable when content_list is available; fallback otherwise (with warning)
    - require: fail indexing when content_list is missing
    """
    raw = os.getenv("PAGEINDEX_STRICT_PAGE_CHUNKING_MODE", default)
    token = str(raw or "").strip().lower()
    if token in {"off", "auto", "require"}:
        return token
    logger.warning(
        "Invalid PAGEINDEX_STRICT_PAGE_CHUNKING_MODE=%s; expected off/auto/require. Falling back to %s.",
        raw,
        default,
    )
    return default


def strict_page_chunking_enabled() -> bool:
    return strict_page_chunking_mode() != "off"

def toc_check_page_num(default: int = 20) -> int:
    return _env_int("PAGEINDEX_TOC_CHECK_PAGE_NUM", default, minimum=1)


def max_page_num_each_node(default: int = 10) -> int:
    return _env_int("PAGEINDEX_MAX_PAGE_NUM_EACH_NODE", default, minimum=1)


def max_token_num_each_node(default: int = 20000) -> int:
    return _env_int("PAGEINDEX_MAX_TOKEN_NUM_EACH_NODE", default, minimum=1)


__all__ = [
    "pageindex_enabled",
    "section_index_enabled",
    "section_summary_enabled",
    "section_summary_model_name",
    "section_summary_max_tokens",
    "section_summary_top_k",
    "section_summary_max_concurrency",
    "section_summary_leaf_chunk_max_chars",
    "section_top_k",
    "section_retrieval_candidates_k",
    "section_rrf_k",
    "section_min_keep",
    "section_score_weight",
    "section_faiss_index_path",
    "section_faiss_index_type",
    "section_faiss_two_stage_enabled",
    "section_faiss_two_stage_prefetch_k",
    "section_bm25_index_path",
    "section_level_conflict_ratio",
    "section_level_force_flat_if_uniform",
    "section_level_max",
    "section_numbering_enabled",
    "section_numbering_max_level",
    "pageindex_tree_filename",
    "pageindex_nodes_filename",
    "section_path_delimiter",
    "section_chunk_match_snippet_chars",
    "section_page_match_snippet_chars",
    "section_page_match_max_pages",
    "strict_page_chunking_mode",
    "strict_page_chunking_enabled",
    "toc_check_page_num",
    "max_page_num_each_node",
    "max_token_num_each_node",
]
