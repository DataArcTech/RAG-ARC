import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s (%s); falling back to %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s (%s); falling back to %s", name, raw, default)
    return default


def _limit_or_none(value: int) -> int | None:
    if ENABLE_ALL_EVIDENCE:
        return None
    return max(value, 0)


def _max_images_or_none(value: int) -> int | None:
    """
    Multimodal image attachment limits.

    Semantics:
    - If ENABLE_ALL_EVIDENCE is on: unlimited (None)
    - If value <= 0: unlimited (None)
    - Else: cap to the provided positive integer
    """
    if ENABLE_ALL_EVIDENCE:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer for %s (%s); ignoring override", name, raw)
        return None


ENABLE_ALL_EVIDENCE = _env_bool("ENABLE_ALL_EVIDENCE", False)

CHAT_TOP_CHUNKS = _limit_or_none(_env_int("CHAT_TOP_CHUNKS", 5))
CHAT_TOP_TRIPLES = _limit_or_none(_env_int("CHAT_TOP_TRIPLES", 5))
CHAT_TOP_SEED_ENTITIES = _limit_or_none(_env_int("CHAT_TOP_SEED_ENTITIES", 5))
CHAT_MAX_IMAGE_INPUTS = _max_images_or_none(_env_int("CHAT_MAX_IMAGE_INPUTS", 4))

# Chatbot-only: allow sending more sources to the LLM than the UI displays.
# (UI uses CHATBOT_TOP_SOURCES; LLM uses this to improve coverage for broad questions like "有什么特点".)
CHATBOT_LLM_TOP_SOURCES = _env_int("CHATBOT_LLM_TOP_SOURCES", 10)

# Retrieval observability (server logs / progress payloads).
# Default to disabled to avoid log noise; enable explicitly when debugging.
RAG_RETRIEVAL_OBSERVABILITY = _env_bool("RAG_RETRIEVAL_OBSERVABILITY", False)
RAG_RETRIEVAL_LOG_TOP_FILES = _env_int("RAG_RETRIEVAL_LOG_TOP_FILES", 10)
RAG_RETRIEVAL_LOG_TOP_CHUNKS = _env_int("RAG_RETRIEVAL_LOG_TOP_CHUNKS", 5)

DEEPSEARCH_TOP_CHUNKS = _limit_or_none(_env_int("DEEPSEARCH_TOP_CHUNKS", 10))
DEEPSEARCH_TOP_TRIPLES = _limit_or_none(_env_int("DEEPSEARCH_TOP_TRIPLES", 30))
DEEPSEARCH_TOP_SEED_ENTITIES = _limit_or_none(_env_int("DEEPSEARCH_TOP_SEED_ENTITIES", 15))
DEEPSEARCH_MAX_IMAGE_INPUTS = _max_images_or_none(_env_int("DEEPSEARCH_MAX_IMAGE_INPUTS", 6))
DEEPSEARCH_MAX_REASONING_STEPS = _limit_or_none(_env_int("DEEPSEARCH_MAX_REASONING_STEPS", 32))
DEEPSEARCH_MAX_STAGE_HISTORY = _limit_or_none(_env_int("DEEPSEARCH_MAX_STAGE_HISTORY", 10))
DEEPSEARCH_MAX_EXTERNAL_CALLS = _limit_or_none(_env_int("DEEPSEARCH_MAX_EXTERNAL_CALLS", 5))
DEEPSEARCH_MAX_TOOL_METADATA = _limit_or_none(_env_int("DEEPSEARCH_MAX_TOOL_METADATA", 5))

DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS = _env_int("DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS", 180)
DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT = _env_int("DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT", 3)
DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES = _env_int("DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES", 2000)
DEEPSEARCH_GRAPH_EXPORT_MAX_ALIASES = _env_int("DEEPSEARCH_GRAPH_EXPORT_MAX_ALIASES", 8)
DEEPSEARCH_MINDMAP_MAX_NODES_PER_CHUNK = _env_int("DEEPSEARCH_MINDMAP_MAX_NODES_PER_CHUNK", 30)

# DeepSearch answer/source presentation (HippoRAG-compatible citations).
DEEPSEARCH_SOURCE_MAX_CHARS = _env_int("DEEPSEARCH_SOURCE_MAX_CHARS", 1600)
DEEPSEARCH_SOURCE_TITLE_MAX_CHARS = _env_int("DEEPSEARCH_SOURCE_TITLE_MAX_CHARS", 80)

# Weaver tool rendering limits (kept centralized; not exposed as env knobs by default).
DEEPSEARCH_WEAVER_TOOL_ABOUT_CHARS = 200
DEEPSEARCH_WEAVER_TOOL_QUERY_CHARS = 280
DEEPSEARCH_WEAVER_TOOL_PURPOSE_CHARS = 240
DEEPSEARCH_WEAVER_TOOL_CODE_PREVIEW_CHARS = 220
DEEPSEARCH_WEAVER_TOOL_ERROR_CHARS = 600
DEEPSEARCH_WEAVER_TOOL_EXEC_ERROR_CHARS = 800

KNOWLEDGE_GRAPH_EXPORT_MAX_NODES = _env_int("KNOWLEDGE_GRAPH_EXPORT_MAX_NODES", 1000)
KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES = _env_int("KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES", 5000)
KNOWLEDGE_MINDMAP_EXPORT_MAX_CHUNKS = _env_int("KNOWLEDGE_MINDMAP_EXPORT_MAX_CHUNKS", 60)
KNOWLEDGE_MINDMAP_EXPORT_SEGMENT_SNIPPET_CHARS = _env_int("KNOWLEDGE_MINDMAP_EXPORT_SEGMENT_SNIPPET_CHARS", 600)
GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS = _env_int("GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS", 240)
GRAPH_EXPORT_EDGE_FETCH_FACTOR = _env_int("GRAPH_EXPORT_EDGE_FETCH_FACTOR", 10)
GRAPH_EXPORT_EDGE_FETCH_MAX = _env_int("GRAPH_EXPORT_EDGE_FETCH_MAX", 50000)
GRAPH_EXPORT_FILTER_NUMERIC_TIME_ENTITIES = _env_bool("GRAPH_EXPORT_FILTER_NUMERIC_TIME_ENTITIES", True)

SEMANTIC_UNIT_MAX_MATCHED_SLICES = _limit_or_none(_env_int("SEMANTIC_UNIT_MAX_MATCHED_SLICES", 3))
TABLE_MAX_MERGED_ROWS = _limit_or_none(_env_int("TABLE_MAX_MERGED_ROWS", 30))
SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS = _limit_or_none(_env_int("SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS", 1200))
SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS = _limit_or_none(_env_int("SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS", 3000))


def _graph_budget(chunk_limit: int | None, triple_limit: int | None, seed_limit: int | None, minimum: int) -> int | None:
    if ENABLE_ALL_EVIDENCE:
        return None
    total = (chunk_limit or 0) + (triple_limit or 0) * 2 + (seed_limit or 0)
    return max(total, minimum)


CHAT_GRAPH_NODE_LIMIT = _graph_budget(CHAT_TOP_CHUNKS, CHAT_TOP_TRIPLES, CHAT_TOP_SEED_ENTITIES, 30)
CHAT_GRAPH_EDGE_LIMIT = None if ENABLE_ALL_EVIDENCE else max((CHAT_TOP_TRIPLES or 0) * 4, 60)

_DEEPSEARCH_NODE_OVERRIDE = _env_optional_int("DEEPSEARCH_GRAPH_NODE_LIMIT")
_DEEPSEARCH_EDGE_OVERRIDE = _env_optional_int("DEEPSEARCH_GRAPH_EDGE_LIMIT")

if ENABLE_ALL_EVIDENCE:
    DEEPSEARCH_GRAPH_NODE_LIMIT = None
    DEEPSEARCH_GRAPH_EDGE_LIMIT = None
else:
    default_node_budget = _graph_budget(DEEPSEARCH_TOP_CHUNKS, DEEPSEARCH_TOP_TRIPLES, None, 75)
    default_edge_budget = max((DEEPSEARCH_TOP_TRIPLES or 0) * 4, 200)
    DEEPSEARCH_GRAPH_NODE_LIMIT = (
        max(_DEEPSEARCH_NODE_OVERRIDE, 0) if _DEEPSEARCH_NODE_OVERRIDE is not None else default_node_budget
    )
    DEEPSEARCH_GRAPH_EDGE_LIMIT = (
        max(_DEEPSEARCH_EDGE_OVERRIDE, 0) if _DEEPSEARCH_EDGE_OVERRIDE is not None else default_edge_budget
    )
