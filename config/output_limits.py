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

DEEPSEARCH_TOP_CHUNKS = _limit_or_none(_env_int("DEEPSEARCH_TOP_CHUNKS", 10))
DEEPSEARCH_TOP_TRIPLES = _limit_or_none(_env_int("DEEPSEARCH_TOP_TRIPLES", 30))
DEEPSEARCH_TOP_SEED_ENTITIES = _limit_or_none(_env_int("DEEPSEARCH_TOP_SEED_ENTITIES", 15))
DEEPSEARCH_MAX_REASONING_STEPS = _limit_or_none(_env_int("DEEPSEARCH_MAX_REASONING_STEPS", 32))
DEEPSEARCH_MAX_STAGE_HISTORY = _limit_or_none(_env_int("DEEPSEARCH_MAX_STAGE_HISTORY", 10))
DEEPSEARCH_MAX_EXTERNAL_CALLS = _limit_or_none(_env_int("DEEPSEARCH_MAX_EXTERNAL_CALLS", 5))
DEEPSEARCH_MAX_TOOL_METADATA = _limit_or_none(_env_int("DEEPSEARCH_MAX_TOOL_METADATA", 5))


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
