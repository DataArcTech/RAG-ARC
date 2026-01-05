"""Shared helpers for Cypher-backed deterministic graph tools."""
from typing import Any, List, Sequence, Set

from core.utils.text_processing import text_processing

from core.knowledge_graph.schema import normalize_relation_token


def normalize_predicates(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        token = normalize_relation_token(str(value or ""))
        if not token:
            continue
        out.append(token)
    return sorted(set(out))


def direction_sensitive_predicates(adapter: Any) -> Set[str]:
    retriever = getattr(adapter, "retriever", None)
    graph_store = getattr(retriever, "graph_store", None) if retriever is not None else None
    schema = getattr(graph_store, "kg_schema", None) if graph_store is not None else None
    if schema is None:
        return set()
    getter = getattr(schema, "direction_sensitive_relations_all", None)
    if not callable(getter):
        return set()
    try:
        return {normalize_relation_token(str(item)) for item in (getter() or []) if str(item).strip()}
    except Exception:
        return set()


def kg_schema_loaded(adapter: Any) -> bool:
    retriever = getattr(adapter, "retriever", None)
    graph_store = getattr(retriever, "graph_store", None) if retriever is not None else None
    schema = getattr(graph_store, "kg_schema", None) if graph_store is not None else None
    return schema is not None


def enforce_direction_for_sensitive_predicates(
    direction: str,
    predicates: Sequence[str],
    *,
    sensitive_predicates: Set[str],
    default_direction: str,
) -> tuple[str, bool]:
    token = str(direction or "").strip().lower()
    if token not in {"out", "in", "both"}:
        token = default_direction
    if token != "both":
        return token, False
    if not predicates:
        return token, False
    normalized = {normalize_relation_token(p) for p in predicates if str(p).strip()}
    if normalized.isdisjoint(sensitive_predicates):
        return token, False
    return default_direction, True


def enforce_undirected_for_non_sensitive_predicates(
    direction: str,
    predicates: Sequence[str],
    *,
    sensitive_predicates: Set[str],
    schema_loaded: bool,
) -> tuple[str, bool]:
    """
    If all provided predicates are NOT direction-sensitive, force undirected traversal.

    This is important when storage canonicalizes non-direction-sensitive facts into a single direction:
    callers using `out`/`in` should still observe undirected semantics for those predicates.
    """
    token = str(direction or "").strip().lower()
    if token not in {"out", "in", "both"}:
        token = "out"
    if not schema_loaded:
        return token, False
    if token == "both":
        return token, False
    if not predicates:
        return token, False
    normalized = {normalize_relation_token(p) for p in predicates if str(p).strip()}
    if not normalized:
        return token, False
    if normalized.isdisjoint(sensitive_predicates):
        return "both", True
    return token, False


def limit_int(raw: Any, default: int, *, min_value: int = 1, max_value: int = 100) -> int:
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(min_value, min(max_value, value))


def rel_pattern(direction: str, *, rel_var: str, rel_type: str) -> str:
    token = str(direction or "out").strip().lower()
    if token == "both":
        return f"-[{rel_var}:{rel_type}]-"
    if token == "in":
        return f"<-[{rel_var}:{rel_type}]-"
    return f"-[{rel_var}:{rel_type}]->"


def rel_pattern_varlen(direction: str, *, rel_type: str, max_hops: int) -> str:
    token = str(direction or "out").strip().lower()
    max_hops = max(1, int(max_hops))
    if token == "both":
        return f"-[:{rel_type}*1..{max_hops}]-"
    if token == "in":
        return f"<-[:{rel_type}*1..{max_hops}]-"
    return f"-[:{rel_type}*1..{max_hops}]->"


def normalize_entity_name(raw: Any) -> str:
    return text_processing(str(raw or ""))
