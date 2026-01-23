"""Shared helpers for Cypher-backed deterministic graph tools."""
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DirectionalityConfig:
    schema_loaded: bool
    policy: str  # whitelist | blacklist
    directed_relations: Set[str]
    undirected_relations: Set[str]

    def is_sensitive(self, predicate: str) -> bool:
        token = normalize_relation_token(str(predicate or ""))
        if not token:
            return False
        if str(self.policy or "whitelist").strip().lower() == "blacklist":
            return token not in self.undirected_relations
        return token in self.directed_relations


def directionality_config(adapter: Any) -> DirectionalityConfig:
    retriever = getattr(adapter, "retriever", None)
    graph_store = getattr(retriever, "graph_store", None) if retriever is not None else None
    schema = getattr(graph_store, "kg_schema", None) if graph_store is not None else None
    if schema is None:
        return DirectionalityConfig(schema_loaded=False, policy="whitelist", directed_relations=set(), undirected_relations=set())

    policy = "whitelist"
    getter_policy = getattr(schema, "direction_policy_all", None)
    if callable(getter_policy):
        try:
            policy = str(getter_policy() or "whitelist").strip().lower() or "whitelist"
        except Exception:
            policy = "whitelist"

    directed: Set[str] = set()
    getter_directed = getattr(schema, "direction_sensitive_relations_all", None)
    if callable(getter_directed):
        try:
            directed = {normalize_relation_token(str(item)) for item in (getter_directed() or []) if str(item).strip()}
        except Exception:
            directed = set()

    undirected: Set[str] = set()
    getter_undirected = getattr(schema, "direction_insensitive_relations_all", None)
    if callable(getter_undirected):
        try:
            undirected = {normalize_relation_token(str(item)) for item in (getter_undirected() or []) if str(item).strip()}
        except Exception:
            undirected = set()

    return DirectionalityConfig(schema_loaded=True, policy=policy, directed_relations=directed, undirected_relations=undirected)


def kg_schema_loaded(adapter: Any) -> bool:
    retriever = getattr(adapter, "retriever", None)
    graph_store = getattr(retriever, "graph_store", None) if retriever is not None else None
    schema = getattr(graph_store, "kg_schema", None) if graph_store is not None else None
    return schema is not None


def enforce_direction_for_sensitive_predicates(
    direction: str,
    predicates: Sequence[str],
    *,
    directionality: DirectionalityConfig,
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
    if not any(directionality.is_sensitive(p) for p in normalized):
        return token, False
    return default_direction, True


def enforce_undirected_for_non_sensitive_predicates(
    direction: str,
    predicates: Sequence[str],
    *,
    directionality: DirectionalityConfig,
) -> tuple[str, bool]:
    """
    If all provided predicates are NOT direction-sensitive, force undirected traversal.

    This is important when storage canonicalizes non-direction-sensitive facts into a single direction:
    callers using `out`/`in` should still observe undirected semantics for those predicates.
    """
    token = str(direction or "").strip().lower()
    if token not in {"out", "in", "both"}:
        token = "out"
    if not directionality.schema_loaded:
        return token, False
    if token == "both":
        return token, False
    if not predicates:
        return token, False
    normalized = {normalize_relation_token(p) for p in predicates if str(p).strip()}
    if not normalized:
        return token, False
    if not any(directionality.is_sensitive(p) for p in normalized):
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
