"""Optional adapter protocol for deterministic Cypher-backed graph queries."""
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, runtime_checkable

from core.graph_adapter.base import GraphAccessScope


_CYPHER_BLOCKLIST_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV|APOC)\b",
    flags=re.IGNORECASE,
)
_CYPHER_ALLOWED_PREFIX_RE = re.compile(r"^\s*(MATCH|WITH|RETURN|UNWIND|EXPLAIN|PROFILE|CALL)\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_SEMICOLON_RE = re.compile(r";")
_CYPHER_BLOCKLIST_PERIODIC_COMMIT_RE = re.compile(r"\bPERIODIC\s+COMMIT\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_TX_META_RE = re.compile(r"\bTRANSACTION(S)?\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_SCHEMA_RE = re.compile(r"\b(CONSTRAINT|INDEX)\b", flags=re.IGNORECASE)
_CYPHER_CALL_TOKEN_RE = re.compile(r"\bCALL\b", flags=re.IGNORECASE)
_CYPHER_ALLOWED_CALL_SUBQUERY_RE = re.compile(r"^\s*CALL\s*\{", flags=re.IGNORECASE)


def _strip_leading_cypher_comments(text: str) -> str:
    """Remove leading Cypher comment blocks so prefix checks remain stable."""

    remainder = str(text or "")
    while True:
        remainder = remainder.lstrip()
        if remainder.startswith("//"):
            if "\n" not in remainder:
                return ""
            remainder = remainder.split("\n", 1)[1]
            continue
        if remainder.startswith("/*"):
            end = remainder.find("*/")
            if end < 0:
                return ""
            remainder = remainder[end + 2 :]
            continue
        return remainder


def _allows_only_subquery_calls(text: str) -> bool:
    """Permit only Cypher CALL { ... } subqueries; reject CALL proc()/dbms/apoc/etc."""

    for match in _CYPHER_CALL_TOKEN_RE.finditer(text):
        tail = text[match.start() :]
        if not _CYPHER_ALLOWED_CALL_SUBQUERY_RE.match(tail):
            return False
    return True


def is_read_only_cypher(cypher: str) -> bool:
    """
    Best-effort read-only guard for Cypher strings.

    This is intentionally conservative: if it cannot confidently treat the query as read-only,
    it returns False.
    """
    text = _strip_leading_cypher_comments(str(cypher or "")).strip()
    if not text:
        return False
    if _CYPHER_BLOCKLIST_SEMICOLON_RE.search(text):
        return False
    if not _CYPHER_ALLOWED_PREFIX_RE.search(text):
        return False
    if _CYPHER_BLOCKLIST_RE.search(text):
        return False
    if not _allows_only_subquery_calls(text):
        return False
    if _CYPHER_BLOCKLIST_PERIODIC_COMMIT_RE.search(text):
        return False
    if _CYPHER_BLOCKLIST_TX_META_RE.search(text):
        return False
    if _CYPHER_BLOCKLIST_SCHEMA_RE.search(text):
        return False
    return True


def assert_read_only_cypher(cypher: str) -> None:
    if not is_read_only_cypher(cypher):
        raise ValueError("Cypher query rejected: only single-statement read-only Cypher is allowed.")


@runtime_checkable
class GraphCypherQueryable(Protocol):
    """Adapters that can execute deterministic Cypher queries against a graph store."""

    def cypher_capable(self) -> bool:
        """Return True when the adapter is backed by a Cypher-capable graph store (e.g., Neo4j)."""

    async def acypher(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
        *,
        access_scope: Optional[GraphAccessScope] = None,
    ) -> list[Dict[str, Any]]:
        """Execute a Cypher query and return rows as dictionaries."""


def adapter_supports_cypher(adapter: Any) -> bool:
    """Best-effort check for whether an adapter can run deterministic Cypher queries.

    Preference order:
    1) adapter.cypher_capable() when present
    2) adapter.metadata().capabilities includes "cypher_query"
    """
    if adapter is None:
        return False
    acypher = getattr(adapter, "acypher", None)
    if not callable(acypher):
        return False

    cap = getattr(adapter, "cypher_capable", None)
    if callable(cap):
        try:
            return bool(cap())
        except Exception:
            return False

    meta_fn = getattr(adapter, "metadata", None)
    if not callable(meta_fn):
        return False
    try:
        meta = meta_fn()
    except Exception:
        return False
    capabilities = getattr(meta, "capabilities", None)
    if capabilities is None and isinstance(meta, dict):
        capabilities = meta.get("capabilities")
    if not isinstance(capabilities, Iterable):
        return False
    for item in capabilities:
        if item is None:
            continue
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if str(name or "").strip() == "cypher_query":
            return True
    return False
