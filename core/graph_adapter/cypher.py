"""Optional adapter protocol for deterministic Cypher-backed graph queries."""
import re
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from core.graph_adapter.base import GraphAccessScope


_CYPHER_BLOCKLIST_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL|LOAD\s+CSV|APOC)\b",
    flags=re.IGNORECASE,
)
_CYPHER_ALLOWED_PREFIX_RE = re.compile(r"^\s*(MATCH|WITH|RETURN|UNWIND|EXPLAIN|PROFILE)\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_SEMICOLON_RE = re.compile(r";")
_CYPHER_BLOCKLIST_PERIODIC_COMMIT_RE = re.compile(r"\bPERIODIC\s+COMMIT\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_TX_META_RE = re.compile(r"\bTRANSACTION\b", flags=re.IGNORECASE)
_CYPHER_BLOCKLIST_SCHEMA_RE = re.compile(r"\b(CONSTRAINT|INDEX)\b", flags=re.IGNORECASE)


def is_read_only_cypher(cypher: str) -> bool:
    """
    Best-effort read-only guard for Cypher strings.

    This is intentionally conservative: if it cannot confidently treat the query as read-only,
    it returns False.
    """
    text = str(cypher or "").strip()
    if not text:
        return False
    if _CYPHER_BLOCKLIST_SEMICOLON_RE.search(text):
        return False
    if not _CYPHER_ALLOWED_PREFIX_RE.search(text):
        return False
    if _CYPHER_BLOCKLIST_RE.search(text):
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
