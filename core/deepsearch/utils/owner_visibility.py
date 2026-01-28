"""Owner visibility resolution for DeepSearch tools.

DeepSearch tools often need to query multiple "owner" indices (e.g. me + share) while still
respecting algorithm-level scope constraints. This module centralizes:
- parsing tool args (extra.owner_ids / extra.owner_id)
- allowed owner whitelist (me + optional SHARE_OWNER_ID)
- deterministic, observable rejection (no silent fallback)
"""
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from core.graph_adapter.base import GraphAccessScope
from core.utils.owner_guard import get_share_owner_id, is_admin_owner, normalize_owner_id, normalize_owner_ids


@dataclass(frozen=True)
class OwnerVisibilityResolution:
    primary_owner_id: str | None
    owner_ids_used: tuple[str, ...]
    owner_ids_requested: tuple[str, ...]
    owner_ids_allowed: tuple[str, ...]
    owner_ids_rejected: tuple[str, ...]
    source: str

    @property
    def enabled(self) -> bool:
        return bool(self.owner_ids_used)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_owner_id": self.primary_owner_id,
            "owner_ids_used": list(self.owner_ids_used),
            "owner_ids_requested": list(self.owner_ids_requested),
            "owner_ids_allowed": list(self.owner_ids_allowed),
            "owner_ids_rejected": list(self.owner_ids_rejected),
            "source": self.source,
        }


def _coerce_owner_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        return normalize_owner_ids(tokens)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return normalize_owner_ids(raw)
    return normalize_owner_ids([raw])


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def resolve_owner_visibility(
    *,
    extra: Mapping[str, Any] | None,
    access_scope: GraphAccessScope | None,
    graph_context_metadata: Mapping[str, Any] | None = None,
) -> OwnerVisibilityResolution:
    """Resolve owner_ids for tools.

    Priority:
    1) explicit tool args: extra.owner_ids / extra.owner_id
    2) graph_context metadata hints: metadata.owner_ids (optional, best-effort)
    3) default: primary owner only (access_scope.scope_id)

    Allowed whitelist:
    - primary owner (me)
    - SHARE_OWNER_ID (when configured)

    Admin owner:
    - when primary owner is ADMIN_OWNER_ID, allow any requested owner_ids (still normalized).
    """

    extra_payload = dict(extra or {})
    ctx_payload = dict(graph_context_metadata or {})

    primary = normalize_owner_id(access_scope.scope_id) if access_scope and access_scope.scope_id else None
    if not primary:
        return OwnerVisibilityResolution(
            primary_owner_id=None,
            owner_ids_used=tuple(),
            owner_ids_requested=tuple(),
            owner_ids_allowed=tuple(),
            owner_ids_rejected=tuple(),
            source="missing_primary",
        )

    # Tool-arg owner_ids (preferred)
    requested = _coerce_owner_ids(extra_payload.get("owner_ids") or extra_payload.get("owners"))
    if not requested:
        requested = _coerce_owner_ids(extra_payload.get("owner_id"))
    source = "default"
    if requested:
        source = "tool_args"

    # Optional graph_context hint (best effort; used only when tool args omitted)
    if not requested:
        hinted = _coerce_owner_ids(ctx_payload.get("owner_ids"))
        if hinted:
            requested = hinted
            source = "graph_context"

    if not requested:
        requested = [primary]
        source = "primary_only"

    # Allowed owners (me + share)
    allowed: list[str] = [primary]
    share = get_share_owner_id()
    if share and share not in allowed:
        allowed.append(share)

    allowed_tuple = _unique(allowed)
    requested_tuple = _unique(requested)

    if is_admin_owner(primary):
        # Global admin: respect explicit request; when omitted, still default to primary_only above.
        used = requested_tuple
        rejected: tuple[str, ...] = tuple()
    else:
        allowed_set = set(allowed_tuple)
        used = tuple([oid for oid in requested_tuple if oid in allowed_set])
        rejected = tuple([oid for oid in requested_tuple if oid not in allowed_set])

    return OwnerVisibilityResolution(
        primary_owner_id=primary,
        owner_ids_used=used,
        owner_ids_requested=requested_tuple,
        owner_ids_allowed=allowed_tuple,
        owner_ids_rejected=rejected,
        source=source,
    )


__all__ = ["OwnerVisibilityResolution", "resolve_owner_visibility"]

