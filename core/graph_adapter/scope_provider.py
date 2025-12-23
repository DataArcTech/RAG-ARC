"""Centralised builder/registry for GraphAccessScope instances."""
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import GraphAccessScope


class AccessScopeProvider:
    """Provides consistent GraphAccessScope objects across facades and registries."""

    def __init__(self, *, default_scope: Optional[GraphAccessScope] = None):
        self._default_scope = default_scope

    @property
    def default_scope(self) -> Optional[GraphAccessScope]:
        return self._default_scope

    def ensure(
        self,
        scope: Optional[GraphAccessScope] = None,
        *,
        scope_id: Optional[str] = None,
        scope_type: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> GraphAccessScope:
        """Return an explicit scope or fall back to configured defaults."""

        if scope is not None:
            return scope
        token = (scope_id or "").strip() if scope_id else ""
        if token:
            return GraphAccessScope(
                scope_id=token,
                scope_type=(scope_type or "owner"),
                labels=tuple(labels or ()),
                attributes=dict(attributes) if attributes is not None else None,
            )
        if self._default_scope is not None:
            return self._default_scope
        raise ValueError(
            "GraphAccessScope is required but neither an explicit scope nor a default scope is configured."
        )


def scope_to_dict(scope: Optional[GraphAccessScope]) -> Optional[Dict[str, Any]]:
    """Serialize GraphAccessScope into a JSON-friendly payload."""

    if scope is None:
        return None
    return {
        "scope_id": scope.scope_id,
        "scope_type": scope.scope_type,
        "labels": list(scope.labels or ()),
        "attributes": scope.attributes or None,
    }


def configure_scope_provider(*, default_scope: Optional[GraphAccessScope] = None) -> None:
    """Override the global scope provider (mainly used by tests or bootstrappers)."""

    global _PROVIDER
    _PROVIDER = AccessScopeProvider(default_scope=default_scope or _load_scope_from_env())


def _refresh_scope_from_env_if_needed() -> None:
    """Re-evaluate default scope from env variables when not explicitly configured."""

    global _PROVIDER
    if _PROVIDER.default_scope is not None:
        return
    env_scope = _load_scope_from_env()
    if env_scope is None:
        return
    _PROVIDER = AccessScopeProvider(default_scope=env_scope)


def current_scope_provider() -> AccessScopeProvider:
    """Return the active scope provider."""

    _refresh_scope_from_env_if_needed()
    return _PROVIDER


def require_scope(
    scope: Optional[GraphAccessScope] = None,
    *,
    scope_id: Optional[str] = None,
    scope_type: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    attributes: Optional[Mapping[str, Any]] = None,
) -> GraphAccessScope:
    """Return a scope or raise if neither explicit nor default scope is available."""

    _refresh_scope_from_env_if_needed()
    return _PROVIDER.ensure(
        scope,
        scope_id=scope_id,
        scope_type=scope_type,
        labels=labels,
        attributes=attributes,
    )


def _load_scope_from_env() -> Optional[GraphAccessScope]:
    scope_id = _first_env(
        "DEEPSEARCH_SCOPE_ID",
        "DEEPSEARCH_DEFAULT_SCOPE_ID",
        "DEEPSEARCH_TOOL_MCP_SCOPE_ID",
        "DEVELOP_OWNER_ID",
        "ADMIN_OWNER_ID",
    )
    if not scope_id:
        return None
    scope_type = (
        _first_env("DEEPSEARCH_SCOPE_TYPE", "DEEPSEARCH_TOOL_MCP_SCOPE_TYPE") or "owner"
    )
    labels = _parse_labels(
        _first_env("DEEPSEARCH_SCOPE_LABELS", "DEEPSEARCH_TOOL_MCP_SCOPE_LABELS")
    )
    attributes = _parse_attributes(
        _first_env("DEEPSEARCH_SCOPE_ATTRIBUTES", "DEEPSEARCH_TOOL_MCP_SCOPE_ATTRIBUTES")
    )
    return GraphAccessScope(
        scope_id=str(scope_id).strip(),
        scope_type=str(scope_type or "owner"),
        labels=labels,
        attributes=attributes,
    )


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_labels(raw: Optional[str]) -> Sequence[str]:
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return tuple(str(item).strip() for item in parsed if str(item).strip())
    except json.JSONDecodeError:
        pass
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _parse_attributes(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


_PROVIDER = AccessScopeProvider(default_scope=_load_scope_from_env())

__all__ = [
    "AccessScopeProvider",
    "configure_scope_provider",
    "current_scope_provider",
    "require_scope",
    "scope_to_dict",
]
