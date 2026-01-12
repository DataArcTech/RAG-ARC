"""Stable error typing for DeepSearch tool invocations (governance + observability)."""
import asyncio
from enum import Enum
from typing import Optional

from pydantic import ValidationError

from core.graph_adapter.cypher import CypherPolicyError


class ToolErrorKind(str, Enum):
    """Normalized error taxonomy for tool invocation governance."""

    SCHEMA_ERROR = "schema_error"
    EMPTY_HIT = "empty_hit"
    POLICY_REJECT = "policy_reject"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    UNKNOWN = "unknown"


class ToolInvocationError(RuntimeError):
    """Tool invocation failed with a typed, observable error kind."""

    def __init__(
        self,
        *,
        tool_name: str,
        kind: ToolErrorKind,
        message: str,
        route: Optional[str] = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.kind = kind
        self.route = route
        if cause is not None:
            self.__cause__ = cause


def classify_tool_exception(exc: Exception, *, route: Optional[str] = None) -> ToolErrorKind:
    """Map an exception into a stable ToolErrorKind."""

    if isinstance(exc, ToolInvocationError):
        return exc.kind
    try:
        from core.deepsearch.tooling.budget import ToolBudgetExceededError

        if isinstance(exc, ToolBudgetExceededError):
            return ToolErrorKind.BUDGET_EXCEEDED
    except Exception:
        pass
    if isinstance(exc, asyncio.TimeoutError):
        return ToolErrorKind.TIMEOUT
    if isinstance(exc, CypherPolicyError):
        return ToolErrorKind.POLICY_REJECT
    if isinstance(exc, (ValidationError, TypeError, ValueError)):
        return ToolErrorKind.SCHEMA_ERROR
    if isinstance(exc, KeyError):
        return ToolErrorKind.TOOL_UNAVAILABLE
    if route == "remote":
        return ToolErrorKind.PROVIDER_ERROR
    return ToolErrorKind.UNKNOWN


def wrap_tool_exception(
    tool_name: str,
    exc: Exception,
    *,
    route: Optional[str] = None,
    message: Optional[str] = None,
) -> ToolInvocationError:
    """Wrap an arbitrary exception into ToolInvocationError (idempotent)."""

    if isinstance(exc, ToolInvocationError):
        return exc
    kind = classify_tool_exception(exc, route=route)
    return ToolInvocationError(
        tool_name=tool_name,
        kind=kind,
        message=message or str(exc) or kind.value,
        route=route,
        cause=exc,
    )
