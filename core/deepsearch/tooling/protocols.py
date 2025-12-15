"""Protocols used by core reasoning loops to invoke tools without depending on infra."""
from typing import Any, Dict, Protocol

from encapsulation.data_model.deepsearch import ToolResultPayload


class ToolInvoker(Protocol):
    async def invoke(self, tool_name: str, *, payload: Dict[str, Any]) -> ToolResultPayload:
        """Invoke a tool and return a normalized payload."""

