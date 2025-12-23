"""Graph adapter configuration that builds adapters via the registry."""
import os
from typing import Any, Dict, Literal

from pydantic import Field

from core.graph_adapter import registry
from framework.config import AbstractConfig


class GraphAdapterConfig(AbstractConfig):
    """Resolve GraphDeepSearchAdapter instances in a configurable way."""

    type: Literal["graph_adapter"] = "graph_adapter"
    adapter_name: str = Field(..., description="Registered adapter name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Adapter-specific keyword arguments")

    def build(self):
        env = os.getenv
        name = env("DEEPSEARCH_DEFAULT_ADAPTER") or self.adapter_name
        self._validate_parameters(name)
        return registry.build_adapter(name, **self.parameters)

    def _validate_parameters(self, name: str) -> None:
        """Ensure required parameters are present for known adapters."""

        if name == "hipporag":
            params = self.parameters or {}
            has_retriever = bool(params.get("retriever"))
            has_retriever_config = bool(params.get("retriever_config"))
            if not has_retriever and not has_retriever_config:
                raise ValueError(
                    "HippoRAG adapter requires 'retriever_config' under graph_adapter.parameters "
                    "(or an already constructed 'retriever'). Please update your config JSON or set "
                    "DEEPSEARCH_TOOL_MCP_ADAPTER_CONFIG to a file that includes this payload."
                )
