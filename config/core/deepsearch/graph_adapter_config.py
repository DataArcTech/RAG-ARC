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
        return registry.build_adapter(name, **self.parameters)
