"""Config factory for GraphTraversal settings."""
import os
from typing import Literal

from pydantic import Field

from core.deepsearch.reasoning import GraphTraversalSettings
from framework.config import AbstractConfig


class GraphTraversalSettingsConfig(AbstractConfig):
    """Builds GraphTraversalSettings for DeepSearch graph loop."""

    type: Literal["graph_traversal_settings"] = "graph_traversal_settings"
    strategy_name: str = Field("ppr_chain", description="Traversal strategy label")
    allow_semantic_channel: bool = Field(True, description="Allow semantic filters alongside relational ones")
    chain_depth: int = Field(4, description="Maximum hops explored during traversal")

    def build(self) -> GraphTraversalSettings:
        env = os.getenv

        def _bool(name: str, default: bool) -> bool:
            value = env(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        strategy = env("DEEPSEARCH_GRAPH_STRATEGY") or self.strategy_name
        allow_semantic = _bool("DEEPSEARCH_ALLOW_SEMANTIC_CHANNEL", self.allow_semantic_channel)
        chain_depth = int(env("DEEPSEARCH_CHAIN_DEPTH") or self.chain_depth)

        return GraphTraversalSettings(
            strategy_name=strategy,
            allow_semantic_channel=allow_semantic,
            chain_depth=chain_depth,
        )
