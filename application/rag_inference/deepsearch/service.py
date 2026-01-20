"""Service façade wiring DeepSearch planner, reasoning loop, and reporting."""

import logging
from typing import Any, Dict, Type

from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.state import DeepSearchState
from core.deepsearch.tooling.protocols import ToolInvoker

from .service_runtime import (
    DeepSearchServiceArtifactsMixin,
    DeepSearchServiceContextMixin,
    DeepSearchServiceInitialThinkMixin,
    DeepSearchServicePipelineMixin,
    DeepSearchServiceQualityMixin,
    DeepSearchServiceRoutingMixin,
    DeepSearchServiceRunMixin,
    DeepSearchServiceStageMixin,
)

logger = logging.getLogger(__name__)


class DeepSearchService(
    DeepSearchServiceRunMixin,
    DeepSearchServiceStageMixin,
    DeepSearchServicePipelineMixin,
    DeepSearchServiceContextMixin,
    DeepSearchServiceArtifactsMixin,
    DeepSearchServiceRoutingMixin,
    DeepSearchServiceQualityMixin,
    DeepSearchServiceInitialThinkMixin,
):
    """Application-layer facade with a shared async run() entry point for FastAPI/CLI/MCP."""

    def __init__(
        self,
        planner: DeepSearchPlanner,
        graph_loop: GraphReasoningLoop,
        reporter: DeepSearchReporter,
        tool_manager: ToolInvoker,
        *,
        state_cls: Type[DeepSearchState] = DeepSearchState,
        config: Dict[str, Any] | None = None,
    ):
        self.planner = planner
        self.graph_loop = graph_loop
        self.reporter = reporter
        self.tool_manager = tool_manager
        self.state_cls = state_cls

        self.config = self._coerce_config(config)
        self.experiment_output_dir = self._resolve_experiment_dir()
        self.artifact_store = self._resolve_artifact_store()
