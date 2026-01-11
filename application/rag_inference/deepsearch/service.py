"""Service façade wiring DeepSearch planner, reasoning loop, gap detection, and reporting."""

import logging
from typing import Any, Dict, Type

from core.deepsearch.gap import GapDetectionEngine
from core.deepsearch.plan import DeepSearchPlanner
from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.state import DeepSearchState
from core.deepsearch.tooling.protocols import ToolInvoker
from encapsulation.deepsearch.external import ExternalSearchChannel

from .service_runtime import (
    DeepSearchServiceArtifactsMixin,
    DeepSearchServiceContextMixin,
    DeepSearchServiceExternalMixin,
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
    DeepSearchServiceExternalMixin,
    DeepSearchServiceInitialThinkMixin,
):
    """Application-layer facade with a shared async run() entry point for FastAPI/CLI/MCP."""

    def __init__(
        self,
        planner: DeepSearchPlanner,
        graph_loop: GraphReasoningLoop,
        gap_detector: GapDetectionEngine,
        reporter: DeepSearchReporter,
        tool_manager: ToolInvoker,
        *,
        external_channel: ExternalSearchChannel | None = None,
        state_cls: Type[DeepSearchState] = DeepSearchState,
        config: Dict[str, Any] | None = None,
    ):
        self.planner = planner
        self.graph_loop = graph_loop
        self.gap_detector = gap_detector
        self.reporter = reporter
        self.tool_manager = tool_manager
        self.external_channel = external_channel
        self.state_cls = state_cls

        self.config = self._coerce_config(config)
        self.experiment_output_dir = self._resolve_experiment_dir()
        self.artifact_store = self._resolve_artifact_store()

