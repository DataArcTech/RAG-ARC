"""Service façade wiring DeepSearch reasoning loop and reporting."""

import hashlib
import logging
from typing import Any, Dict, Type

from core.deepsearch.reasoning import GraphReasoningLoop
from core.deepsearch.report import DeepSearchReporter
from core.deepsearch.state import DeepSearchState
from core.deepsearch.tooling.protocols import ToolInvoker
from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN
from core.prompts.deepsearch.plan_templates import (
    PLAN_TEMPLATE_SELECTOR_SYSTEM_PROMPT_EN,
    PLAN_TEMPLATE_SELECTOR_USER_PROMPT_TEMPLATE_EN,
)
from core.deepsearch.planning import build_template_fingerprint

from .service_runtime import (
    DeepSearchServiceArtifactsMixin,
    DeepSearchServiceContextMixin,
    DeepSearchServiceInitialThinkMixin,
    DeepSearchServicePipelineMixin,
    DeepSearchServiceRunMixin,
    DeepSearchServiceStageMixin,
)
from .runtime_cache import DeepSearchInitialThinkCache

logger = logging.getLogger(__name__)


class DeepSearchService(
    DeepSearchServiceRunMixin,
    DeepSearchServiceStageMixin,
    DeepSearchServicePipelineMixin,
    DeepSearchServiceContextMixin,
    DeepSearchServiceArtifactsMixin,
    DeepSearchServiceInitialThinkMixin,
):
    """Application-layer facade with a shared async run() entry point for FastAPI/CLI/MCP."""

    def __init__(
        self,
        graph_loop: GraphReasoningLoop,
        reporter: DeepSearchReporter,
        tool_manager: ToolInvoker,
        *,
        state_cls: Type[DeepSearchState] = DeepSearchState,
        config: Dict[str, Any] | None = None,
    ):
        self.graph_loop = graph_loop
        self.reporter = reporter
        self.tool_manager = tool_manager
        self.state_cls = state_cls

        self.config = self._coerce_config(config)
        self.experiment_output_dir = self._resolve_experiment_dir()
        self.artifact_store = self._resolve_artifact_store()

        self._initial_think_cache = self._build_initial_think_cache()
        # Used as part of the initial think cache key so prompt/template changes auto-bust.
        prompt_blob = "\n".join(
            [
                THINK_TOOL_SYSTEM_PROMPT_EN,
                PLAN_TEMPLATE_SELECTOR_SYSTEM_PROMPT_EN,
                PLAN_TEMPLATE_SELECTOR_USER_PROMPT_TEMPLATE_EN,
                build_template_fingerprint(),
            ]
        )
        self._think_prompt_fingerprint = hashlib.sha256(prompt_blob.encode("utf-8")).hexdigest()

    def _build_initial_think_cache(self) -> DeepSearchInitialThinkCache | None:
        cfg = None
        try:
            cfg = (getattr(self, "config", None) or {}).get("plan_cache")
        except Exception:
            cfg = None
        if not isinstance(cfg, dict):
            return None
        if cfg.get("enabled") is False:
            return None
        try:
            max_entries = int(cfg.get("max_entries") or 0)
        except Exception:
            max_entries = 0
        try:
            ttl_seconds = float(cfg.get("ttl_seconds") or 0.0)
        except Exception:
            ttl_seconds = 0.0
        if max_entries <= 0 or ttl_seconds <= 0:
            # Treat invalid config as disabled; do not raise at runtime.
            return None
        return DeepSearchInitialThinkCache(max_entries=max_entries, ttl_seconds=ttl_seconds)
