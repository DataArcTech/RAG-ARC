import inspect
import logging
from typing import Any, Dict, Optional, Sequence

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope

logger = logging.getLogger(__name__)


class DeepSearchServicePipelineMixin:
    async def _plan_stage(self, *, question: str, scope: GraphAccessScope) -> Dict[str, Any]:
        planner = getattr(self, "planner", None)
        if planner is None:
            raise RuntimeError("planner is not configured")
        return await planner.build_plan(question, access_scope=scope)

    async def _reasoning_stage(
        self,
        *,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        reasoning_context: GraphQueryContext,
        settings_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        graph_loop = getattr(self, "graph_loop", None)
        runner = getattr(graph_loop, "run", None)
        if not callable(runner):
            raise RuntimeError("graph_loop does not expose an async run() method")
        kwargs: Dict[str, Any] = {"graph_context": reasoning_context}
        try:
            sig = inspect.signature(runner)
            if "settings_override" in sig.parameters and settings_override:
                kwargs["settings_override"] = settings_override
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to inspect graph_loop.run signature: %s", exc, exc_info=True)
        return await runner(question, plan_steps, **kwargs)

    def _report_stage(
        self,
        *,
        reasoning_trace: Dict[str, Any],
        external_logs: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        reporter = getattr(self, "reporter", None)
        if reporter is None:
            raise RuntimeError("reporter is not configured")
        return reporter.compose(reasoning_trace, external_logs)

