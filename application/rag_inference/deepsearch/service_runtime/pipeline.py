import logging
from typing import Any, Dict, Optional, Sequence

from encapsulation.data_model.deepsearch import GraphQueryContext
logger = logging.getLogger(__name__)


class DeepSearchServicePipelineMixin:
    async def _think_loop_stage(
        self,
        *,
        question: str,
        reasoning_context: GraphQueryContext,
        initial_think_notes: Sequence[Any] | None = None,
        seed_evidences: Sequence[Any] | None = None,
        plan_items: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        graph_loop = getattr(self, "graph_loop", None)
        runner = getattr(graph_loop, "run_think_loop", None)
        if not callable(runner):
            raise RuntimeError("graph_loop does not expose run_think_loop()")
        kwargs: Dict[str, Any] = {"graph_context": reasoning_context}
        return await runner(
            question,
            **kwargs,
            initial_think_notes=initial_think_notes,
            seed_evidences=seed_evidences,
            plan_items=plan_items,
        )

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
