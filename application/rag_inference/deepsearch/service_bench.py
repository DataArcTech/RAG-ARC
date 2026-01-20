"""Benchmark-mode (experiment) DeepSearch runner.

Kept separate from `application/rag_inference/deepsearch/service.py` so benchmark runs
can bypass product features (report/quality loops) without impacting
normal service behavior.
"""
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.report.bench_answer import synthesize_benchmark_answer
from config.benchmark_mode import benchmark_mode_enabled


@dataclass(frozen=True, slots=True)
class DeepSearchBenchResult:
    answer: str
    plan: Dict[str, Any]
    reasoning: Dict[str, Any]
    bench_report: Dict[str, Any]


def _filter_bench_plan_steps(steps: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    filtered: list[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        channel = str(step.get("channel") or "graph").strip().lower()
        if channel == "web":
            continue
        filtered.append(step)
    return filtered


class DeepSearchBenchService:
    """Non-streaming benchmark runner built from an existing DeepSearchService instance."""

    def __init__(self, deepsearch_service: Any) -> None:
        self._service = deepsearch_service

    async def run(
        self,
        question: str,
        *,
        owner_id: str,
        access_scope: Optional[GraphAccessScope] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeepSearchBenchResult:
        if not benchmark_mode_enabled():
            raise RuntimeError("bench_mode=1 is required to run DeepSearchBenchService")

        text = str(question or "").strip()
        if not text:
            raise ValueError("question must be a non-empty string")

        scope = access_scope
        if scope is None:
            try:
                uuid.UUID(str(owner_id))
            except Exception as exc:  # noqa: BLE001
                raise ValueError("owner_id must be a valid UUID string") from exc
            scope = GraphAccessScope(scope_id=str(owner_id), scope_type="owner")

        plan_result = await self._service.planner.build_plan(text, access_scope=scope)
        plan_payload = plan_result.get("plan") if isinstance(plan_result, dict) else None
        if not isinstance(plan_payload, dict):
            raise RuntimeError("planner returned invalid plan payload")

        steps = plan_payload.get("steps") or []
        if not isinstance(steps, list) or not steps:
            raise RuntimeError("planner returned an empty plan (no fallback execution allowed)")
        bench_steps = _filter_bench_plan_steps(steps)
        if not bench_steps:
            raise RuntimeError("planner returned only external/web steps; benchmark mode refuses fallback")

        graph_context_payload = plan_payload.get("graph_context") or {}
        if not isinstance(graph_context_payload, dict):
            graph_context_payload = {}
        graph_context = GraphQueryContext(**graph_context_payload)
        if metadata:
            try:
                graph_context.metadata.setdefault("request_metadata", {}).update(metadata)
            except Exception:
                pass

        reasoning = await self._service.graph_loop.run(text, bench_steps, graph_context=graph_context)
        llm_connector = getattr(self._service.planner, "llm_connector", None) or getattr(self._service.reporter, "llm_connector", None)

        reporter = getattr(self._service, "reporter", None)
        bench_cfg = None
        if reporter is not None:
            cfg = getattr(reporter, "config", None)
            if isinstance(cfg, dict):
                bench_cfg = cfg.get("bench_answer")

        question_type = None
        if isinstance(metadata, dict):
            question_type = metadata.get("question_type") or metadata.get("type")

        # Legacy fallbacks for older configs/tests: keep using reporter budgets when bench_answer config is absent.
        max_items = getattr(reporter, "max_evidence_items", None) if reporter is not None else None
        max_chars = getattr(reporter, "report_max_evidence_chars", None) if reporter is not None else None
        report = await synthesize_benchmark_answer(
            llm_connector=llm_connector,
            question=text,
            reasoning_trace=reasoning,
            external_evidence=None,
            question_type=str(question_type) if question_type else None,
            bench_answer_config=bench_cfg if isinstance(bench_cfg, dict) else None,
            max_evidence_items=max_items if isinstance(max_items, int) else None,
            max_evidence_chars=max_chars if isinstance(max_chars, int) else None,
        )
        try:
            reasoning = dict(reasoning or {})
            reasoning["bench_report"] = dict(report or {})
        except Exception:
            pass
        return DeepSearchBenchResult(
            answer=str((report or {}).get("answer") or "").strip(),
            plan=plan_payload,
            reasoning=reasoning,
            bench_report=dict(report or {}),
        )
