import asyncio
from typing import Any, Dict, List, Sequence

import pytest

from application.rag_inference.deepsearch.service import DeepSearchService
from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope


class _StubPlanner:
    def __init__(self):
        self.scopes: List[GraphAccessScope] = []

    async def build_plan(self, question: str, *, access_scope: GraphAccessScope):
        self.scopes.append(access_scope)
        context = GraphQueryContext(
            adapter_name="hipporag",
            question=question,
            access_scope=access_scope,
        )
        return {
            "plan_id": "plan-test",
            "plan": {
                "plan_id": "plan-test",
                "question": question,
                "steps": [],
                "graph_context": context.model_dump(exclude_none=True),
            },
        }


class _StubGraphLoop:
    def __init__(self):
        self.calls: List[str] = []

    async def run(
        self,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        *,
        graph_context: GraphQueryContext,
    ):
        scope = graph_context.resolve_scope()
        self.calls.append(scope.scope_id if scope else "missing")
        return {
            "question": question,
            "graph_context": graph_context.model_dump(exclude_none=True),
            "adapter_metadata": {},
            "plan_steps": list(plan_steps),
            "graph_traversals": [],
            "reasoning_steps": [],
            "evidences": [],
            "tool_results": [],
            "pending_external": [],
            "think_notes": [],
            "coverage_metrics": {},
        }


class _StubGapDetector:
    def evaluate(self, reasoning_trace: Dict[str, Any]):
        return {
            "coverage_score": 1.0,
            "confidence_score": 1.0,
            "should_trigger_external": False,
            "reason": "sufficient",
            "missing_topics": [],
            "diagnostics": {},
        }


class _StubReporter:
    def compose(self, reasoning_trace: Dict[str, Any], external_evidence=None):
        return {
            "answer": "stub",
            "evidences": reasoning_trace.get("evidences", []),
            "metadata": reasoning_trace.get("coverage_metrics", {}),
        }


class _StubToolManager:
    pass


@pytest.mark.asyncio
async def test_service_converts_owner_to_scope():
    planner = _StubPlanner()
    graph_loop = _StubGraphLoop()
    service = DeepSearchService(
        planner=planner,
        graph_loop=graph_loop,
        gap_detector=_StubGapDetector(),
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config={"fingerprint": "service-test"},
    )

    result = await service.run("Explain HippoRAG impact", owner_id="tenant-123")

    assert planner.scopes and planner.scopes[0].scope_id == "tenant-123"
    assert graph_loop.calls == ["tenant-123"]
    reasoning_scope = (
        result["reasoning"]["graph_context"].get("access_scope", {}).get("scope_id")
    )
    assert reasoning_scope == "tenant-123"
    assert result["report"]["answer"] == "stub"
    snapshot = result["state"]
    assert snapshot["stage"] == "reported"
