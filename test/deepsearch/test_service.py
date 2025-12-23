import asyncio
import json
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


class _StubGraphLoopWithWorkerError(_StubGraphLoop):
    async def run(self, question: str, plan_steps: Sequence[Dict[str, Any]], *, graph_context: GraphQueryContext):
        trace = await super().run(question, plan_steps, graph_context=graph_context)
        trace["coverage_metrics"] = {
            "worker_error_count": 1,
            "worker_errors": [{"agent_id": "worker_01", "error": "worker_timeout"}],
        }
        return trace


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
        evidences = list(reasoning_trace.get("evidences", []))
        if external_evidence:
            evidences.extend(external_evidence)
        return {
            "answer": "stub",
            "evidences": evidences,
            "metadata": reasoning_trace.get("coverage_metrics", {}),
        }


class _QualityLoopReporter:
    def __init__(self):
        self.calls = 0

    def compose(self, reasoning_trace: Dict[str, Any], external_evidence=None):
        self.calls += 1
        evidences = list(reasoning_trace.get("evidences", []))
        if external_evidence:
            evidences.extend(external_evidence)

        if self.calls == 1:
            structured_report = {
                "title": "stub",
                "summary": "This sentence makes a concrete factual claim without citations. "
                "This second sentence also lacks citations and should trigger the quality gate.",
                "sections": [
                    {
                        "title": "Findings",
                        "body_markdown": "A long, uncited factual claim appears here and should be repaired by follow-up retrieval.",
                    }
                ],
            }
        else:
            structured_report = {
                "title": "stub",
                "summary": "This sentence is supported by evidence [ev2]. "
                "This second sentence is also supported [ev1].",
                "sections": [
                    {
                        "title": "Findings",
                        "body_markdown": "Now the claim is properly cited. [ev2]",
                    }
                ],
            }

        return {
            "answer": "stub",
            "evidences": evidences,
            "structured_report": structured_report,
            "metadata": {},
        }


class _StubToolManager:
    pass


class _TriggerGapDetector:
    def evaluate(self, reasoning_trace: Dict[str, Any]):
        return {
            "coverage_score": 0.2,
            "confidence_score": 0.3,
            "should_trigger_external": True,
            "reason": "coverage",
            "missing_topics": ["timeline"],
            "diagnostics": {},
        }


class _StubExternalChannel:
    def __init__(self):
        self.calls: List[List[Dict[str, Any]]] = []

    async def run(self, tasks, **kwargs):
        self.calls.append(tasks)
        return {
            "evidences": [
                {"chunk_id": "ext-1", "source": "web.stub", "content": "external evidence"},
            ],
            "logs": [
                {"step_id": tasks[0]["step_id"], "provider": "stub", "status": "ok"},
            ],
        }


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

    metadata = {"priority": "urgent"}
    result = await service.run("Explain HippoRAG impact", owner_id="tenant-123", metadata=metadata)

    assert planner.scopes and planner.scopes[0].scope_id == "tenant-123"
    assert graph_loop.calls == ["tenant-123"]
    reasoning_scope = (
        result["reasoning"]["graph_context"].get("access_scope", {}).get("scope_id")
    )
    assert reasoning_scope == "tenant-123"
    assert result["report"]["answer"] == "stub"
    snapshot = result["state"]
    assert snapshot["stage"] == "reported"
    assert snapshot.get("request_metadata") == metadata
    graph_metadata = result["reasoning"]["graph_context"].get("metadata", {})
    assert graph_metadata.get("request_metadata") == metadata


@pytest.mark.asyncio
async def test_service_persists_experiment_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_EXPERIMENT_OUTPUT_DIR", str(tmp_path))
    planner = _StubPlanner()
    graph_loop = _StubGraphLoop()
    service = DeepSearchService(
        planner=planner,
        graph_loop=graph_loop,
        gap_detector=_StubGapDetector(),
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config={"fingerprint": "experiment-test"},
    )

    await service.run("Run experiment", owner_id="tenant-321")

    files = list(tmp_path.iterdir())
    assert files, "Experiment snapshot should be persisted"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["question"] == "Run experiment"
    assert payload["plan_id"] == "plan-test"


@pytest.mark.asyncio
async def test_service_creates_external_task_when_gap_detected():
    planner = _StubPlanner()
    graph_loop = _StubGraphLoop()
    external_channel = _StubExternalChannel()
    service = DeepSearchService(
        planner=planner,
        graph_loop=graph_loop,
        gap_detector=_TriggerGapDetector(),
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        external_channel=external_channel,
        config={"fingerprint": "service-gap"},
    )

    result = await service.run("Need more info", owner_id="tenant-gap")

    assert external_channel.calls, "external channel should be invoked when gap is detected"
    synthesized = external_channel.calls[0]
    assert synthesized and synthesized[0]["step_id"].startswith("gap_web_")
    assert result["report"]["evidences"], "external evidences should feed into report output"
    snapshot = result["state"]
    assert snapshot["external_calls"], "state snapshot should contain external call logs"


@pytest.mark.asyncio
async def test_service_surfaces_worker_failures_into_state_errors():
    planner = _StubPlanner()
    graph_loop = _StubGraphLoopWithWorkerError()
    service = DeepSearchService(
        planner=planner,
        graph_loop=graph_loop,
        gap_detector=_StubGapDetector(),
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config={"fingerprint": "service-worker-errors"},
    )

    result = await service.run("Explain HippoRAG impact", owner_id="tenant-123")

    errors = result["state"].get("errors") or []
    assert errors, "worker errors should be surfaced into state.errors"
    assert any(entry.get("stage") == "graph_reasoning" for entry in errors if isinstance(entry, dict))


class _GraphLoopTwoPasses:
    def __init__(self):
        self.calls: List[Sequence[Dict[str, Any]]] = []

    async def run(
        self,
        question: str,
        plan_steps: Sequence[Dict[str, Any]],
        *,
        graph_context: GraphQueryContext,
    ):
        self.calls.append(list(plan_steps))
        if len(self.calls) == 1:
            evidences = [{"chunk_id": "ev1", "source": "hipporag", "content": "evidence one"}]
        else:
            evidences = [
                {"chunk_id": "ev1", "source": "hipporag", "content": "evidence one"},
                {"chunk_id": "ev2", "source": "hipporag", "content": "evidence two"},
            ]
        return {
            "question": question,
            "graph_context": graph_context.model_dump(exclude_none=True),
            "adapter_metadata": {},
            "plan_steps": list(plan_steps),
            "graph_traversals": [],
            "reasoning_steps": [],
            "evidences": evidences,
            "tool_results": [],
            "pending_external": [],
            "think_notes": [],
            "coverage_metrics": {},
        }


@pytest.mark.asyncio
async def test_service_quality_loop_triggers_followup_round(monkeypatch):
    monkeypatch.setenv("DEEPSEARCH_QUALITY_LOOP_ENABLED", "true")
    monkeypatch.setenv("DEEPSEARCH_QUALITY_LOOP_MAX_ROUNDS", "2")
    monkeypatch.setenv("DEEPSEARCH_QUALITY_LOOP_MIN_CITATION_SENTENCE_COVERAGE", "0.8")

    planner = _StubPlanner()
    graph_loop = _GraphLoopTwoPasses()
    reporter = _QualityLoopReporter()
    service = DeepSearchService(
        planner=planner,
        graph_loop=graph_loop,
        gap_detector=_StubGapDetector(),
        reporter=reporter,
        tool_manager=_StubToolManager(),
        config={"fingerprint": "service-quality-loop"},
    )

    result = await service.run("Need citations", owner_id="tenant-ql")

    assert len(graph_loop.calls) == 2, "quality loop should trigger a follow-up reasoning round"
    gates = result["state"].get("quality_gates") or []
    assert len(gates) == 2
    assert gates[0].get("passed") is False
    assert gates[1].get("passed") is True
