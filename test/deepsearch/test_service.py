import asyncio
import json
from typing import Any, Dict, List, Sequence

import pytest

from application.rag_inference.deepsearch.service import DeepSearchService
from encapsulation.data_model.deepsearch import GraphQueryContext, ThinkNote, ToolResultPayload
from core.graph_adapter.base import GraphAccessScope


def _default_service_config(*, tmp_path, fingerprint: str, **overrides):  # noqa: ANN001
    base = {
        "fingerprint": fingerprint,
        "artifact_dir": str(tmp_path / "artifacts"),
        "experiment_output_dir": None,
        "coverage_expected_min_chunks": 1,
        "think_tool": "think",
        "quality_loop": {
            "enabled": False,
            "max_rounds": 1,
            "min_citation_sentence_coverage": 1.0,
            "require_consistency": False,
            "max_uncited_sentences": 0,
            "max_actions": 0,
            "enable_llm_judge": True,
            "judge_temperature": 0.0,
            "judge_max_retries": 1,
            "judge_max_evidence_items": 1,
            "judge_max_evidence_chars": 200,
        },
    }
    base.update(overrides)
    return base


class _StubGraphLoop:
    def __init__(self):
        self.calls: List[str] = []

    async def run_think_loop(
        self,
        question: str,
        *,
        graph_context: GraphQueryContext,
        initial_think_notes=None,
        seed_evidences=None,
        plan_items=None,
    ):
        scope = graph_context.resolve_scope()
        self.calls.append(scope.scope_id if scope else "missing")
        return {
            "question": question,
            "graph_context": graph_context.model_dump(exclude_none=True),
            "adapter_metadata": {},
            "plan_steps": [],
            "graph_traversals": [],
            "reasoning_steps": [],
            "evidences": [],
            "tool_results": [],
            "think_notes": [],
            "coverage_metrics": {},
            "runtime_plan": {"items": list(plan_items or []), "markdown": "", "version": 0},
        }


class _StubGraphLoopWithWorkerError(_StubGraphLoop):
    async def run_think_loop(
        self,
        question: str,
        *,
        graph_context: GraphQueryContext,
        initial_think_notes=None,
        seed_evidences=None,
        plan_items=None,
    ):
        trace = await super().run_think_loop(
            question,
            graph_context=graph_context,
            initial_think_notes=initial_think_notes,
            seed_evidences=seed_evidences,
            plan_items=plan_items,
        )
        trace["coverage_metrics"] = {
            "worker_error_count": 1,
            "worker_errors": [{"agent_id": "worker_01", "error": "worker_timeout"}],
        }
        return trace


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
    def __init__(self, *, llm_connector):
        self.calls = 0
        self.llm_connector = llm_connector

    def compose(self, reasoning_trace: Dict[str, Any], external_evidence=None):
        self.calls += 1
        evidences = list(reasoning_trace.get("evidences", []))
        if external_evidence:
            evidences.extend(external_evidence)

        if self.calls == 1:
            structured_report = {
                "title": "stub",
                "short_answer": "This sentence makes a concrete factual claim without citations. "
                "This second sentence also lacks citations and should trigger the quality gate.",
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
                "short_answer": "This sentence is supported by evidence [ev2]. "
                "This second sentence is also supported [ev1].",
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
    async def invoke(self, tool_name: str, *, payload: Dict[str, Any]):  # noqa: ANN001
        note = ThinkNote(
            plan_step_id=payload.get("plan_step"),
            reasoning="initial think stub",
            metadata={
                "raw": {
                    "reasoning": "initial think stub",
                    "tool_calls": [],
                    "plan": [],
                    "report_needed": True,
                }
            },
        )
        return ToolResultPayload(
            tool_name=tool_name,
            namespace="stub::think",
            channel="graph",
            profile="H",
            determinism="llm_heavy",
            summary="initial think stub",
            evidences=[],
            diagnostics={},
            think_notes=[note],
        )


@pytest.mark.asyncio
async def test_service_converts_owner_to_scope(tmp_path):
    graph_loop = _StubGraphLoop()
    service = DeepSearchService(
        graph_loop=graph_loop,
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config=_default_service_config(tmp_path=tmp_path, fingerprint="service-test"),
    )

    metadata = {"priority": "urgent"}
    result = await service.run("Explain HippoRAG impact", owner_id="tenant-123", metadata=metadata)

    assert graph_loop.calls == ["tenant-123"]
    reasoning_scope = (
        result["reasoning"]["graph_context"].get("access_scope", {}).get("scope_id")
    )
    assert reasoning_scope == "tenant-123"
    assert result["report"]["answer"] == "stub"
    snapshot = result["state"]
    assert snapshot["stage"] in {"reported", "done"}
    assert snapshot.get("request_metadata") == metadata
    graph_metadata = result["reasoning"]["graph_context"].get("metadata", {})
    assert graph_metadata.get("request_metadata") == metadata


@pytest.mark.asyncio
async def test_service_persists_experiment_snapshot(tmp_path):
    graph_loop = _StubGraphLoop()
    service = DeepSearchService(
        graph_loop=graph_loop,
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config=_default_service_config(tmp_path=tmp_path, fingerprint="experiment-test", experiment_output_dir=str(tmp_path)),
    )

    await service.run("Run experiment", owner_id="tenant-321")

    files = list(tmp_path.glob("*.json"))
    assert files, "Experiment snapshot should be persisted"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["question"] == "Run experiment"
    assert payload.get("plan_id") is None


@pytest.mark.asyncio
async def test_service_surfaces_worker_failures_into_state_errors(tmp_path):
    graph_loop = _StubGraphLoopWithWorkerError()
    service = DeepSearchService(
        graph_loop=graph_loop,
        reporter=_StubReporter(),
        tool_manager=_StubToolManager(),
        config=_default_service_config(tmp_path=tmp_path, fingerprint="service-worker-errors"),
    )

    result = await service.run("Explain HippoRAG impact", owner_id="tenant-123")

    errors = result["state"].get("errors") or []
    assert errors, "worker errors should be surfaced into state.errors"
    assert any(entry.get("stage") == "graph_reasoning" for entry in errors if isinstance(entry, dict))


class _GraphLoopTwoPasses:
    def __init__(self):
        self.calls: List[Sequence[Dict[str, Any]]] = []

    async def run_think_loop(
        self,
        question: str,
        *,
        graph_context: GraphQueryContext,
        initial_think_notes=None,
        seed_evidences=None,
        plan_items=None,
    ):
        self.calls.append(list(initial_think_notes or []))
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
            "plan_steps": [],
            "graph_traversals": [],
            "reasoning_steps": [],
            "evidences": evidences,
            "tool_results": [],
            "think_notes": [],
            "coverage_metrics": {},
            "runtime_plan": {"items": list(plan_items or []), "markdown": "", "version": 0},
        }


@pytest.mark.asyncio
async def test_service_quality_loop_triggers_followup_round(tmp_path):
    class _QualityGateLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "pass": False,
                        "overall": 0.2,
                        "scores": {
                            "factual_accuracy": 0.5,
                            "citation_accuracy": 0.0,
                            "completeness": 0.6,
                            "source_quality": 0.8,
                        },
                        "reasons": ["Missing citations in short_answer/sections."],
                        "missing_topics": [],
                        "missing_claims": ["Provide cited support for the key claims."],
                        "next_actions": [
                            {
                                "action": "graph_search",
                                "query": "ev2",
                                "rationale": "Retrieve additional authoritative evidence to cite.",
                                "priority": 1,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "pass": True,
                    "overall": 0.9,
                    "scores": {
                        "factual_accuracy": 0.9,
                        "citation_accuracy": 0.9,
                        "completeness": 0.9,
                        "source_quality": 0.9,
                    },
                    "reasons": [],
                    "missing_topics": [],
                    "missing_claims": [],
                    "next_actions": [],
                },
                ensure_ascii=False,
            )

    graph_loop = _GraphLoopTwoPasses()
    llm = _QualityGateLLM()
    reporter = _QualityLoopReporter(llm_connector=llm)
    service = DeepSearchService(
        graph_loop=graph_loop,
        reporter=reporter,
        tool_manager=_StubToolManager(),
        config=_default_service_config(
            tmp_path=tmp_path,
            fingerprint="service-quality-loop",
            quality_loop={
                "enabled": True,
                "max_rounds": 2,
                "min_citation_sentence_coverage": 0.8,
                "require_consistency": False,
                "max_uncited_sentences": 6,
                "max_actions": 6,
                "enable_llm_judge": True,
                "judge_temperature": 0.0,
                "judge_max_retries": 1,
                "judge_max_evidence_items": 5,
                "judge_max_evidence_chars": 200,
            },
        ),
    )

    result = await service.run("Need citations", owner_id="tenant-ql")

    assert len(graph_loop.calls) == 2, "quality loop should trigger a follow-up reasoning round"
    gates = result["state"].get("quality_gates") or []
    assert len(gates) == 2
    assert gates[0].get("passed") is False
    assert gates[1].get("passed") is True


@pytest.mark.asyncio
async def test_service_quality_loop_caps_should_iterate_when_max_rounds_reached(tmp_path):
    class _QualityGateLLMAlwaysFail:
        def chat(self, messages, **kwargs):  # noqa: ANN001
            return json.dumps(
                {
                    "pass": False,
                    "overall": 0.2,
                    "scores": {
                        "factual_accuracy": 0.5,
                        "citation_accuracy": 0.0,
                        "completeness": 0.6,
                        "source_quality": 0.8,
                    },
                    "reasons": ["Missing citations."],
                    "missing_topics": [],
                    "missing_claims": ["Provide cited support for the key claims."],
                    "next_actions": [
                        {
                            "action": "graph_search",
                            "query": "ev2",
                            "rationale": "Retrieve additional authoritative evidence to cite.",
                            "priority": 1,
                        }
                    ],
                },
                ensure_ascii=False,
            )

    class _ReporterAlwaysUncited(_QualityLoopReporter):
        def compose(self, reasoning_trace: Dict[str, Any], external_evidence=None):
            self.calls += 1
            evidences = list(reasoning_trace.get("evidences", []))
            if external_evidence:
                evidences.extend(external_evidence)
            structured_report = {
                "title": "stub",
                "short_answer": "This sentence makes a concrete factual claim without citations.",
                "summary": "This sentence makes a concrete factual claim without citations.",
                "sections": [
                    {
                        "title": "Findings",
                        "body_markdown": "A long, uncited factual claim appears here and should be repaired by follow-up retrieval.",
                    }
                ],
            }
            return {
                "answer": "stub",
                "evidences": evidences,
                "structured_report": structured_report,
                "metadata": {},
            }

    graph_loop = _GraphLoopTwoPasses()
    llm = _QualityGateLLMAlwaysFail()
    reporter = _ReporterAlwaysUncited(llm_connector=llm)
    service = DeepSearchService(
        graph_loop=graph_loop,
        reporter=reporter,
        tool_manager=_StubToolManager(),
        config=_default_service_config(
            tmp_path=tmp_path,
            fingerprint="service-quality-loop-cap",
            quality_loop={
                "enabled": True,
                "max_rounds": 2,
                "min_citation_sentence_coverage": 0.8,
                "require_consistency": False,
                "max_uncited_sentences": 6,
                "max_actions": 6,
                "enable_llm_judge": True,
                "judge_temperature": 0.0,
                "judge_max_retries": 1,
                "judge_max_evidence_items": 5,
                "judge_max_evidence_chars": 200,
            },
        ),
    )

    result = await service.run("Need citations", owner_id="tenant-ql-cap")

    gates = result["state"].get("quality_gates") or []
    assert len(gates) == 2
    assert gates[-1].get("passed") is False
    assert gates[-1].get("should_iterate") is False
    diagnostics = gates[-1].get("diagnostics") or {}
    assert diagnostics.get("termination_reason") == "max_rounds_reached"
    assert diagnostics.get("max_rounds") == 2
