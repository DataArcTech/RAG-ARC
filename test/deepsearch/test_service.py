import asyncio
import json
from typing import Any, Dict, List

import pytest

from application.rag_inference.deepsearch.service import DeepSearchService
from encapsulation.data_model.deepsearch import GraphQueryContext, ThinkNote, ToolResultPayload


def _default_service_config(*, tmp_path, fingerprint: str, **overrides):  # noqa: ANN001
    base = {
        "fingerprint": fingerprint,
        "artifact_dir": str(tmp_path / "artifacts"),
        "experiment_output_dir": None,
        "coverage_expected_min_chunks": 1,
        "think_tool": "think",
    }
    base.update(overrides)
    return base


class _StubGraphLoop:
    def __init__(self):
        self.calls: List[str] = []

    @staticmethod
    def _stub_page_evidence() -> List[Dict[str, Any]]:
        # DeepSearch report generation requires at least one read.pages evidence (full-page context).
        # Keep the stub minimal: tests here validate scope wiring, not retrieval quality.
        return [
            {
                "chunk_id": "stub:read.pages:1",
                "source": "read.pages",
                "content": "stub page evidence",
                "kind": "primary",
                "score": None,
                "provenance": {"page_start": 1, "page_end": 1, "source_file_id": "stub-file"},
            }
        ]

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
            "evidences": self._stub_page_evidence(),
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
