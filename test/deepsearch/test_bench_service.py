import uuid

import pytest

from application.rag_inference.deepsearch.service_bench import DeepSearchBenchService
from core.graph_adapter.base import GraphAccessScope


class _StubPlanner:
    def __init__(self, *, llm_connector):
        self.llm_connector = llm_connector
        self.calls = []

    async def build_plan(self, question: str, *, access_scope: GraphAccessScope):
        self.calls.append({"question": question, "scope": access_scope.scope_id})
        return {
            "plan_id": "plan-test",
            "plan": {
                "plan_id": "plan-test",
                "question": question,
                "steps": [
                    {"step_id": "plan_01", "channel": "graph", "description": "Collect evidence", "metadata": {}},
                    {"step_id": "plan_web", "channel": "web", "description": "External search", "metadata": {}},
                ],
                "graph_context": {
                    "adapter_name": "hipporag",
                    "question": question,
                    "access_scope": access_scope.__dict__,
                    "metadata": {},
                },
            },
        }


class _StubGraphLoop:
    def __init__(self):
        self.calls = []

    async def run(self, question: str, plan_steps, *, graph_context):  # noqa: ANN001
        self.calls.append({"question": question, "steps": plan_steps})
        return {"question": question, "evidences": [{"content": "e1"}], "graph_context": graph_context.model_dump(exclude_none=True)}


class _StubReporter:
    max_evidence_items = 2
    report_max_evidence_chars = 200


class _StubLLM:
    async def achat(self, messages, **kwargs):  # noqa: ANN001
        return "bench-answer"


@pytest.mark.asyncio
async def test_deepsearch_bench_service_filters_web_steps_and_synthesizes_answer(monkeypatch):
    monkeypatch.setenv("bench_mode", "1")

    llm = _StubLLM()
    planner = _StubPlanner(llm_connector=llm)
    graph_loop = _StubGraphLoop()
    reporter = _StubReporter()
    service = type("Svc", (), {"planner": planner, "graph_loop": graph_loop, "reporter": reporter})()

    bench = DeepSearchBenchService(service)
    owner_id = str(uuid.uuid4())
    result = await bench.run("Who wrote The Hobbit?", owner_id=owner_id)

    assert result.answer == "bench-answer"
    assert isinstance(result.bench_report, dict)
    assert planner.calls and planner.calls[0]["scope"] == owner_id
    assert graph_loop.calls
    used_steps = graph_loop.calls[0]["steps"]
    assert all(step.get("channel") != "web" for step in used_steps)
