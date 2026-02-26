import uuid

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ThinkNote, ToolResultPayload
from core.deepsearch.reasoning.graph_loop import GraphReasoningLoop
from core.graph_adapter.base import GraphAccessScope


class _DummyAdapter:
    async def prepare(self, question: str, *, access_scope: GraphAccessScope | None = None) -> None:  # pragma: no cover
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope: GraphAccessScope | None = None, query_options=None):  # pragma: no cover
        return {}

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope: GraphAccessScope | None = None):  # pragma: no cover
        return data

    async def summarize(self, channel: str, data, *, access_scope: GraphAccessScope | None = None) -> str:  # pragma: no cover
        return ""

    async def chain_traverse(self, strategy, *, access_scope: GraphAccessScope | None = None):  # pragma: no cover
        return {}

    def metadata(self):
        return {"adapter_name": "dummy"}


class _FakeToolInvoker:
    def __init__(self, think_payloads: list[ToolResultPayload]):
        self._think_payloads = list(think_payloads)
        self.invocations: list[str] = []

    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        self.invocations.append(tool_name)
        if tool_name == "think":
            if not self._think_payloads:
                raise AssertionError("think invoked more times than expected")
            return self._think_payloads.pop(0)
        if tool_name == "code.python":
            ev = EvidenceChunk(
                kind="derived",
                chunk_id=f"ev_compute_{uuid.uuid4().hex}",
                source="code.python",
                content="result=42",
                score=None,
                provenance={},
            )
            return ToolResultPayload(
                tool_name="code.python",
                namespace="test",
                channel="graph",
                profile="H",
                determinism="deterministic",
                summary="computed",
                evidences=[ev],
                diagnostics={},
                think_notes=[],
            )
        raise AssertionError(f"unexpected tool invocation: {tool_name}")


def _think_payload(*, tool_calls: list[dict], is_final: bool | None) -> ToolResultPayload:
    raw = {
        "reasoning": "next",
        "tool_calls": tool_calls,
        "plan": [{"text": "step", "checked": False}],
    }
    if is_final is not None:
        raw["is_final"] = is_final
    note = ThinkNote(plan_step_id="think", reasoning="note", metadata={"raw": raw})
    return ToolResultPayload(
        tool_name="think",
        namespace="test",
        channel="graph",
        profile="H",
        determinism="llm_heavy",
        summary="think",
        evidences=[],
        diagnostics={},
        think_notes=[note],
    )


@pytest.mark.asyncio
async def test_computable_gate_requires_code_python_before_stopping() -> None:
    # Seed primary evidence so the existing evidence hard gate is satisfied.
    seed_evidences = [
        EvidenceChunk(
            kind="primary",
            chunk_id="ev_read_pages",
            source="read.pages",
            content="EBITDA=100; Revenue=200",
            score=1.0,
            provenance={"source_file_id": str(uuid.uuid4()), "page_start": 0, "page_end": 0},
        )
    ]
    context = GraphQueryContext(
        adapter_name="dummy",
        question="q",
        metadata={
            "report_needed": True,
            "report_style": "deepsearch",
            "query_spec": {"is_computable": True},
        },
        access_scope=GraphAccessScope(scope_id="test"),
    )

    tool_manager = _FakeToolInvoker(
        think_payloads=[
            # Round 1: no tool calls → should be blocked by computable gate.
            _think_payload(tool_calls=[], is_final=None),
            # Round 2: propose code.python → produces compute evidence.
            _think_payload(
                tool_calls=[
                    {
                        "tool_name": "code.python",
                        "tool_args": {"code": "result=42", "inputs": {}},
                        "rationale": "compute",
                        "parallelizable": False,
                    }
                ],
                is_final=None,
            ),
            # Round 3: finalization allowed because compute evidence exists.
            _think_payload(tool_calls=[], is_final=True),
        ]
    )
    loop = GraphReasoningLoop(
        adapter=_DummyAdapter(),
        llm_connector=None,
        strategy_config={
            "tool_timeout_seconds": 0,
            "coverage_expected_min_chunks": 1,
            "tool_context_max_evidences": 0,
            "think": {
                "tool_name": "think",
                "include_llm_tools": False,
                "every_n_steps": 0,
                "min_coverage": 0.0,
                "always_run": False,
                "enable_tool_calls": True,
                "max_tool_calls": 4,
                "tool_call_concurrency": 1,
                "tool_catalog_max_items": 50,
                "tool_catalog_allowlist": None,
                "recent_tool_runs_max": 0,
                "max_rounds_per_checkpoint": 3,
            },
        },
        tool_manager=tool_manager,
    )

    out = await loop.run_think_loop("q", graph_context=context, seed_evidences=seed_evidences)
    sources = [row.get("source") for row in (out.get("evidences") or []) if isinstance(row, dict)]
    assert "code.python" in sources
    assert tool_manager.invocations.count("think") == 3
    assert tool_manager.invocations.count("code.python") == 1


@pytest.mark.asyncio
async def test_non_computable_question_can_stop_without_code_python() -> None:
    seed_evidences = [
        EvidenceChunk(
            kind="primary",
            chunk_id="ev_read_pages",
            source="read.pages",
            content="policy terms",
            score=1.0,
            provenance={"source_file_id": str(uuid.uuid4()), "page_start": 0, "page_end": 0},
        )
    ]
    context = GraphQueryContext(
        adapter_name="dummy",
        question="q",
        metadata={
            "report_needed": True,
            "report_style": "deepsearch",
            "query_spec": {"is_computable": False},
        },
        access_scope=GraphAccessScope(scope_id="test"),
    )

    tool_manager = _FakeToolInvoker(think_payloads=[_think_payload(tool_calls=[], is_final=None)])
    loop = GraphReasoningLoop(
        adapter=_DummyAdapter(),
        llm_connector=None,
        strategy_config={
            "tool_timeout_seconds": 0,
            "coverage_expected_min_chunks": 1,
            "tool_context_max_evidences": 0,
            "think": {
                "tool_name": "think",
                "include_llm_tools": False,
                "every_n_steps": 0,
                "min_coverage": 0.0,
                "always_run": False,
                "enable_tool_calls": True,
                "max_tool_calls": 2,
                "tool_call_concurrency": 1,
                "tool_catalog_max_items": 50,
                "tool_catalog_allowlist": None,
                "recent_tool_runs_max": 0,
                "max_rounds_per_checkpoint": 1,
            },
        },
        tool_manager=tool_manager,
    )

    out = await loop.run_think_loop("q", graph_context=context, seed_evidences=seed_evidences)
    sources = [row.get("source") for row in (out.get("evidences") or []) if isinstance(row, dict)]
    assert "code.python" not in sources
    assert tool_manager.invocations == ["think"]

