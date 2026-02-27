import asyncio

import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext, ThinkNote, ToolResultPayload
from core.deepsearch.reasoning.graph_loop_runtime import GraphLoopRuntimeMixin
from core.deepsearch.reasoning.graph_loop_state import _RUN_TOOL_MEMO, _RUN_PLAN_STATE
from core.deepsearch.tooling.run_tool_memo import RunToolMemoizer
from core.deepsearch.memory.plan_state import PlanState
from core.graph_adapter.base import GraphAccessScope


class _StubToolManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        self.calls.append((tool_name, payload))
        return ToolResultPayload(
            tool_name=tool_name,
            namespace=f"stub.{tool_name}",
            channel="graph",
            profile="F",
            determinism="deterministic",
            summary="ok",
            evidences=[],
            diagnostics={"from": "stub"},
            think_notes=[],
        )


class _DummyLoop(GraphLoopRuntimeMixin):
    def __init__(self) -> None:
        self._think_config = {"max_tool_calls": 8, "tool_call_concurrency": 1}
        self.tool_manager = _StubToolManager()
        self._tool_timeout = 0.0

    async def _extend_shared_evidences(self, _additions) -> None:
        return

    def _build_tool_payload(self, *, plan_step_id, question, context, evidences, coverage_hint, extra):
        return {
            "plan_step": plan_step_id,
            "question": question,
            "context_evidences": [],
            "access_scope": context.access_scope,
            "graph_context": context.model_dump(exclude_none=True),
            "coverage_metrics": coverage_hint or {},
            "extra": extra or {},
        }


@pytest.mark.asyncio
async def test_run_scoped_tool_memoization_replays_across_checkpoints():
    loop = _DummyLoop()
    ctx = GraphQueryContext(
        adapter_name="stub",
        question="q",
        access_scope=GraphAccessScope(scope_id="owner-1", scope_type="owner"),
        metadata={"file_scope": {"filename_contains": ["x"], "source": "test"}},
    )

    memo = RunToolMemoizer(
        enabled=True,
        max_entries=16,
        include_prefixes=("locate",),
        exclude_prefixes=(),
    )
    memo_token = _RUN_TOOL_MEMO.set(memo)
    plan_token = _RUN_PLAN_STATE.set(PlanState())
    try:
        note1 = ThinkNote(
            reasoning="r1",
            metadata={
                "raw": {
                    "tool_calls": [
                        {"tool_name": "locate", "tool_args": {"query": "abc", "top_k": 3}},
                    ]
                }
            },
        )
        await loop._execute_tool_calls_from_think(
            think_step_id="think_01",
            question="q",
            context=ctx,
            evidences=[],
            coverage_metrics={},
            think_notes=[note1],
            tool_runs=[],
            think_notes_out=[],
            available_tool_names=set(),
        )

        note2 = ThinkNote(
            reasoning="r2",
            metadata={
                "raw": {
                    "tool_calls": [
                        {"tool_name": "locate", "tool_args": {"query": "abc", "top_k": 3}},
                    ]
                }
            },
        )
        tool_runs_2: list[dict] = []
        records, summary = await loop._execute_tool_calls_from_think(
            think_step_id="think_02",
            question="q",
            context=ctx,
            evidences=[],
            coverage_metrics={},
            think_notes=[note2],
            tool_runs=tool_runs_2,
            think_notes_out=[],
            available_tool_names=set(),
        )

        # Underlying tool invoked once total; second checkpoint replays from memo.
        assert len(loop.tool_manager.calls) == 1
        assert summary["proposed"] == 1
        assert records
        assert records[-1].diagnostics.get("reason") in {"tool_memoization_replay", "think_tool_call"}
        assert tool_runs_2
        diag = tool_runs_2[-1]["result"]["diagnostics"]
        assert diag.get("memoization", {}).get("hit") is True
    finally:
        _RUN_TOOL_MEMO.reset(memo_token)
        _RUN_PLAN_STATE.reset(plan_token)
