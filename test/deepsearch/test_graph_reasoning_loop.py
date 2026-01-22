import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ThinkNote, ToolResultPayload
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.reasoning import GraphReasoningLoop


class _StubAdapter:
    def metadata(self):  # noqa: D401
        return {"adapter_name": "stub"}


class _StubToolManager:
    def __init__(self, think_payloads: list[dict]):
        self._think_payloads = list(think_payloads)
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        self.calls.append((tool_name, payload))
        if tool_name == "think":
            raw = self._think_payloads.pop(0)
            note = ThinkNote(
                plan_step_id=payload.get("plan_step"),
                reasoning=str(raw.get("reasoning") or ""),
                metadata={"raw": raw},
            )
            return ToolResultPayload(
                tool_name="think",
                namespace="stub::think",
                channel="graph",
                profile="H",
                determinism="llm_heavy",
                summary=str(raw.get("reasoning") or ""),
                evidences=[],
                diagnostics={},
                think_notes=[note],
            )
        if tool_name == "explore":
            chunk = EvidenceChunk(chunk_id="e1", source="explore", content="evidence")
            return ToolResultPayload(
                tool_name="explore",
                namespace="stub::explore",
                channel="graph",
                profile="X",
                determinism="hybrid",
                summary="explore ok",
                evidences=[chunk],
                diagnostics={},
                think_notes=[],
            )
        if tool_name == "code.python":
            chunk = EvidenceChunk(chunk_id="c1", source="code.python", content="result=3")
            return ToolResultPayload(
                tool_name="code.python",
                namespace="stub::code",
                channel="graph",
                profile="F",
                determinism="deterministic",
                summary="code ok",
                evidences=[chunk],
                diagnostics={},
                think_notes=[],
            )
        raise RuntimeError(f"Unexpected tool: {tool_name}")


def _strategy_config(*, think_overrides: dict | None = None) -> dict:
    base = {
        "strategy_name": "ppr_chain",
        "allow_semantic_channel": True,
        "chain_depth": 1,
        "parallel_branches": 1,
        "step_summary_max_chars": 2000,
        "tool_context_max_evidences": 5,
        "tool_context_max_chars": 800,
        "coverage_expected_min_chunks": 1,
        "trace_reflection_enabled": False,
        "trace_reflection_max": 0,
        "tool_timeout_seconds": 0.0,
        "think": {
            "tool_name": "think",
            "every_n_steps": 1,
            "min_coverage": 0.0,
            "always_run": True,
            "enable_tool_calls": True,
            "max_tool_calls": 2,
            "tool_call_concurrency": 2,
            "tool_catalog_max_items": 6,
            "include_llm_tools": True,
            "max_rounds_per_checkpoint": 2,
        },
    }
    if think_overrides:
        merged = dict(base["think"])
        merged.update(dict(think_overrides))
        base["think"] = merged
    return base


@pytest.mark.asyncio
async def test_think_loop_executes_tool_calls_and_updates_plan() -> None:
    think_payloads = [
        {
            "reasoning": "Need evidence; will explore.",
            "tool_calls": [
                {"tool_name": "explore", "tool_args": {"actions": [{"tool": "search"}]}, "rationale": "fetch data", "parallelizable": True}
            ],
            "plan": [{"text": "Collect evidence", "checked": False}],
        },
        {
            "reasoning": "Evidence ready.",
            "tool_calls": [],
            "plan": [{"text": "Collect evidence", "checked": True}],
        },
    ]
    tool_manager = _StubToolManager(think_payloads)
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
    )

    result = await loop.run_think_loop("Q", graph_context=context)

    tool_names = [name for name, _ in tool_manager.calls]
    assert tool_names.count("think") == 2
    assert "explore" in tool_names
    assert any(ev.get("chunk_id") == "e1" for ev in result.get("evidences") or [])

    plan = result.get("runtime_plan") or {}
    assert plan.get("items") == [{"text": "Collect evidence", "checked": True}]
    assert "- [x] Collect evidence" in (plan.get("markdown") or "")


@pytest.mark.asyncio
async def test_think_loop_skips_unknown_tools() -> None:
    think_payloads = [
        {
            "reasoning": "Try an unsupported tool.",
            "tool_calls": [
                {"tool_name": "search", "tool_args": {"focus_query": "x"}, "rationale": "should be gated", "parallelizable": True}
            ],
            "plan": [{"text": "Probe", "checked": False}],
        },
        {
            "reasoning": "No more actions.",
            "tool_calls": [],
            "plan": [{"text": "Probe", "checked": True}],
        },
    ]
    tool_manager = _StubToolManager(think_payloads)
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "tool_catalog_allowlist": ["explore", "code.python"],
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
    )

    result = await loop.run_think_loop("Q", graph_context=context)

    tool_names = [name for name, _ in tool_manager.calls]
    assert tool_names.count("think") == 2
    assert "search" not in tool_names
    failures = [step for step in result.get("reasoning_steps") or [] if step.get("status") == "failed"]
    assert any(step.get("diagnostics", {}).get("reason") == "unknown_tool" for step in failures)
