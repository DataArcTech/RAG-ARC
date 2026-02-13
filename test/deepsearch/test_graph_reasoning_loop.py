import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ThinkNote, ToolResultPayload
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.reasoning import GraphReasoningLoop
import asyncio


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


class _FileRoutingToolManager(_StubToolManager):
    def __init__(self, think_payloads: list[dict], *, routed_file_ids: list[str]):
        super().__init__(think_payloads)
        self._routed_file_ids = list(routed_file_ids)

    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        if tool_name != "explore":
            return await super().invoke(tool_name, payload=payload)

        # Simulate explore(action=search.file) returning file candidates.
        return ToolResultPayload(
            tool_name="explore",
            namespace="stub::explore",
            channel="graph",
            profile="X",
            determinism="hybrid",
            summary="explore ok",
            evidences=[],
            diagnostics={
                "actions": [
                    {
                        "id": "auto_search_file",
                        "tool": "search.file",
                        "status": "ok",
                        "summary": "search.file ok",
                        "diagnostics": {
                            "results": [{"file_id": fid} for fid in self._routed_file_ids],
                        },
                    }
                ]
            },
            think_notes=[],
        )


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


class _ReadPagesAwareToolManager(_StubToolManager):
    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        if tool_name != "explore":
            return await super().invoke(tool_name, payload=payload)

        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        actions = extra.get("actions") if isinstance(extra.get("actions"), list) else []
        wants_pages = any(isinstance(a, dict) and a.get("tool") == "read.pages" for a in actions)
        if wants_pages:
            chunk = EvidenceChunk(chunk_id="p1", source="read.pages", content="page 1")
            return ToolResultPayload(
                tool_name="explore",
                namespace="stub::explore",
                channel="graph",
                profile="X",
                determinism="hybrid",
                summary="read.pages ok",
                evidences=[chunk],
                diagnostics={},
                think_notes=[],
            )
        return await super().invoke(tool_name, payload=payload)


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
    explore_payload = next(payload for name, payload in tool_manager.calls if name == "explore")
    assert explore_payload.get("access_scope").scope_id == "owner"
    assert explore_payload.get("adapter") is loop.adapter
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


@pytest.mark.asyncio
async def test_think_loop_honors_is_final_stop_signal_and_skips_tool_calls() -> None:
    think_payloads = [
        {
            "reasoning": "We already have enough evidence; stop now.",
            "tool_calls": [
                {"tool_name": "explore", "tool_args": {"actions": [{"tool": "search"}]}, "rationale": "should be skipped", "parallelizable": True}
            ],
            "plan": [],
            "is_final": True,
        },
    ]
    tool_manager = _StubToolManager(think_payloads)
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "max_rounds_per_checkpoint": 1,
                "tool_call_concurrency": 1,
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
        metadata={"report_needed": True, "report_style": "deepsearch"},
    )
    seed = [EvidenceChunk(chunk_id="seed_p", source="read.pages", content="seed page")]

    await loop.run_think_loop("Q", graph_context=context, seed_evidences=seed)

    tool_names = [name for name, _ in tool_manager.calls]
    assert tool_names == ["think"]


@pytest.mark.asyncio
async def test_think_loop_final_signal_requires_read_pages_evidence() -> None:
    think_payloads = [
        {
            "reasoning": "Stop now (but we have no page evidence yet).",
            "tool_calls": [],
            "plan": [],
            "is_final": True,
        },
        {
            "reasoning": "Need page evidence; will read pages.",
            "tool_calls": [
                {
                    "tool_name": "explore",
                    "tool_args": {"actions": [{"tool": "read.pages", "args": {"file_id": "f1", "page_start": 1, "page_end": 1}}]},
                    "rationale": "collect citeable evidence",
                    "parallelizable": False,
                }
            ],
            "plan": [],
        },
        {
            "reasoning": "Now we can stop.",
            "tool_calls": [],
            "plan": [],
            "is_final": True,
        },
    ]
    tool_manager = _ReadPagesAwareToolManager(think_payloads)
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "max_rounds_per_checkpoint": 3,
                "tool_call_concurrency": 1,
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
        metadata={"report_needed": True, "report_style": "deepsearch"},
    )
    result = await loop.run_think_loop("Q", graph_context=context)
    assert any(ev.get("source") == "read.pages" for ev in result.get("evidences") or [])


class _InflightToolManager(_StubToolManager):
    """Tool manager that records max concurrent in-flight tool calls (excluding think)."""

    def __init__(self, think_payloads: list[dict]):
        super().__init__(think_payloads)
        self.inflight = 0
        self.max_inflight = 0

    async def invoke(self, tool_name: str, *, payload: dict) -> ToolResultPayload:
        if tool_name == "think":
            return await super().invoke(tool_name, payload=payload)

        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            # Give the scheduler a chance to overlap calls if concurrency > 1.
            await asyncio.sleep(0.05)
            return await super().invoke(tool_name, payload=payload)
        finally:
            self.inflight -= 1


@pytest.mark.asyncio
async def test_think_tool_call_concurrency_zero_is_sequential() -> None:
    think_payloads = [
        {
            "reasoning": "Run two tool calls.",
            "tool_calls": [
                {"tool_name": "explore", "tool_args": {"actions": [{"tool": "search"}]}, "rationale": "call 1", "parallelizable": True},
                {"tool_name": "code.python", "tool_args": {"code": "1+2"}, "rationale": "call 2", "parallelizable": True},
            ],
            "plan": [{"text": "Do two calls", "checked": False}],
        },
        {
            "reasoning": "Done.",
            "tool_calls": [],
            "plan": [{"text": "Do two calls", "checked": True}],
        },
    ]
    tool_manager = _InflightToolManager(think_payloads)
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "tool_call_concurrency": 0,  # 0 => sequential
                "max_tool_calls": 2,
                "max_rounds_per_checkpoint": 1,
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
    )

    await loop.run_think_loop("Q", graph_context=context)
    assert tool_manager.max_inflight == 1


@pytest.mark.asyncio
async def test_file_scope_is_locked_in_from_search_file_results() -> None:
    fid = "00000000-0000-0000-0000-000000000123"
    think_payloads = [
        {
            "reasoning": "Route to a file.",
            "tool_calls": [
                {"tool_name": "explore", "tool_args": {"actions": [{"tool": "search.file"}]}, "rationale": "route", "parallelizable": True}
            ],
            "plan": [],
        },
        {
            "reasoning": "Now continue.",
            "tool_calls": [],
            "plan": [],
        },
    ]
    tool_manager = _FileRoutingToolManager(think_payloads, routed_file_ids=[fid])
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(
            think_overrides={
                "tool_catalog_allowlist": ["explore", "code.python"],
                "tool_call_concurrency": 1,
                "max_rounds_per_checkpoint": 2,
            }
        ),
        tool_manager=tool_manager,
    )
    context = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
    )

    await loop.run_think_loop("Q", graph_context=context)

    think_payloads_seen = [payload for name, payload in tool_manager.calls if name == "think"]
    assert len(think_payloads_seen) >= 2
    second = think_payloads_seen[1]
    meta = (second.get("graph_context") or {}).get("metadata") or {}
    scope = meta.get("file_scope")
    assert isinstance(scope, dict)
    assert scope.get("file_ids") == [fid]
