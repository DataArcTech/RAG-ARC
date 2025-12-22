import asyncio

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ToolResultPayload
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.reasoning import GraphReasoningLoop


class _StubAdapter:
    async def prepare(self, question: str, *, access_scope=None) -> None:  # pragma: no cover - simple stub
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
        return {
            "chunks": [
                {"content": f"chunk::{query}", "metadata": {"chunk_id": "c1"}},
            ],
            "nodes": [{"id": "n1"}],
            "edges": [{"id": "e1"}],
            "metadata": {"adapter": "hipporag"},
        }

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        return f"summary::{channel}"

    async def chain_traverse(self, strategy, *, access_scope=None):
        return {"strategy": strategy.get("strategy"), "hops": 1}

    def metadata(self):
        return type(
            "_Meta",
            (),
            {
                "adapter_name": "hipporag",
                "graph_type": "hipporag",
                "version": "test",
                "capabilities": (),
                "domain_tags": (),
                "config_fingerprint": None,
            },
        )()


class _StubToolManager:
    def __init__(self):
        self.calls = []

    async def invoke(self, tool_name: str, *, payload):
        self.calls.append((tool_name, payload))
        chunk = EvidenceChunk(chunk_id=f"tool-{tool_name}-0", source="tool", content="tool evidence")
        return ToolResultPayload(
            tool_name=tool_name,
            namespace=f"stub::{tool_name}",
            channel="text",
            profile="F",
            determinism="deterministic",
            summary=f"{tool_name} summary",
            evidences=[chunk],
            diagnostics={},
            think_notes=[],
        )


class _HangingToolManager:
    def __init__(self, delay: float = 0.2):
        self.delay = delay

    async def invoke(self, tool_name: str, *, payload):
        await asyncio.sleep(self.delay)
        chunk = EvidenceChunk(chunk_id="slow-tool", source="tool", content="slow output")
        return ToolResultPayload(
            tool_name=tool_name,
            namespace="stub::slow",
            channel="text",
            profile="F",
            determinism="deterministic",
            summary="slow result",
            evidences=[chunk],
            diagnostics={},
            think_notes=[],
        )


@pytest.mark.asyncio
async def test_graph_reasoning_combines_traversal_and_tools():
    adapter = _StubAdapter()
    tool_manager = _StubToolManager()
    loop = GraphReasoningLoop(adapter=adapter, llm_connector=None, strategy_config={"strategy_name": "ppr_chain"}, tool_manager=tool_manager)

    plan_steps = [
        {"step_id": "plan_01", "description": "Inspect graph", "channel": "graph", "tool": "graph_adapter.query"},
        {"step_id": "plan_02", "description": "Rollup", "channel": "text", "tool": "graph.context_rollup"},
        {"step_id": "plan_03", "description": "Call web", "channel": "web", "tool": "web.search", "requires_external": True},
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Who founded OpenAI?", access_scope=GraphAccessScope(scope_id="scope-graph-test"))
    result = await loop.run("Who founded OpenAI?", plan_steps, graph_context=context)

    assert result["graph_traversals"], "graph traversal should run for the first step"
    assert any(run["tool_name"] == "graph.context_rollup" for run in result["tool_results"])
    assert result["pending_external"] and result["pending_external"][0]["step_id"] == "plan_03"
    assert tool_manager.calls and tool_manager.calls[0][0] == "graph.context_rollup"
    assert tool_manager.calls[0][1]["context_evidences"], "tool should receive evidence context"
    statuses = {step["step_id"]: step["status"] for step in result["reasoning_steps"]}
    assert statuses["plan_01"] == "done"
    assert statuses["plan_02"] == "done"
    assert statuses["plan_03"] == "pending_external"
    coverage = result.get("coverage_metrics") or {}
    assert "coverage_ratio" in coverage
    assert "coverage_score" in coverage


@pytest.mark.asyncio
async def test_graph_reasoning_marks_missing_tool_manager_skips_step():
    adapter = _StubAdapter()
    loop = GraphReasoningLoop(adapter=adapter, llm_connector=None, strategy_config={})

    plan_steps = [
        {"step_id": "plan_02", "description": "Text summary", "channel": "text", "tool": "graph.context_rollup"},
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Summarize", access_scope=GraphAccessScope(scope_id="scope-text-test"))
    result = await loop.run("Summarize", plan_steps, graph_context=context)

    assert result["reasoning_steps"], "reasoning log should exist even when tool calls fail"
    entry = result["reasoning_steps"][0]
    assert entry["status"] == "skipped"
    assert entry["diagnostics"]["reason"] == "tool_manager_disabled"
    assert entry["diagnostics"]["tool"] == "graph.context_rollup"


@pytest.mark.asyncio
async def test_graph_reasoning_populates_seed_entities_from_plan():
    adapter = _StubAdapter()
    tool_manager = _StubToolManager()
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config={},
        tool_manager=tool_manager,
    )

    plan_steps = [
        {
            "step_id": "plan_seed",
            "description": "Probe entities",
            "channel": "graph",
            "tool": "graph_adapter.query",
            "metadata": {"seed_entities": ["OpenAI"]},
            "tool_args": {"seed_entities": ["Anthropic"]},
        }
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Investigate", access_scope=GraphAccessScope(scope_id="scope-seed-test"))
    result = await loop.run("Investigate", plan_steps, graph_context=context)
    context = result["graph_context"]
    assert sorted(context["seed_entities"]) == ["Anthropic", "OpenAI"]
    assert result["graph_traversals"], "graph traversal should be recorded"
    first_traversal = result["graph_traversals"][0]
    assert sorted(first_traversal["seed_entities"]) == ["Anthropic", "OpenAI"]


@pytest.mark.asyncio
async def test_graph_reasoning_inserts_periodic_think():
    adapter = _StubAdapter()
    tool_manager = _StubToolManager()
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config={
            "think_every_n_steps": 1,
            "think_min_coverage": 1.1,
        },
        tool_manager=tool_manager,
    )

    plan_steps = [
        {"step_id": "plan_01", "description": "Inspect graph", "channel": "graph", "tool": "graph_adapter.query"},
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Need more thinking", access_scope=GraphAccessScope(scope_id="scope-think-test"))
    result = await loop.run("Need more thinking", plan_steps, graph_context=context)

    think_steps = [step for step in result["reasoning_steps"] if step["step_id"].startswith("think_auto_")]
    assert think_steps, "Think checkpoint should be inserted after cadence is met"
    assert any(call[0] == "graph.think" for call in tool_manager.calls)
    assert any(run["tool_name"] == "graph.think" for run in result["tool_results"])


@pytest.mark.asyncio
async def test_graph_reasoning_think_timeout_falls_back_to_heuristic_note():
    adapter = _StubAdapter()
    tool_manager = _HangingToolManager(delay=0.2)
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config={
            "think_every_n_steps": 1,
            "think_min_coverage": 1.1,
            "tool_timeout_seconds": 0.01,
        },
        tool_manager=tool_manager,
    )

    plan_steps = [
        {"step_id": "plan_01", "description": "Inspect graph", "channel": "graph", "tool": "graph_adapter.query"},
    ]

    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Need more thinking",
        access_scope=GraphAccessScope(scope_id="scope-think-timeout"),
    )
    result = await loop.run("Need more thinking", plan_steps, graph_context=context)

    think_steps = [step for step in result["reasoning_steps"] if step["step_id"].startswith("think_auto_")]
    assert think_steps
    assert think_steps[0]["status"] == "done"
    assert think_steps[0]["diagnostics"]["reason"] == "tool_timeout_fallback"
    assert result["think_notes"], "Timeout fallback should still record a ThinkNote"
class _SeedAwareAdapter(_StubAdapter):
    def __init__(self):
        super().__init__()
        self.chain_payloads = []

    async def chain_traverse(self, strategy, *, access_scope=None):
        self.chain_payloads.append(strategy)
        return await super().chain_traverse(strategy, access_scope=access_scope)


@pytest.mark.asyncio
async def test_graph_traversal_executor_propagates_seed_entities():
    from core.deepsearch.reasoning.traversal import GraphTraversalExecutor, GraphTraversalSettings
    from encapsulation.data_model.deepsearch import GraphQueryContext, PlanSpec

    adapter = _SeedAwareAdapter()
    executor = GraphTraversalExecutor(adapter=adapter, settings=GraphTraversalSettings(chain_depth=2))
    context = GraphQueryContext(adapter_name="hipporag", question="Q1", seed_entities=["NodeA"])
    plan = [
        PlanSpec(step_id="plan_seed", description="Hop", channel="graph", metadata={}),
    ]
    await executor.run(plan, context, tool_args_map={"plan_seed": {"seed_entities": ["NodeB"]}})

    assert adapter.chain_payloads, "chain traversal should run"
    payload = adapter.chain_payloads[0]
    assert payload["seed_entities"] == ["NodeA", "NodeB"]


class _SlowAdapter(_StubAdapter):
    def __init__(self):
        super().__init__()
        self.concurrent_calls = 0
        self.max_concurrency = 0

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
        self.concurrent_calls += 1
        self.max_concurrency = max(self.max_concurrency, self.concurrent_calls)
        try:
            await asyncio.sleep(0.05)
            return await super().aquery_subgraph(query, channel=channel, access_scope=access_scope)
        finally:
            self.concurrent_calls -= 1


@pytest.mark.asyncio
async def test_graph_reasoning_parallelises_tool_steps_when_configured():
    class _SlowToolManager(_StubToolManager):
        def __init__(self):
            super().__init__()
            self.concurrent_calls = 0
            self.max_concurrency = 0

        async def invoke(self, tool_name: str, *, payload):
            self.concurrent_calls += 1
            self.max_concurrency = max(self.max_concurrency, self.concurrent_calls)
            try:
                await asyncio.sleep(0.05)
                return await super().invoke(tool_name, payload=payload)
            finally:
                self.concurrent_calls -= 1

    tool_manager = _SlowToolManager()
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config={"parallel_branches": 3},
        tool_manager=tool_manager,
    )

    plan_steps = [
        {"step_id": f"plan_{idx:02d}", "description": f"Rollup {idx}", "channel": "text", "tool": "graph.context_rollup"}
        for idx in range(1, 4)
    ]
    context = GraphQueryContext(adapter_name="hipporag", question="Parallel tools?", access_scope=GraphAccessScope(scope_id="scope-parallel"))
    await loop.run("Parallel tools?", plan_steps, graph_context=context)

    assert tool_manager.max_concurrency >= 2, "tool steps should overlap when parallel_branches > 1"


@pytest.mark.asyncio
async def test_graph_reasoning_auto_parallel_requires_scheduler_hint():
    class _SlowToolManager(_StubToolManager):
        def __init__(self):
            super().__init__()
            self.concurrent_calls = 0
            self.max_concurrency = 0

        async def invoke(self, tool_name: str, *, payload):
            self.concurrent_calls += 1
            self.max_concurrency = max(self.max_concurrency, self.concurrent_calls)
            try:
                await asyncio.sleep(0.05)
                return await super().invoke(tool_name, payload=payload)
            finally:
                self.concurrent_calls -= 1

    tool_manager = _SlowToolManager()
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config={"parallel_branches": 0, "max_parallel_branches": 3},
        tool_manager=tool_manager,
    )

    plan_steps = [
        {"step_id": f"plan_{idx:02d}", "description": f"Rollup {idx}", "channel": "text", "tool": "graph.context_rollup"}
        for idx in range(1, 5)
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Auto parallel?", access_scope=GraphAccessScope(scope_id="scope-auto"))
    await loop.run("Auto parallel?", plan_steps, graph_context=context)

    assert tool_manager.max_concurrency == 1, "auto parallel stays serial unless plan opts in"
    assert getattr(loop, "_active_parallel_branches") == 1


@pytest.mark.asyncio
async def test_graph_reasoning_auto_parallel_with_scheduler_hint():
    class _SlowToolManager(_StubToolManager):
        def __init__(self):
            super().__init__()
            self.concurrent_calls = 0
            self.max_concurrency = 0

        async def invoke(self, tool_name: str, *, payload):
            self.concurrent_calls += 1
            self.max_concurrency = max(self.max_concurrency, self.concurrent_calls)
            try:
                await asyncio.sleep(0.05)
                return await super().invoke(tool_name, payload=payload)
            finally:
                self.concurrent_calls -= 1

    tool_manager = _SlowToolManager()
    loop = GraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config={"parallel_branches": 0, "max_parallel_branches": 3},
        tool_manager=tool_manager,
    )

    plan_steps = [
        {
            "step_id": f"plan_{idx:02d}",
            "description": f"Rollup {idx}",
            "channel": "text",
            "tool": "graph.context_rollup",
            "metadata": {"scheduler": "parallel"},
        }
        for idx in range(1, 5)
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Auto parallel?", access_scope=GraphAccessScope(scope_id="scope-auto"))
    await loop.run("Auto parallel?", plan_steps, graph_context=context)

    assert tool_manager.max_concurrency >= 2
    assert getattr(loop, "_active_parallel_branches") == 3


@pytest.mark.asyncio
async def test_graph_reasoning_serializes_adapter_calls_even_when_parallel_enabled():
    adapter = _SlowAdapter()
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config={"parallel_branches": 3},
        tool_manager=_StubToolManager(),
    )

    plan_steps = [
        {"step_id": f"plan_{idx:02d}", "description": f"Probe {idx}", "channel": "graph", "tool": "graph_adapter.query"}
        for idx in range(1, 4)
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Parallel adapter?", access_scope=GraphAccessScope(scope_id="scope-parallel-adapter"))
    await loop.run("Parallel adapter?", plan_steps, graph_context=context)

    assert adapter.max_concurrency == 1, "adapter calls are serialized to avoid shared-state corruption"


@pytest.mark.asyncio
async def test_graph_reasoning_marks_tool_timeout():
    adapter = _StubAdapter()
    tool_manager = _HangingToolManager(delay=0.2)
    loop = GraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config={"tool_timeout_seconds": 0.05},
        tool_manager=tool_manager,
    )

    plan_steps = [
        {"step_id": "plan_timeout", "description": "Rollup", "channel": "text", "tool": "graph.context_rollup"},
    ]

    context = GraphQueryContext(adapter_name="hipporag", question="Timeout?", access_scope=GraphAccessScope(scope_id="scope-timeout"))
    result = await loop.run("Timeout?", plan_steps, graph_context=context)

    entry = result["reasoning_steps"][0]
    assert entry["status"] == "failed"
    assert entry["diagnostics"]["reason"] == "tool_timeout"


class _ScopeRecordingAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def prepare(self, question: str, *, access_scope=None) -> None:
        await asyncio.sleep(0.02)
        self.calls.append(("prepare", getattr(access_scope, "scope_id", None)))

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
        await asyncio.sleep(0.01)
        scope_id = getattr(access_scope, "scope_id", None)
        self.calls.append(("query", scope_id))
        return {
            "chunks": [{"id": f"chunk:{scope_id}", "content": f"chunk:{scope_id}", "metadata": {}}],
            "nodes": [],
            "edges": [],
            "metadata": {},
        }

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        self.calls.append(("filter", getattr(access_scope, "scope_id", None)))
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        scope_id = getattr(access_scope, "scope_id", None)
        self.calls.append(("summarize", scope_id))
        return f"sum:{scope_id}"

    async def chain_traverse(self, strategy, *, access_scope=None):
        self.calls.append(("chain", getattr(access_scope, "scope_id", None)))
        return {"hops": 1}

    def metadata(self):
        return type(
            "_Meta",
            (),
            {
                "adapter_name": "hipporag",
                "graph_type": "hipporag",
                "version": "test",
                "capabilities": (),
                "domain_tags": (),
                "config_fingerprint": None,
            },
        )()


@pytest.mark.asyncio
async def test_graph_reasoning_concurrent_runs_do_not_mix_scopes_or_evidence():
    adapter = _ScopeRecordingAdapter()
    loop = GraphReasoningLoop(adapter=adapter, llm_connector=None, strategy_config={"parallel_branches": 2})
    plan_steps = [{"step_id": "plan_01", "description": "Inspect graph", "channel": "graph", "tool": "graph_adapter.query"}]

    async def _run(scope_id: str):
        context = GraphQueryContext(
            adapter_name="hipporag",
            question=f"Q:{scope_id}",
            access_scope=GraphAccessScope(scope_id=scope_id),
        )
        result = await loop.run(f"Q:{scope_id}", plan_steps, graph_context=context)
        return [ev["content"] for ev in result.get("evidences") or []]

    evid_a, evid_b = await asyncio.gather(_run("scope-a"), _run("scope-b"))
    assert evid_a == ["chunk:scope-a"]
    assert evid_b == ["chunk:scope-b"]

    prepared_scopes = [scope for action, scope in adapter.calls if action == "prepare"]
    assert set(prepared_scopes) == {"scope-a", "scope-b"}
