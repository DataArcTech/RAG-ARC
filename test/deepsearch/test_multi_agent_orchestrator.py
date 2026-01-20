import asyncio

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ToolResultPayload
from core.deepsearch.reasoning import MultiAgentGraphReasoningLoop
from core.graph_adapter.base import GraphAccessScope


def _strategy_config(*, overrides: dict | None = None, think_overrides: dict | None = None) -> dict:
    base = {
        "strategy_name": "ppr_chain",
        "allow_semantic_channel": True,
        "chain_depth": 2,
        "parallel_branches": 1,
        "max_parallel_branches": 4,
        "step_summary_max_chars": 2000,
        "tool_context_max_evidences": 5,
        "tool_context_max_chars": 800,
        "coverage_expected_min_chunks": 3,
        "trace_reflection_enabled": False,
        "trace_reflection_max": 0,
        "tool_timeout_seconds": 0.0,
        "think": {
            "tool_name": "think",
            "every_n_steps": 0,
            "min_coverage": 0.0,
            "enable_tool_calls": False,
            "max_tool_calls": 0,
            "tool_call_concurrency": 0,
            "tool_catalog_max_items": 0,
            "include_llm_tools": True,
            "max_rounds_per_checkpoint": 1,
        },
    }
    if overrides:
        base.update(dict(overrides))
    if think_overrides:
        merged = dict(base["think"])
        merged.update(dict(think_overrides))
        base["think"] = merged
    return base


def _multi_agent_settings(*, overrides: dict | None = None) -> dict:
    base = {
        "enabled": True,
        "max_subagents": 2,
        "subagent_concurrency": 2,
        "enable_parallel_tool_probes": False,
        "probe_tool_names": [],
        "probe_concurrency": 1,
        "lead_tool_names": [],
        "lead_tool_concurrency": 1,
        "worker_timeout_seconds": None,
        "worker_retry_attempts": 0,
        "fail_fast": False,
        "incremental_parallelism": False,
        "initial_worker_count": 1,
        "stop_min_evidence_count": 0,
        "stop_min_coverage_ratio": 0.0,
        "max_merge_evidences": 60,
    }
    if overrides:
        base.update(dict(overrides))
    return base


class _StubAdapter:
    async def prepare(self, question: str, *, access_scope=None) -> None:  # pragma: no cover
        await asyncio.sleep(0)

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        return {
            "chunks": [
                {"content": f"chunk::{query}", "metadata": {"chunk_id": f"c::{query}"}},
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


class _NonConcurrentAdapter(_StubAdapter):
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        if self.in_flight:
            raise RuntimeError("concurrent adapter access")
        self.in_flight = 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return await super().aquery_subgraph(query, channel=channel, access_scope=access_scope, query_options=query_options)
        finally:
            self.in_flight = 0


class _ConcurrentToolManager:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.concurrent_calls = 0
        self.max_concurrency = 0
        self.calls = []

    async def invoke(self, tool_name: str, *, payload):
        self.concurrent_calls += 1
        self.max_concurrency = max(self.max_concurrency, self.concurrent_calls)
        try:
            await asyncio.sleep(self.delay)
            self.calls.append((tool_name, payload))
            chunk = EvidenceChunk(chunk_id=f"{tool_name}-ev", source=tool_name, content=f"ev::{tool_name}")
            return ToolResultPayload(
                tool_name=tool_name,
                namespace=f"stub::{tool_name}",
                channel="graph",
                profile="F",
                determinism="deterministic",
                summary=f"{tool_name} ok",
                evidences=[chunk],
                diagnostics={},
                think_notes=[],
            )
        finally:
            self.concurrent_calls -= 1


class _SelectiveSlowToolManager:
    def __init__(self, *, slow_plan_step: str, slow_delay: float, fast_delay: float = 0.0):
        self.slow_plan_step = slow_plan_step
        self.slow_delay = slow_delay
        self.fast_delay = fast_delay

    async def invoke(self, tool_name: str, *, payload):
        plan_step = str(payload.get("plan_step") or "")
        if plan_step == self.slow_plan_step:
            await asyncio.sleep(self.slow_delay)
        elif self.fast_delay:
            await asyncio.sleep(self.fast_delay)
        chunk = EvidenceChunk(chunk_id=f"{plan_step}::ev", source=tool_name, content=f"ev::{plan_step}")
        return ToolResultPayload(
            tool_name=tool_name,
            namespace=f"stub::{tool_name}",
            channel="graph",
            profile="F",
            determinism="deterministic",
            summary=f"{tool_name} ok",
            evidences=[chunk],
            diagnostics={},
            think_notes=[],
        )


@pytest.mark.asyncio
async def test_multi_agent_runs_workers_and_invokes_tools_concurrently():
    tool_manager = _ConcurrentToolManager(delay=0.05)
    loop = MultiAgentGraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(overrides={"parallel_branches": 1}),
        tool_manager=tool_manager,
        settings=_multi_agent_settings(
            overrides={
                "max_subagents": 3,
                "subagent_concurrency": 3,
                "enable_parallel_tool_probes": True,
                "probe_tool_names": ["search"],
                "probe_concurrency": 2,
                "lead_tool_names": ["graph.llm_chain_explorer"],
                "lead_tool_concurrency": 2,
                "incremental_parallelism": False,
                "initial_worker_count": 3,
                "max_merge_evidences": 60,
            }
        ),
        graph_channel_tool="graph_adapter.query",
    )

    plan_steps = [
        {
            "step_id": f"plan_{idx:02d}",
            "description": f"Probe {idx}",
            "channel": "graph",
            "tool": "graph_adapter.query",
            "metadata": {"scheduler": "parallel"},
        }
        for idx in range(1, 4)
    ]
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-multi-agent"),
    )
    result = await loop.run("Q", plan_steps, graph_context=context)

    assert result["agent_sessions"], "multi-agent loop should report agent sessions"
    assert len(result["agent_sessions"]) == 3
    assert result["question"] == "Q"
    assert result["evidences"], "merged evidence should include traversal and probe outputs"
    assert tool_manager.calls, "lead/probe tools should be invoked via tool manager"
    assert tool_manager.max_concurrency >= 2, "tool invocations should overlap when concurrency > 1"
    for session in result["agent_sessions"]:
        assert session.get("session_id")
        assert session.get("assigned_step_ids")
        assert session.get("debrief")


@pytest.mark.asyncio
async def test_multi_agent_serializes_shared_adapter_access():
    adapter = _NonConcurrentAdapter(delay=0.05)
    loop = MultiAgentGraphReasoningLoop(
        adapter=adapter,
        llm_connector=None,
        strategy_config=_strategy_config(overrides={"parallel_branches": 1}),
        tool_manager=None,
        settings=_multi_agent_settings(
            overrides={
                "max_subagents": 2,
                "subagent_concurrency": 2,
            }
        ),
        graph_channel_tool="graph_adapter.query",
    )

    plan_steps = [
        {
            "step_id": "plan_01",
            "description": "Probe 1",
            "channel": "graph",
            "tool": "graph_adapter.query",
            "metadata": {"scheduler": "parallel"},
        },
        {
            "step_id": "plan_02",
            "description": "Probe 2",
            "channel": "graph",
            "tool": "graph_adapter.query",
            "metadata": {"scheduler": "parallel"},
        },
    ]
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-multi-agent"),
    )
    result = await loop.run("Q", plan_steps, graph_context=context)

    assert adapter.max_in_flight == 1
    assert result["reasoning_steps"], "multi-agent loop should include reasoning steps"
    assert all(step.get("status") == "done" for step in result["reasoning_steps"])


@pytest.mark.asyncio
async def test_multi_agent_surfaces_worker_failures_in_coverage_metrics():
    tool_manager = _SelectiveSlowToolManager(slow_plan_step="plan_slow", slow_delay=0.15)
    loop = MultiAgentGraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(overrides={"parallel_branches": 1}),
        tool_manager=tool_manager,
        settings=_multi_agent_settings(
            overrides={
                "max_subagents": 2,
                "subagent_concurrency": 2,
                "worker_timeout_seconds": 0.05,
            }
        ),
        graph_channel_tool="graph_adapter.query",
    )

    plan_steps = [
        {
            "step_id": "plan_slow",
            "description": "Slow step",
            "channel": "graph",
            "tool": "graph.llm_chain_explorer",
            "metadata": {"scheduler": "parallel"},
        },
        {
            "step_id": "plan_fast",
            "description": "Fast step",
            "channel": "graph",
            "tool": "graph.llm_chain_explorer",
            "metadata": {"scheduler": "parallel"},
        },
    ]
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-multi-agent"),
    )
    result = await loop.run("Q", plan_steps, graph_context=context)

    coverage = result.get("coverage_metrics") or {}
    assert coverage.get("worker_error_count") == 1
    errors = coverage.get("worker_errors") or []
    assert isinstance(errors, list) and errors
    assert any(entry.get("error") == "worker_timeout" for entry in errors if isinstance(entry, dict))


@pytest.mark.asyncio
async def test_multi_agent_respects_serial_scheduler_and_runs_single_worker():
    loop = MultiAgentGraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(overrides={"parallel_branches": 1}),
        tool_manager=None,
        settings=_multi_agent_settings(
            overrides={
                "max_subagents": 4,
                "subagent_concurrency": 4,
            }
        ),
        graph_channel_tool="graph_adapter.query",
    )

    plan_steps = [
        {"step_id": "s1", "description": "First", "channel": "graph", "tool": "graph_adapter.query", "metadata": {"scheduler": "serial"}},
        {"step_id": "s2", "description": "Second", "channel": "graph", "tool": "graph_adapter.query", "metadata": {"scheduler": "serial"}},
    ]
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-multi-agent"),
    )
    result = await loop.run("Q", plan_steps, graph_context=context)
    assert len(result.get("agent_sessions") or []) == 1
    session = (result.get("agent_sessions") or [None])[0] or {}
    assert session.get("assigned_step_ids") == ["s1", "s2"]
