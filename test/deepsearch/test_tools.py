import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, PlanSpec, ToolExecutionLog
from encapsulation.mcp.client import MCPToolCallOutcome
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata
from core.deepsearch.tools import (
    ContextRewriterTool,
    ContextRollupTool,
    CrossAdapterPlannerTool,
    EvidenceCrosscheckTool,
    BeamSearchTool,
    GraphThinkTool,
    GraphTool,
    HybridNeighborhoodProbeTool,
    LLMChainExplorerTool,
    ParallelThinkTool,
    PathCacheTool,
    PatternProbeTool,
    ToolDescriptor,
    ToolResult,
    ToolRunRequest,
    get_tool_descriptor,
)
from core.deepsearch.plan.runtime import DeepSearchPlanner
from core.deepsearch.reasoning.traversal import GraphTraversalExecutor, GraphTraversalSettings
from encapsulation.deepsearch.tooling import DeepSearchToolManager, LocalToolRegistry
from core.deepsearch.tooling import describe_available_tools, clear_tool_hints, register_tool_hints
from core.deepsearch.tooling.registry import ToolHintRegistry


class _StubAdapter:
    def __init__(self):
        capability = GraphAdapterCapability(name="test")
        self._metadata = GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type="hipporag",
            version="test",
            capabilities=(capability,),
        )

    async def prepare(self, question: str, *, access_scope=None) -> None:
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
        return {
            "chunks": [
                {"content": f"{query} chunk A", "metadata": {"id": 1}},
                {"content": "unrelated text", "metadata": {"id": 2}},
            ],
            "metadata": {"adapter": "hipporag"},
        }

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        return f"summarized::{channel}"

    async def chain_traverse(self, strategy, *, access_scope=None):
        payload = {"strategy": strategy.get("strategy"), "hops": 1}
        if strategy.get("strategy") == "beam_search":
            payload["paths"] = [
                {
                    "path_id": "beam-0",
                    "nodes": ["OpenAI", "Microsoft", "Azure"],
                    "score": 0.8,
                    "summary": "OpenAI integrates with Azure via strategic partnership.",
                },
                {
                    "path_id": "beam-1",
                    "nodes": ["OpenAI", "Anthropic"],
                    "score": 0.3,
                },
            ]
        return payload

    def metadata(self):
        return self._metadata


class _StubLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, messages, **kwargs):
        return self.response


class _StubPlanGenerator:
    def __init__(self):
        self.settings = SimpleNamespace(mode="react", max_steps=2, enable_sub_question=True)

    async def agenerate_plan(self, question: str, context=None):
        return [
            PlanSpec(
                step_id="plan_01",
                description="Investigate question",
                channel="graph",
                metadata={"seed_entities": ["OpenAI"]},
            )
        ]


@pytest.mark.asyncio
async def test_pattern_probe_scans_chunks():
    adapter = _StubAdapter()
    tool = PatternProbeTool()
    request = ToolRunRequest(
        question="OpenAI founders",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert "Pattern scan succeeded" in result.summary
    assert result.evidences
    assert result.evidences[0].content.lower().startswith("openai")


@pytest.mark.asyncio
async def test_pattern_probe_handles_chinese_tokens():
    adapter = _StubAdapter()
    tool = PatternProbeTool()
    request = ToolRunRequest(
        question="谁提出了深度搜索？",
        plan_step="plan_zh",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.evidences  # 应该匹配中文 chunk
    assert result.diagnostics["keywords"]  # 记录被选中的关键词
    assert "Pattern scan succeeded" in result.summary


@pytest.mark.asyncio
async def test_hybrid_probe_combines_scans_and_chain(monkeypatch):
    adapter = _StubAdapter()
    llm = _StubLLM("hybrid summary")
    tool = HybridNeighborhoodProbeTool(llm)

    request = ToolRunRequest(
        question="Analyze OpenAI history",
        plan_step=None,
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert "hybrid summary" in result.summary
    assert result.evidences
    assert result.diagnostics["enriched_count"] == len(result.evidences)
    assert 0 <= result.diagnostics["determinism_ratio"] <= 1


def test_describe_available_tools_defaults(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    monkeypatch.setenv("DEEPSEARCH_ENABLE_LLM_TOOLS", "true")
    clear_tool_hints()
    hints = describe_available_tools()
    names = [hint["name"] for hint in hints]
    assert "graph.pattern_scan" in names
    assert "graph.llm_chain_explorer" in names
    assert all(
        "profile" in hint
        and "determinism" in hint
        and "speed" in hint
        and "cost" in hint
        and "strategy_tags" in hint
        for hint in hints
    )


def test_describe_available_tools_includes_registered_hints(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    clear_tool_hints()
    register_tool_hints(
        [
            {
                "name": "custom.remote_planner",
                "channel": "graph",
                "description": "remote planner",
                "profile": "H",
                "determinism": "llm_heavy",
                "namespace": "rag-arc.deepsearch.remote.planner",
                "speed": "slow",
                "cost": "high",
                "strategy_tags": ["remote"],
            }
        ]
    )
    hints = describe_available_tools()
    assert any(hint["name"] == "custom.remote_planner" for hint in hints)


def test_local_registry_registers_custom_tool_hints(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    clear_tool_hints()

    class _CustomTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.inline",
            channel="graph",
            description="inline",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_inline",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary="inline")

    LocalToolRegistry(tool_configs={}, injected_tools={"custom.inline": _CustomTool()})
    hints = describe_available_tools()
    assert any(hint["name"] == "custom.inline" for hint in hints)


@pytest.mark.asyncio
async def test_planner_plan_reflects_registered_remote_tools(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    clear_tool_hints()

    planner = DeepSearchPlanner(
        prompt_store={},
        llm_connector=None,
        config={"persist_plan": False},
        plan_generator=_StubPlanGenerator(),
    )

    register_tool_hints(
        [
            {
                "name": "custom.remote_planner",
                "channel": "graph",
                "description": "remote planner",
                "profile": "H",
                "determinism": "llm_heavy",
                "namespace": "rag-arc.deepsearch.remote.planner",
                "speed": "slow",
                "cost": "high",
                "strategy_tags": ["remote"],
            }
        ]
    )

    artifact = await planner.build_plan(
        "Investigate remote hint",
        access_scope=GraphAccessScope(scope_id="planner-test"),
    )
    plan = artifact["plan"]
    tool_names = [spec["name"] for spec in plan["available_tools"]]
    assert "custom.remote_planner" in tool_names
    graph_context = plan.get("graph_context")
    assert graph_context
    assert graph_context["adapter_name"] == planner.graph_adapter_name
    assert "OpenAI" in graph_context.get("seed_entities", [])


def test_describe_available_tools_respects_llm_toggle(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    clear_tool_hints()
    monkeypatch.setenv("DEEPSEARCH_DISABLE_LLM_TOOLS", "true")
    hints = describe_available_tools()
    names = {hint["name"] for hint in hints}
    assert "graph.llm_chain_explorer" not in names
    assert "graph.parallel_think" not in names

    monkeypatch.delenv("DEEPSEARCH_DISABLE_LLM_TOOLS", raising=False)
    monkeypatch.setenv("DEEPSEARCH_ENABLE_LLM_TOOLS", "true")
    hints = describe_available_tools()
    names = {hint["name"] for hint in hints}
    assert "graph.llm_chain_explorer" in names
    assert "graph.parallel_think" in names


@pytest.mark.asyncio
async def test_tool_manager_invokes_local_tool():
    class _LocalTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.echo",
            channel="graph",
            description="echo",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary=f"echo::{request.question}")

    class _Telemetry:
        def __init__(self):
            self.logs = []

        def log_tool_invocation(self, **kwargs):
            self.logs.append(kwargs)

    telemetry = _Telemetry()
    manager = DeepSearchToolManager(
        tool_configs={},
        telemetry_client=telemetry,
        local_tools={"custom.echo": _LocalTool()},
    )
    result = await manager.invoke(
        "custom.echo",
        payload={"question": "hello", "context_evidences": []},
    )
    assert result.summary == "echo::hello"
    assert telemetry.logs


@pytest.mark.asyncio
async def test_tool_manager_without_mcp_client_avoids_remote():
    class _Telemetry:
        def log_tool_invocation(self, **kwargs):
            return None

    manager = DeepSearchToolManager(
        tool_configs={"enable_builtin_tools": True},
        telemetry_client=_Telemetry(),
        mcp_client=None,
    )
    with pytest.raises(KeyError):
        await manager.invoke("graph.parallel_think", payload={"question": "fallback", "context_evidences": []})


@pytest.mark.asyncio
async def test_tool_manager_persists_artifacts(tmp_path):
    class _LocalTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.echo",
            channel="graph",
            description="echo",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary=f"echo::{request.question}")

    class _Telemetry:
        def log_tool_invocation(self, **kwargs):  # pragma: no cover - noop
            return None

    manager = DeepSearchToolManager(
        tool_configs={"artifact_dir": str(tmp_path)},
        telemetry_client=_Telemetry(),
        local_tools={"custom.echo": _LocalTool()},
    )
    result = await manager.invoke("custom.echo", payload={"question": "persist", "context_evidences": []})
    artifacts = result.diagnostics.get("artifacts")
    assert artifacts and Path(artifacts[0]["path"]).is_file()


def test_bridge_lookup_schema_exposes_seed_entities():
    descriptor = get_tool_descriptor("graph.bridge_lookup")
    assert descriptor is not None
    extra_schema = descriptor.input_schema["properties"]["extra"]["properties"]
    assert "seed_entities" in extra_schema
    assert extra_schema["seed_entities"]["type"] == "array"


@pytest.mark.asyncio
async def test_evidence_crosscheck_detects_missing_and_confirmed():
    tool = EvidenceCrosscheckTool()
    evidences = [
        EvidenceChunk(
            chunk_id="chunk-1",
            source="hipporag",
            content="OpenAI was founded by Sam Altman in San Francisco.",
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            source="hipporag",
            content="No mention of GPT-4 release dates appears here.",
        ),
    ]
    request = ToolRunRequest(
        question="Who founded OpenAI and when was GPT-4 released?",
        plan_step="plan_01",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        extra={
            "triples": [
                {"head": "OpenAI", "relation": "founded_by", "tail": "Sam Altman"},
                {"head": "GPT-4", "relation": "released_in", "tail": "2023"},
            ]
        },
    )

    result = await tool.run(request)
    assert result.diagnostics["triple_count"] == 2
    assert result.diagnostics["confirmed"] == 1
    assert result.diagnostics["missing"] == 1
    breakdown = result.diagnostics.get("token_breakdown")
    assert breakdown and "deterministic_tokens" in breakdown and "llm_tokens" in breakdown
    assert result.think_notes  # gap should trigger a think note


@pytest.mark.asyncio
async def test_evidence_crosscheck_consumes_triples_from_traversal_evidence():
    class _Adapter:
        async def prepare(self, question: str, *, access_scope=None):
            return None

        async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
            return {
                "chunks": [{"id": "kb_chunk_1", "content": "OpenAI was founded by Sam Altman.", "metadata": {}}],
                "nodes": [{"id": "OpenAI"}, {"id": "Sam Altman"}],
                "edges": [{"source": "OpenAI", "relation": "founded_by", "target": "Sam Altman"}],
                "metadata": {"adapter": "dummy"},
            }

        async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
            return data

        async def summarize(self, channel: str, data, *, access_scope=None) -> str:
            return "OpenAI was founded by Sam Altman."

        async def chain_traverse(self, strategy, *, access_scope=None):
            return {"strategy": strategy.get("strategy", "ppr_chain"), "hops": 1, "visited": ["OpenAI", "Sam Altman"]}

        def metadata(self):
            return GraphAdapterMetadata(adapter_name="dummy", graph_type="dummy", version="0")

    executor = GraphTraversalExecutor(_Adapter(), settings=GraphTraversalSettings(chain_depth=1))
    ctx = GraphQueryContext(adapter_name="dummy", question="Who founded OpenAI?")
    step = PlanSpec(step_id="plan_01", description="OpenAI founders", channel="graph", metadata={})

    _, _, evidences = await executor.run_step(step, ctx)
    assert evidences
    triples = (evidences[0].provenance or {}).get("triples") or []
    assert triples and triples[0]["head"] == "OpenAI" and triples[0]["tail"] == "Sam Altman"

    tool = EvidenceCrosscheckTool(llm_connector=None)
    request = ToolRunRequest(
        question="Who founded OpenAI?",
        plan_step="plan_verify",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.diagnostics["triple_count"] == 1
    assert result.diagnostics["confirmed"] == 1
    assert result.diagnostics["missing"] == 0


@pytest.mark.asyncio
async def test_traversal_executor_emits_raw_chunks_not_summary():
    class _Adapter:
        async def prepare(self, question: str, *, access_scope=None):
            return None

        async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None):
            return {
                "chunks": [{"id": "kb_chunk_1", "content": "RAW_CHUNK", "metadata": {}}],
                "nodes": [],
                "edges": [],
                "metadata": {"adapter": "dummy"},
            }

        async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
            return data

        async def summarize(self, channel: str, data, *, access_scope=None) -> str:
            return "SUMMARY_ONLY"

        async def chain_traverse(self, strategy, *, access_scope=None):
            return {"strategy": strategy.get("strategy", "ppr_chain"), "hops": 1, "visited": []}

        def metadata(self):
            return GraphAdapterMetadata(adapter_name="dummy", graph_type="dummy", version="0")

    executor = GraphTraversalExecutor(_Adapter(), settings=GraphTraversalSettings(chain_depth=1))
    ctx = GraphQueryContext(adapter_name="dummy", question="Q")
    step = PlanSpec(step_id="plan_01", description="desc", channel="graph", metadata={})

    _, reasoning, evidences = await executor.run_step(step, ctx)
    assert reasoning.output_summary == "SUMMARY_ONLY"
    assert evidences and evidences[0].content == "RAW_CHUNK"
    assert evidences[0].chunk_id == "kb_chunk_1"


@pytest.mark.asyncio
async def test_context_rollup_chunk_ids_do_not_collide():
    llm = _StubLLM("first rollup")
    tool = ContextRollupTool(llm, window_size=1)
    base_request = ToolRunRequest(
        question="Q",
        plan_step="plan_01",
        context_evidences=[EvidenceChunk(chunk_id="c1", source="test", content="a")],
        adapter=None,
        access_scope=None,
        extra={},
    )
    out1 = await tool.run(base_request)
    assert out1.evidences
    first_id = out1.evidences[0].chunk_id

    tool.llm_connector = _StubLLM("second rollup")
    out2 = await tool.run(base_request)
    assert out2.evidences
    assert out2.evidences[0].chunk_id != first_id


@pytest.mark.asyncio
async def test_path_cache_returns_evidence_when_adapter_supports_strategy():
    class _Adapter:
        def metadata(self):
            return GraphAdapterMetadata(
                adapter_name="dummy",
                graph_type="dummy",
                version="0",
                capabilities=(GraphAdapterCapability(name="chain_of_exploration", modes=("ppr_prefetch",)),),
            )

        async def chain_traverse(self, strategy, *, access_scope=None):
            return {
                "strategy": strategy.get("strategy"),
                "hops": 2,
                "paths": [{"path_id": "p1", "nodes": ["A", "B", "C"], "score": 0.9}],
            }

    tool = PathCacheTool()
    req = ToolRunRequest(
        question="q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_Adapter(),
        access_scope=None,
        extra={"seed_entities": ["A", "C"]},
    )
    out = await tool.run(req)
    assert out.evidences and out.evidences[0].chunk_id == "p1"


def test_tool_hint_registry_isolated_from_global_hints(monkeypatch):
    monkeypatch.delenv("DEEPSEARCH_TOOL_HINTS", raising=False)
    clear_tool_hints()
    registry = ToolHintRegistry()

    class _CustomTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.isolated",
            channel="graph",
            description="isolated",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_isolated",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary="ok")

    LocalToolRegistry(tool_configs={}, injected_tools={"custom.isolated": _CustomTool()}, tool_hint_registry=registry)
    assert any(hint["name"] == "custom.isolated" for hint in describe_available_tools(registry=registry))
    assert not any(hint["name"] == "custom.isolated" for hint in describe_available_tools())


@pytest.mark.asyncio
async def test_graph_think_includes_graph_context_metadata():
    llm = _StubLLM(json.dumps({"reasoning": "pause", "confidence_delta": 0.4, "coverage_delta": 0.3, "next_actions": ["rerun"]}))
    tool = GraphThinkTool(llm)
    graph_context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q1",
        access_scope=GraphAccessScope(scope_id="tool-think-scope"),
    )
    request = ToolRunRequest(
        question="Q1",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
        graph_context=graph_context,
        coverage_metrics={"coverage_score": 0.2, "confidence_score": 0.3},
    )
    result = await tool.run(request)
    note = result.think_notes[0]
    assert note.reasoning == "pause"
    assert note.metadata["graph_context"]["adapter_name"] == "hipporag"
    assert result.diagnostics["graph_context"]["adapter_name"] == "hipporag"
    thought_log = result.diagnostics.get("thought_log")
    assert isinstance(thought_log, list) and thought_log
    assert thought_log[0]["reasoning_tags"]


class _RaisingLLM:
    def chat(self, messages, **kwargs):  # noqa: ANN001
        raise RuntimeError("network failure")


@pytest.mark.asyncio
async def test_graph_think_llm_failure_falls_back_to_heuristic_note():
    tool = GraphThinkTool(_RaisingLLM())
    request = ToolRunRequest(
        question="Q1",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
        graph_context=None,
        coverage_metrics={"coverage_score": 0.2, "confidence_score": 0.3},
    )
    result = await tool.run(request)
    assert result.think_notes, "LLM failures should still yield a heuristic ThinkNote"
    note = result.think_notes[0]
    assert "heuristic" in note.reasoning.lower()
    assert note.metadata.get("context_size") == 0


@pytest.mark.asyncio
async def test_llm_chain_explorer_generates_plan_and_evidence():
    llm = _StubLLM(json.dumps([{"query": "hop one", "channel": "graph", "rationale": "test"}]))
    tool = LLMChainExplorerTool(llm)
    adapter = _StubAdapter()
    request = ToolRunRequest(
        question="Explain OpenAI",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.evidences
    assert result.think_notes
    assert "hop one" in result.summary


@pytest.mark.asyncio
async def test_context_rollup_produces_summary():
    llm = _StubLLM("rollup summary")
    tool = ContextRollupTool(llm, window_size=2)
    evidences = [
        EvidenceChunk(chunk_id=f"c-{idx}", source="test", content=f"content {idx}") for idx in range(3)
    ]
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_01",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.evidences[0].content == "rollup summary"
    breakdown = result.diagnostics.get("token_breakdown")
    assert breakdown and breakdown["llm_tokens"] >= 0 and breakdown["deterministic_tokens"] > 0


@pytest.mark.asyncio
async def test_tool_manager_routes_to_mcp_with_arguments(monkeypatch):
    remote_payload = {
        "summary": "remote summary",
        "evidences": [
            {"chunk_id": "remote-1", "source": "mcp", "content": "remote evidence"},
        ],
        "think_notes": [
            {"plan_step_id": "plan_01", "reasoning": "remote think"},
        ],
        "diagnostics": {"remote": True},
    }
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="graph.think",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="remote summary",
        latency_ms=10,
        extra={"transport": "stdio"},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"tool_name": tool_name, "arguments": arguments, "graph_context": graph_context})
            return MCPToolCallOutcome(result=call_result, log=log)

    mcp_client = _StubMCPClient()
    manager = DeepSearchToolManager(
        tool_configs={
            "enable_builtin_tools": True,
            "enabled_tools": {"graph.think": {"enabled": False}},
            "remote_argument_templates": {"graph.think": {"custom_field": "$question"}},
        },
        telemetry_client=None,
        mcp_client=mcp_client,
    )
    graph_context = GraphQueryContext(adapter_name="hipporag", question="hello")
    result = await manager.invoke(
        "graph.think",
        payload={"question": "hello", "context_evidences": [], "graph_context": graph_context},
    )
    assert result.summary == "remote summary"
    assert result.evidences and result.evidences[0].chunk_id == "remote-1"
    assert result.think_notes
    assert mcp_client.calls[0]["arguments"]["question"] == "hello"
    assert mcp_client.calls[0]["arguments"]["custom_field"] == "hello"


@pytest.mark.asyncio
async def test_tool_manager_serializes_complex_payloads_for_remote():
    remote_payload = {"summary": "complex summary"}
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="graph.think",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="complex summary",
        latency_ms=15,
        extra={},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"arguments": arguments})
            return MCPToolCallOutcome(result=call_result, log=log)

    meta = GraphAdapterMetadata(
        adapter_name="hipporag",
        graph_type="hipporag",
        version="1.0",
        capabilities=(GraphAdapterCapability(name="test"),),
    )
    context_evidences = [
        EvidenceChunk(chunk_id="local-1", source="graph", content="cached evidence"),
        {"chunk_id": "dict-2", "source": "graph", "content": "dict style"},
    ]

    manager = DeepSearchToolManager(
        tool_configs={
            "enable_builtin_tools": False,
            "remote_tools": {
                "graph.think": {
                    "description": "remote think",
                    "namespace": "rag-arc.deepsearch.remote.think",
                    "profile": "H",
                    "determinism": "llm_heavy",
                    "speed": "slow",
                    "cost": "high",
                    "strategy_tags": ["think"],
                }
            },
            "max_remote_context_chars": 16,
        },
        telemetry_client=None,
        mcp_client=_StubMCPClient(),
    )

    adapter = _StubAdapter()
    adapter._metadata = meta

    await manager.invoke(
        "graph.think",
        payload={
            "question": "serialize payload",
            "context_evidences": context_evidences,
            "adapter": adapter,
            "extra": {"alternate_adapters": [meta]},
        },
    )

    arguments = manager.mcp_router.mcp_client.calls[0]["arguments"]
    context_window = arguments["context_evidences"]
    assert context_window[0]["chunk_id"] == "local-1"
    assert context_window[1]["chunk_id"] == "dict-2"
    assert set(context_window[0].keys()) == {"chunk_id", "source", "score", "content"}
    assert len(context_window[0]["content"]) <= 16
    alt_adapters = arguments["extra"]["alternate_adapters"]
    assert isinstance(alt_adapters[0], dict)


@pytest.mark.asyncio
async def test_tool_manager_routes_remote_only_descriptor():
    remote_payload = {
        "summary": "remote-only summary",
        "evidences": [],
        "diagnostics": {},
    }
    call_result = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(remote_payload))])
    log = ToolExecutionLog(
        tool_name="custom.remote",
        server_name="stub",
        arguments_snapshot={},
        response_excerpt="remote-only summary",
        latency_ms=12,
        extra={"transport": "stdio"},
    )

    class _StubMCPClient:
        def __init__(self):
            self.calls = []

        async def call_tool(self, tool_name, arguments=None, graph_context=None, server_name=None):
            self.calls.append({"tool_name": tool_name, "arguments": arguments})
            return MCPToolCallOutcome(result=call_result, log=log)

    mcp_client = _StubMCPClient()
    manager = DeepSearchToolManager(
        tool_configs={
            "enable_builtin_tools": False,
            "remote_tools": {
                "custom.remote": {
                    "description": "remote only",
                    "namespace": "rag-arc.remote.custom",
                    "profile": "H",
                    "determinism": "llm_heavy",
                    "speed": "slow",
                    "cost": "high",
                    "strategy_tags": ["remote"],
                }
            },
        },
        telemetry_client=None,
        mcp_client=mcp_client,
    )
    result = await manager.invoke(
        "custom.remote",
        payload={"question": "remote q", "context_evidences": []},
    )
    assert result.summary == "remote-only summary"
    assert mcp_client.calls


@pytest.mark.asyncio
async def test_tool_manager_falls_back_to_local_when_remote_errors():
    class _LocalTool(GraphTool):
        descriptor = ToolDescriptor(
            name="custom.echo",
            channel="graph",
            description="echo",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            return ToolResult(summary=f"local::{request.question}")

    class _FailingMCPClient:
        async def call_tool(self, *args, **kwargs):
            raise RuntimeError("mcp-down")

    class _Telemetry:
        def __init__(self):
            self.remote_logs = []

        def log_tool_invocation(self, **kwargs):
            return None

        def log_remote_tool(self, *, tool_name, log):
            self.remote_logs.append(tool_name)

    telemetry = _Telemetry()
    manager = DeepSearchToolManager(
        tool_configs={
            "enabled_tools": {"custom.echo": {"mcp_fallback": True}},
        },
        telemetry_client=telemetry,
        mcp_client=_FailingMCPClient(),
        local_tools={"custom.echo": _LocalTool()},
    )

    result = await manager.invoke("custom.echo", payload={"question": "fallback", "context_evidences": []})
    assert result.summary == "local::fallback"
    assert "remote_fallback_reason" in result.diagnostics


@pytest.mark.asyncio
async def test_cross_adapter_planner_handles_multiple_adapters():
    """Planner tool should run and emit evidences when metadata is serializable."""

    llm = _StubLLM(json.dumps({"summary": "comparison done", "actions": ["align depth"]}))
    tool = CrossAdapterPlannerTool(llm)
    adapter = _StubAdapter()

    request = ToolRunRequest(
        question="Compare adapters",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={
            "alternate_adapters": [
                {"adapter_name": "lightrag", "graph_type": "lightrag", "version": "1.0"},
            ]
        },
    )

    result = await tool.run(request)
    assert "comparison" in result.summary.lower()
    assert result.evidences
    assert result.diagnostics["adapter_count"] == 2


@pytest.mark.asyncio
async def test_parallel_think_generates_think_notes():
    llm = _StubLLM(json.dumps([{"thought": "branch one", "action": "probe"}]))
    tool = ParallelThinkTool(llm, branches=1)
    request = ToolRunRequest(
        question="Need more options",
        plan_step="plan_01",
        context_evidences=[],
        adapter=None,
        access_scope=None,
        extra={},
    )

    result = await tool.run(request)
    assert "parallel think" in result.summary.lower()
    assert result.think_notes
    assert result.diagnostics["branches"] == 1
    assert result.diagnostics.get("thought_log")


@pytest.mark.asyncio
async def test_context_rewriter_emits_rewritten_chunk():
    llm = _StubLLM("rewritten context")
    tool = ContextRewriterTool(llm, window_size=2)
    evidences = [
        EvidenceChunk(chunk_id="c1", source="test", content="original text 1"),
        EvidenceChunk(chunk_id="c2", source="test", content="original text 2"),
    ]
    request = ToolRunRequest(
        question="Summarize",
        plan_step="plan_01",
        context_evidences=evidences,
        adapter=None,
        access_scope=None,
        extra={},
    )

    result = await tool.run(request)
    assert result.evidences
    assert result.evidences[0].content == "rewritten context"


@pytest.mark.asyncio
async def test_beam_search_tool_ranks_paths_with_llm():
    llm = _StubLLM(json.dumps([{"path_id": "beam-0", "score": 0.9}, {"path_id": "beam-1", "score": 0.2}]))
    adapter = _StubAdapter()
    tool = BeamSearchTool(llm, beam_size=2, max_depth=2)
    request = ToolRunRequest(
        question="Explain OpenAI partnerships",
        plan_step="plan_beam",
        context_evidences=[],
        adapter=adapter,
        access_scope=None,
        extra={},
    )
    result = await tool.run(request)
    assert result.evidences
    assert "Beam search" in result.summary
    assert result.diagnostics["selected_paths"] == len(result.evidences)
    assert result.think_notes


def test_llm_required_tools_not_created_when_connector_missing_attrs():
    manager = DeepSearchToolManager(
        tool_configs={"enable_builtin_tools": True, "llm_connector": "fake-string"},
        telemetry_client=None,
    )
    assert manager.local_registry.resolve("graph.llm_chain_explorer") is None


@pytest.mark.asyncio
async def test_tool_manager_records_local_latency_in_diagnostics():
    class _EchoTool:
        descriptor = ToolDescriptor(
            name="graph.echo",
            channel="graph",
            description="Echo tool for telemetry tests.",
            profile="F",
            determinism="deterministic",
            namespace="rag-arc.deepsearch.tools.fast.custom_echo",
            mcp_callable=False,
        )

        async def run(self, request: ToolRunRequest) -> ToolResult:
            await asyncio.sleep(0.01)
            chunk = EvidenceChunk(chunk_id="echo-1", source="echo", content="ok")
            return ToolResult(summary="ok", evidences=[chunk], diagnostics={})

    manager = DeepSearchToolManager(
        tool_configs={"enable_builtin_tools": False},
        telemetry_client=None,
        local_tools={"graph.echo": _EchoTool()},
    )
    graph_context = GraphQueryContext(adapter_name="hipporag", question="hello")
    result = await manager.invoke(
        "graph.echo",
        payload={"question": "hello", "context_evidences": [], "graph_context": graph_context},
    )
    assert isinstance(result.diagnostics.get("latency_ms"), int)
    assert result.diagnostics.get("evidence_count") == 1
