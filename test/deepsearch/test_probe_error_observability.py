import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ToolResultPayload
from core.deepsearch.reasoning import MultiAgentGraphReasoningLoop
from core.graph_adapter.base import GraphAccessScope


class _StubAdapter:
    async def prepare(self, question: str, *, access_scope=None) -> None:  # pragma: no cover
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        return {"chunks": [{"content": f"chunk::{query}", "metadata": {"chunk_id": f"c::{query}"}}]}

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


class _FailingProbeToolManager:
    async def invoke(self, tool_name: str, *, payload):
        if tool_name == "search":
            raise RuntimeError("probe boom")
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


def _strategy_config() -> dict:
    return {
        "strategy_name": "ppr_chain",
        "allow_semantic_channel": True,
        "chain_depth": 1,
        "parallel_branches": 1,
        "max_parallel_branches": 2,
        "step_summary_max_chars": 2000,
        "tool_context_max_evidences": 5,
        "tool_context_max_chars": 800,
        "coverage_expected_min_chunks": 1,
        "trace_reflection_enabled": False,
        "trace_reflection_max": 0,
        "tool_timeout_seconds": 0.0,
        "think": {
            "tool_name": "graph.think",
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


def _multi_agent_settings() -> dict:
    return {
        "enabled": True,
        "max_subagents": 1,
        "subagent_concurrency": 1,
        "enable_parallel_tool_probes": True,
        "probe_tool_names": ["search"],
        "probe_concurrency": 2,
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


@pytest.mark.asyncio
async def test_probe_failures_are_recorded_in_coverage_metrics() -> None:
    loop = MultiAgentGraphReasoningLoop(
        adapter=_StubAdapter(),
        llm_connector=None,
        strategy_config=_strategy_config(),
        tool_manager=_FailingProbeToolManager(),
        settings=_multi_agent_settings(),
        graph_channel_tool="graph_adapter.query",
    )

    plan_steps = [
        {
            "step_id": "plan_01",
            "description": "Step 1",
            "channel": "graph",
            "tool": "graph_adapter.query",
            "metadata": {"scheduler": "parallel"},
        }
    ]
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-probe-errors"),
    )
    result = await loop.run("Q", plan_steps, graph_context=context)

    coverage = result.get("coverage_metrics") or {}
    assert coverage.get("probe_error_count") == 1
    errors = coverage.get("probe_errors") or []
    assert isinstance(errors, list) and errors
    assert errors[0]["tool_name"] == "search"
    assert "replay" in errors[0]
