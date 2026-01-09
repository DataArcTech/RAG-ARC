import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.reasoning import GraphReasoningLoop
from core.presentation.evidence import build_deepsearch_evidence


class _SubgraphAdapter:
    async def prepare(self, question: str, *, access_scope=None) -> None:  # pragma: no cover
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):
        return {
            "chunks": [{"content": f"chunk::{query}", "metadata": {"chunk_id": "c1"}}],
            "nodes": [{"id": "n1"}],
            "edges": [{"id": "e1"}],
            "metadata": {
                "adapter": "hipporag",
                "subgraph_info": {
                    "subgraph_nodes": ["1"],
                    "seed_entity_ids": ["EntityA"],
                    "retrieved_chunk_ids": ["c1"],
                    "node_ppr_scores": {"1": 0.9},
                },
            },
        }

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):
        return "summary"

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


@pytest.mark.asyncio
async def test_traversal_evidence_exposes_subgraph_info_for_presentation(monkeypatch):
    loop = GraphReasoningLoop(
        adapter=_SubgraphAdapter(),
        llm_connector=None,
        strategy_config={
            "strategy_name": "ppr_chain",
            "allow_semantic_channel": True,
            "chain_depth": 1,
            "parallel_branches": 1,
            "max_parallel_branches": 1,
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
        },
        tool_manager=None,
        graph_channel_tool="graph_adapter.query",
    )
    context = GraphQueryContext(
        adapter_name="hipporag",
        question="Q",
        access_scope=GraphAccessScope(scope_id="scope-evidence"),
    )
    result = await loop.run(
        "Q",
        [{"step_id": "plan_01", "description": "Inspect", "channel": "graph", "tool": "graph_adapter.query"}],
        graph_context=context,
    )

    evidence = result["evidences"][0]
    metadata = (evidence.get("provenance") or {}).get("metadata") or {}
    assert "_subgraph_info" in metadata and metadata["_subgraph_info"], "subgraph info should be attached to evidence"

    def _fake_export(subgraph_info, *, graph_store=None):
        return {
            "nodes": [
                {"id": "EntityA", "name": "EntityA", "ppr_score": 0.9},
                {"id": "EntityB", "name": "EntityB", "ppr_score": 0.8},
            ],
            "edges": [{"source": "EntityA", "target": "EntityB", "relation": "related_to", "weight": 1.0}],
            "chunks": [],
            "metadata": {},
        }

    monkeypatch.setattr("core.presentation.evidence.export_subgraph_snapshot", _fake_export)
    monkeypatch.setattr(
        "core.presentation.evidence.build_graph_chain",
        lambda subgraph_info, snapshot=None, graph_store=None: ["EntityA -[related_to]-> EntityB"],
    )

    payload = {"reasoning": result, "report": {"evidences": []}}
    evidence_payload = build_deepsearch_evidence(payload, chunk_limit=1, graph_store=object())
    assert evidence_payload["graph_chain"] == ["EntityA -[related_to]-> EntityB"]
    assert evidence_payload["triples"] == [
        {"head": "EntityA", "relation": "related_to", "tail": "EntityB", "weight": 1.0}
    ]
