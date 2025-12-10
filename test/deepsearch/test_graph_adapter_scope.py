import asyncio

import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext, PlanSpec
from encapsulation.data_model.schema import Chunk
from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.hipporag import HippoRAGGraphAdapter
from core.deepsearch.reasoning.traversal import GraphTraversalExecutor, GraphTraversalSettings


class _DummyRetriever:
    """Minimal retriever stub that records the arguments passed in by the adapter."""

    def __init__(self):
        self.calls = []

    def invoke(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        chunk = Chunk(
            content=f"Answer for {query}",
            metadata={
                "_subgraph_info": {
                    "subgraph_nodes": [],
                    "seed_entity_ids": [],
                    "retrieved_chunk_ids": [],
                    "node_ppr_scores": {},
                }
            },
        )
        return [chunk]


async def _run_executor(context: GraphQueryContext, retriever: _DummyRetriever):
    adapter = HippoRAGGraphAdapter(retriever=retriever, default_top_k=1, summary_max_chunks=1)
    executor = GraphTraversalExecutor(adapter=adapter, settings=GraphTraversalSettings(chain_depth=1))
    plan = [
        PlanSpec(
            step_id="plan_01",
            description="Inspect graph evidence",
            channel="graph",
            metadata={},
        )
    ]
    await executor.run(plan, context)


@pytest.mark.asyncio
async def test_adapter_uses_owner_scope_for_retrieval():
    retriever = _DummyRetriever()
    context = GraphQueryContext(adapter_name="hipporag", owner_id="user-123", question="foo")

    await _run_executor(context, retriever)

    assert retriever.calls, "retriever should be invoked once"
    _, kwargs = retriever.calls[0]
    assert kwargs["owner_id"] == "user-123"
    assert kwargs["return_subgraph_info"] is True


@pytest.mark.asyncio
async def test_adapter_isolated_for_multiple_owners():
    retriever = _DummyRetriever()
    adapter = HippoRAGGraphAdapter(retriever=retriever, default_top_k=1, summary_max_chunks=1)
    executor = GraphTraversalExecutor(adapter=adapter, settings=GraphTraversalSettings(chain_depth=1))
    plan = [
        PlanSpec(step_id="plan_01", description="hop one", channel="graph", metadata={}),
    ]

    ctx_a = GraphQueryContext(adapter_name="hipporag", owner_id="alice", question="Q1")
    ctx_b = GraphQueryContext(adapter_name="hipporag", owner_id="bob", question="Q2")

    await executor.run(plan, ctx_a)
    await executor.run(plan, ctx_b)

    assert len(retriever.calls) == 2
    owners = [kwargs["owner_id"] for _, kwargs in retriever.calls]
    assert owners == ["alice", "bob"]


@pytest.mark.asyncio
async def test_access_scope_decouples_from_owner_id():
    retriever = _DummyRetriever()
    scope = GraphAccessScope(scope_id="tenant-alpha", scope_type="tenant")
    context = GraphQueryContext(adapter_name="hipporag", owner_id=None, question="bar", access_scope=scope)

    await _run_executor(context, retriever)

    assert retriever.calls, "retriever should be invoked with explicit scope"
    _, kwargs = retriever.calls[0]
    assert kwargs["owner_id"] == "tenant-alpha"
