import pytest

from core.deepsearch.tools import GraphOpsTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata


class _CaptureCypherAdapter:
    def __init__(self):
        self.last_cypher: str | None = None

    async def prepare(self, question: str, *, access_scope=None) -> None:  # noqa: ARG002
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):  # noqa: ARG002
        return {"chunks": [], "nodes": [], "edges": [], "metadata": {"adapter": "stub"}}

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):  # noqa: ARG002
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):  # noqa: ARG002
        return "ok"

    async def chain_traverse(self, strategy, *, access_scope=None):  # noqa: ARG002
        return {"strategy": strategy.get("strategy"), "hops": 1, "visited": []}

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ARG002
        self.last_cypher = str(cypher or "")
        return []

    def metadata(self):
        return GraphAdapterMetadata(
            adapter_name="stub",
            graph_type="neo4j",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )


@pytest.mark.asyncio
async def test_latest_truth_rejects_unsafe_time_property_for_cypher() -> None:
    adapter = _CaptureCypherAdapter()
    tool = GraphOpsTool()
    malicious = "updated_at) RETURN 1 AS injected //"
    req = ToolRunRequest(
        question="repro cypher injection",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={
            "mode": "template",
            "template": "latest_truth",
            "template_args": {"topic": "远程办公政策", "predicates": ["HAS_POLICY"], "time_property": malicious},
        },
    )
    await tool.run(req)
    assert adapter.last_cypher is not None
    assert malicious not in adapter.last_cypher
    assert "RETURN 1 AS injected" not in adapter.last_cypher
