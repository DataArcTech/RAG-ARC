import pytest

from core.deepsearch.tools.fast.graph_ops import GraphSchemaNodesTool
from core.deepsearch.tools import ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata


class _CaptureSchemaAdapter:
    def __init__(self, rows):
        self.rows = rows
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
        return list(self.rows)

    def metadata(self):
        return GraphAdapterMetadata(
            adapter_name="stub",
            graph_type="neo4j",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )


@pytest.mark.asyncio
async def test_schema_nodes_query_by_chunk_id() -> None:
    adapter = _CaptureSchemaAdapter(
        rows=[
            {"schema_id": "schema-1", "layer": "concept", "text": "免赔额", "text_normalized": "免赔额", "level": "1.1"},
        ]
    )
    tool = GraphSchemaNodesTool()
    req = ToolRunRequest(
        question="show schema nodes",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"chunk_id": "chunk_001", "layer": "concept", "limit": 10},
    )
    result = await tool.run(req)
    assert adapter.last_cypher is not None
    assert "HAS_SCHEMA_NODE" in adapter.last_cypher
    assert result.evidences


@pytest.mark.asyncio
async def test_schema_nodes_query_by_term() -> None:
    adapter = _CaptureSchemaAdapter(
        rows=[
            {"schema_id": "schema-2", "layer": "process", "text": "生效期", "text_normalized": "生效期", "level": None},
        ]
    )
    tool = GraphSchemaNodesTool()
    req = ToolRunRequest(
        question="search schema nodes",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"term": "生效", "layer": "process", "limit": 10},
    )
    result = await tool.run(req)
    assert adapter.last_cypher is not None
    assert "CONTAINS" in adapter.last_cypher
    assert result.evidences
