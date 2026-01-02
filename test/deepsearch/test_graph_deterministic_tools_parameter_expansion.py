import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.fast.graph_ops_facts import GraphFactsByTypeTool
from core.deepsearch.tools.fast.graph_ops_intersection import GraphIntersectionTool
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata


class _CaptureCypherAdapter:
    def __init__(self, *, rows: list[dict]):
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
async def test_intersection_supports_left_right_direction_params() -> None:
    adapter = _CaptureCypherAdapter(
        rows=[
            {
                "target": "共同目标",
                "left_fact_ids": ["fact-l"],
                "right_fact_ids": ["fact-r"],
                "left_source_chunk_ids": [["chunk-l"]],
                "right_source_chunk_ids": [["chunk-r"]],
            }
        ]
    )
    tool = GraphIntersectionTool()
    req = ToolRunRequest(
        question="intersection direction test",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={
            "left": "A公司",
            "right": "B公司",
            "left_predicates": ["OWNS"],
            "right_predicates": ["OWNS"],
            "left_direction": "out",
            "right_direction": "in",
        },
    )
    result = await tool.run(req)
    assert result.evidences
    assert adapter.last_cypher is not None
    assert "entity_name_normalized" in adapter.last_cypher
    assert "-[lr:RELATES_TO]->" in adapter.last_cypher
    assert "<-[rr:RELATES_TO]-" in adapter.last_cypher


@pytest.mark.asyncio
async def test_facts_by_type_supports_direction_in_param() -> None:
    adapter = _CaptureCypherAdapter(
        rows=[
            {
                "head": "上游实体",
                "predicate": "DEPENDS_ON",
                "tail": "服务A",
                "fact_id": "fact-1",
                "source_chunk_ids": [["chunk-1"]],
            }
        ]
    )
    tool = GraphFactsByTypeTool()
    req = ToolRunRequest(
        question="facts_by_type direction test",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity_type": "Service", "predicates": ["DEPENDS_ON"], "direction": "in", "limit": 5},
    )
    result = await tool.run(req)
    assert result.evidences
    assert adapter.last_cypher is not None
    assert "MATCH (t:Entity)-[r:RELATES_TO]->(e)" in adapter.last_cypher
    assert "RETURN t.entity_name AS head" in adapter.last_cypher

