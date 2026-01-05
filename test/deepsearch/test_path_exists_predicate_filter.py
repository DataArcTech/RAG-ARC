import pytest

from core.deepsearch.tools.fast.graph_ops_traversal import GraphPathExistsTool
from core.deepsearch.tools import ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata


class _ShortestPathTrapAdapter:
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
        if "shortestPath" in self.last_cypher:
            return []
        return [
            {
                "nodes": ["a公司", "x公司", "b公司"],
                "predicates": ["OWNS", "OWNS"],
                "fact_ids": ["f1", "f2"],
                "source_chunk_ids": [["c1"], ["c2"]],
            }
        ]

    def metadata(self):
        return GraphAdapterMetadata(
            adapter_name="stub",
            graph_type="neo4j",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )


@pytest.mark.asyncio
async def test_path_exists_does_not_use_shortestpath_to_avoid_false_negatives() -> None:
    adapter = _ShortestPathTrapAdapter()
    tool = GraphPathExistsTool()
    req = ToolRunRequest(
        question="A 是否通过 OWNS 到达 B？",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"source": "A公司", "target": "B公司", "predicates": ["OWNS"], "direction": "out", "max_hops": 5},
    )
    result = await tool.run(req)
    assert adapter.last_cypher is not None
    assert "shortestPath" not in adapter.last_cypher
    assert "ok=true" in result.summary.lower()
    assert result.evidences

