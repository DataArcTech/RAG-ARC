import pytest

from core.deepsearch.tools import GraphOpsTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubAdapter:
    def __init__(self):
        self.last_scope = None
        self.rows = [{"id": "n1"}]

    async def acypher(self, cypher, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        self.last_scope = access_scope
        return list(self.rows)

    def cypher_capable(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_graph_ops_passes_access_scope() -> None:
    adapter = _StubAdapter()
    tool = GraphOpsTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_01",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="tenant-1", scope_type="tenant"),
        extra={
            "mode": "cypher",
            "cypher": "MATCH (n) WHERE COALESCE(n.owner_id, $global_owner) = $owner_id RETURN n LIMIT 1",
        },
    )

    result = await tool.run(request)
    assert result.evidences
    assert adapter.last_scope is not None
    assert adapter.last_scope.scope_id == "tenant-1"


@pytest.mark.asyncio
async def test_graph_ops_rejects_missing_owner_filter() -> None:
    adapter = _StubAdapter()
    tool = GraphOpsTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_02",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"mode": "cypher", "cypher": "MATCH (n) RETURN n LIMIT 1"},
    )

    result = await tool.run(request)
    assert "owner" in result.summary.lower()
    assert result.diagnostics.get("reason") == "owner_filter_missing"


@pytest.mark.asyncio
async def test_graph_ops_rejects_write_cypher() -> None:
    adapter = _StubAdapter()
    tool = GraphOpsTool()
    request = ToolRunRequest(
        question="Q",
        plan_step="plan_03",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={
            "mode": "cypher",
            "cypher": "MATCH (n) WHERE COALESCE(n.owner_id, $global_owner) = $owner_id CREATE (m) RETURN m",
        },
    )

    with pytest.raises(ValueError, match="read-only"):
        await tool.run(request)
