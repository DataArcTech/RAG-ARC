import pytest

from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.hipporag import HippoRAGGraphAdapter


class _RecorderStore:
    OWNER_GLOBAL_KEY = "__GLOBAL__"

    def __init__(self) -> None:
        self.called = False

    def _owner_key(self, owner_id: str) -> str:
        return str(owner_id)

    def _execute_query(self, query: str, params=None):  # noqa: ANN001
        self.called = True
        return [{"query": str(query), "params": dict(params or {})}]


class _Retriever:
    def __init__(self) -> None:
        self.graph_store = _RecorderStore()


@pytest.mark.asyncio
async def test_acypher_rejects_write_queries() -> None:
    adapter = HippoRAGGraphAdapter(_Retriever())
    scope = GraphAccessScope(scope_id="owner-1")
    with pytest.raises(ValueError):
        await adapter.acypher("CREATE (n:Bad) RETURN n", {}, access_scope=scope)
    assert adapter.retriever.graph_store.called is False


@pytest.mark.asyncio
async def test_acypher_allows_read_only_match_queries() -> None:
    adapter = HippoRAGGraphAdapter(_Retriever())
    scope = GraphAccessScope(scope_id="owner-1")
    rows = await adapter.acypher("MATCH (n) RETURN n LIMIT 1", {}, access_scope=scope)
    assert adapter.retriever.graph_store.called is True
    assert isinstance(rows, list)

