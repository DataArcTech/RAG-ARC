import pytest

from core.graph_adapter.base import GraphAccessScope
from core.graph_adapter.hipporag import HippoRAGGraphAdapter


@pytest.mark.asyncio
async def test_chain_traverse_passes_file_scope_into_aquery_subgraph(monkeypatch) -> None:
    adapter = HippoRAGGraphAdapter(retriever=object())
    captured = {}

    async def _fake_aquery_subgraph(query, *, channel="graph", access_scope=None, query_options=None):  # noqa: ANN001
        captured["query"] = query
        captured["channel"] = channel
        captured["access_scope"] = access_scope
        captured["query_options"] = dict(query_options or {})
        # Minimal edges payload that can drive beam_search path generation.
        return {
            "nodes": [],
            "chunks": [],
            "edges": [
                {"source": "A", "target": "B", "relation": "related_to", "weight": 1.0, "directed": False},
                {"source": "B", "target": "C", "relation": "related_to", "weight": 1.0, "directed": False},
            ],
            "metadata": {},
        }

    monkeypatch.setattr(adapter, "aquery_subgraph", _fake_aquery_subgraph)

    scope = GraphAccessScope(scope_id="owner-1")
    result = await adapter.chain_traverse(
        {
            "strategy": "beam_search",
            "question": "q",
            "seed_entities": ["A", "C"],
            "beam_size": 2,
            "max_depth": 3,
            "query_options": {"file_scope": {"file_ids": ["f1"], "filename_contains": [], "source": "test"}},
        },
        access_scope=scope,
    )
    assert isinstance(result, dict)
    assert captured["query_options"].get("export_subgraph") is True
    fs = captured["query_options"].get("file_scope") or {}
    assert fs.get("file_ids") == ["f1"]

