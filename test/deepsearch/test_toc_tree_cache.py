import pytest

from core.deepsearch.tools.explore.toc_tree import TocTreeTool
from core.graph_adapter.base import GraphAccessScope
from encapsulation.data_model.deepsearch import GraphQueryContext
from core.deepsearch.tools.base import ToolRunRequest


class _DummyAdapter:
    supports_concurrent_calls = True

    def cypher_capable(self) -> bool:
        return True

    def metadata(self):
        return {"capabilities": ["cypher_query"]}

    def __init__(self):
        self.calls = []

    async def acypher(self, cypher: str, params=None, *, access_scope=None):
        self.calls.append(cypher)
        if "RETURN count(s) AS section_count" in cypher:
            return [{"section_count": 2, "max_updated_at": "2026-02-06T00:00:00Z"}]
        # section rows
        return [
            {
                "section_id": "s1",
                "section_path": "1",
                "section_title": "Intro",
                "section_level": 1,
                "page_start": 1,
                "page_end": 2,
                "section_parent_id": "",
            },
            {
                "section_id": "s2",
                "section_path": "2",
                "section_title": "Body",
                "section_level": 1,
                "page_start": 3,
                "page_end": 4,
                "section_parent_id": "",
            },
        ]


@pytest.mark.asyncio
async def test_toc_tree_uses_cache_on_repeat_calls():
    tool = TocTreeTool(cache_enabled=True, cache_max_entries=8, cache_ttl_seconds=60.0)
    adapter = _DummyAdapter()
    scope = GraphAccessScope(scope_id="owner-1")
    ctx = GraphQueryContext(adapter_name="test", owner_id="owner-1", question="Q", seed_entities=[], metadata={}, access_scope=scope)

    req = ToolRunRequest(
        question="Show ToC",
        plan_step="p1",
        context_evidences=[],
        adapter=adapter,
        access_scope=scope,
        extra={"file_id": "00000000-0000-0000-0000-000000000001", "max_depth": 2, "max_nodes": 50},
        graph_context=ctx,
        coverage_metrics={},
    )
    out1 = await tool.run(req)
    out2 = await tool.run(req)
    assert "toc.tree returned section tree" in (out1.summary or "")
    assert out2.diagnostics.get("cache", {}).get("hit") is True
    # Second call should avoid the full section-node scan query.
    section_scans = [c for c in adapter.calls if "RETURN s.section_id AS section_id" in c]
    assert len(section_scans) == 1

