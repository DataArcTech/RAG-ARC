import types

import pytest

from core.deepsearch.tools.base import ToolRunRequest
from core.deepsearch.tools.explore import tree_tools


class _Scope:
    def __init__(self, scope_id: str) -> None:
        self.scope_id = scope_id


@pytest.mark.asyncio
async def test_tree_root_tool_cache_hit(monkeypatch):
    calls = {"fingerprint": 0, "sections": 0, "stats": 0}

    async def _fp(*_args, **_kwargs):
        calls["fingerprint"] += 1
        return "fp_test", {"mock": True}

    async def _sections(*_args, **_kwargs):
        calls["sections"] += 1
        section = types.SimpleNamespace(
            section_id="s1",
            parent_id=None,
            title="Root",
            path="Root",
            page_start=1,
            page_end=2,
            node_types={},
        )
        return [section], {"mock": "sections"}

    async def _stats(*_args, **_kwargs):
        calls["stats"] += 1
        return {"s1": {"token_count": 10, "node_count": 1}}, {"mock": "stats"}

    monkeypatch.setattr(tree_tools, "fetch_section_tree_fingerprint", _fp)
    monkeypatch.setattr(tree_tools, "fetch_section_nodes", _sections)
    monkeypatch.setattr(tree_tools, "fetch_section_tree_stats", _stats)

    tool = tree_tools.TreeRootTool(cache_enabled=True, cache_max_entries=8, cache_ttl_seconds=60)
    req = ToolRunRequest(
        question="q",
        plan_step="p",
        context_evidences=[],
        adapter=None,
        access_scope=_Scope("owner1"),
        extra={"file_id": "00000000-0000-0000-0000-000000000001", "max_roots": 5},
    )

    out1 = await tool.run(req)
    assert out1.diagnostics.get("cache", {}).get("hit") is False
    assert calls["sections"] == 1

    out2 = await tool.run(req)
    assert out2.diagnostics.get("cache", {}).get("hit") is True
    assert calls["sections"] == 1  # no second fetch_section_nodes call


@pytest.mark.asyncio
async def test_tree_children_tool_cache_hit(monkeypatch):
    calls = {"children": 0}

    async def _children(*_args, **_kwargs):
        calls["children"] += 1
        node = types.SimpleNamespace(
            file_id="00000000-0000-0000-0000-000000000001",
            node_id="n1",
            node_type="paragraph",
            summary="S",
            page_start=3,
            page_end=3,
        )
        return [node], {"mock": "children"}

    monkeypatch.setattr(tree_tools, "fetch_tree_children", _children)

    tool = tree_tools.TreeChildrenTool(cache_enabled=True, cache_max_entries=8, cache_ttl_seconds=60)
    req = ToolRunRequest(
        question="q",
        plan_step="p",
        context_evidences=[],
        adapter=None,
        access_scope=_Scope("owner1"),
        extra={"node_id": "n1", "max_nodes": 10},
    )

    out1 = await tool.run(req)
    assert out1.diagnostics.get("cache", {}).get("hit") is False
    assert calls["children"] == 1

    out2 = await tool.run(req)
    assert out2.diagnostics.get("cache", {}).get("hit") is True
    assert calls["children"] == 1

