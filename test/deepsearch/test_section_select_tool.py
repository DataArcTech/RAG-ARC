import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.section_select import SectionSelectTool
from core.graph_adapter.base import GraphAccessScope


class _FakeLLM:
    def __init__(self, responses):  # noqa: ANN001
        if isinstance(responses, (list, tuple)):
            self._responses = list(responses)
        else:
            self._responses = [responses]

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class _Hit:
    def __init__(self, content, metadata):  # noqa: ANN001
        self.content = content
        self.metadata = metadata
        self.id = metadata.get("section_id")


class _FakeRetriever:
    def __init__(self, hits):  # noqa: ANN001
        self._hits = hits

    def retrieve_sections(self, query, *, owner_id, file_ids=None):  # noqa: ANN001
        return list(self._hits)


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        if "MENTIONS" in cypher:
            return [
                {"section_id": "sec-1", "entity": "EntityA", "mentions": 3},
                {"section_id": "sec-2", "entity": "EntityB", "mentions": 2},
            ]
        if "HAS_CHUNK" in cypher:
            return [
                {"section_id": "sec-1", "metadata": {"semantic_unit_type": "image"}},
                {"section_id": "sec-2", "metadata": {"semantic_unit_type": "table"}},
                {"section_id": "sec-2", "metadata": {"semantic_unit_type": "table"}},
            ]
        return [
            {
                "section_id": "sec-1",
                "section_path": "1 总览",
                "section_title": "总览",
                "section_level": 1,
                "page_start": 1,
                "page_end": 2,
                "section_parent_id": None,
            },
            {
                "section_id": "sec-1-1",
                "section_path": "1.1 细节",
                "section_title": "细节",
                "section_level": 2,
                "page_start": 2,
                "page_end": 3,
                "section_parent_id": "sec-1",
            },
            {
                "section_id": "sec-2",
                "section_path": "2 条款",
                "section_title": "条款",
                "section_level": 1,
                "page_start": 4,
                "page_end": 5,
                "section_parent_id": None,
            },
        ]


@pytest.mark.asyncio
async def test_section_select_returns_subtree_ids() -> None:
    llm = _FakeLLM(
        [
            '{"node_list":["sec-1","sec-2"],"thinking":"overview + terms"}',
            '{"primary_section_ids":["sec-1"],"supplementary_section_ids":["sec-2"],"enough_info":true,"explanation":"need overview and terms"}',
        ]
    )
    retriever = _FakeRetriever(
        [
            _Hit("总览\n1 总览\nsummary a", {"section_id": "sec-1", "score": 0.9, "page_start": 1, "page_end": 2}),
            _Hit("条款\n2 条款\nsummary b", {"section_id": "sec-2", "score": 0.8, "page_start": 4, "page_end": 5}),
        ]
    )
    tool = SectionSelectTool(llm_connector=llm, pageindex_retriever=retriever)
    req = ToolRunRequest(
        question="介绍条款",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "top_k": 4},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert result.evidences
    selection = result.diagnostics.get("selection", {})
    assert selection.get("primary_section_ids") == ["sec-1"]
    assert "sec-1-1" in selection.get("subtree_section_ids", [])
    assert "sec-2" in selection.get("supplementary_section_ids", [])
    assert "section_node_types" in selection
    seed_entities = selection.get("seed_entities", [])
    assert "EntityA" in seed_entities
    assert "EntityB" in seed_entities


@pytest.mark.asyncio
async def test_section_select_respects_max_depth() -> None:
    llm = _FakeLLM(
        [
            '{"node_list":["sec-1","sec-2"],"thinking":"overview + terms"}',
            '{"primary_section_ids":["sec-1"],"supplementary_section_ids":["sec-2"],"enough_info":true,"explanation":"need overview and terms"}',
        ]
    )
    retriever = _FakeRetriever(
        [
            _Hit("总览\n1 总览\nsummary a", {"section_id": "sec-1", "score": 0.9, "page_start": 1, "page_end": 2}),
            _Hit("条款\n2 条款\nsummary b", {"section_id": "sec-2", "score": 0.8, "page_start": 4, "page_end": 5}),
        ]
    )
    tool = SectionSelectTool(llm_connector=llm, pageindex_retriever=retriever)
    req = ToolRunRequest(
        question="介绍条款",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "top_k": 4, "max_depth": 0},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    selection = result.diagnostics.get("selection", {})
    subtree = selection.get("subtree_section_ids", [])
    assert "sec-1" in subtree
    assert "sec-2" in subtree
    assert "sec-1-1" not in subtree
