import pytest
from types import SimpleNamespace

from core.deepsearch.tools.explore.read_structured import ReadPagesTool
from core.graph_adapter.base import GraphAccessScope


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        section_id = params.get("section_id")
        if section_id == "sec-safety":
            return [
                {
                    "chunk_id": "c2",
                    "content": "Safety warning A.",
                    "metadata": {"chunk_index": 1, "section_id": "sec-safety", "page_start": 1, "page_end": 1},
                },
                {
                    "chunk_id": "c3",
                    "content": "Safety warning B.",
                    "metadata": {"chunk_index": 2, "section_id": "sec-safety", "page_start": 2, "page_end": 2},
                },
            ]
        return [
            {
                "chunk_id": "c1",
                "content": "Intro paragraph.",
                "metadata": {"chunk_index": 0, "section_id": "sec-intro", "page_start": 0, "page_end": 0},
            },
            {
                "chunk_id": "c2",
                "content": "Safety warning A.",
                "metadata": {"chunk_index": 1, "section_id": "sec-safety", "page_start": 1, "page_end": 1},
            },
            {
                "chunk_id": "c3",
                "content": "Safety warning B.",
                "metadata": {"chunk_index": 2, "section_id": "sec-safety", "page_start": 2, "page_end": 2},
            },
            {
                "chunk_id": "c4",
                "content": "Procedure step 1.",
                "metadata": {"chunk_index": 3, "section_id": "sec-proc", "page_start": 3, "page_end": 3},
            },
        ]


class _LegacyFallbackAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        if "c.page_start IS NOT NULL" in cypher:
            return []
        return [
            {
                "chunk_id": "legacy-c2",
                "content": "Legacy page metadata only.",
                "metadata": {"chunk_index": 1, "page_start": 1, "page_end": 1},
            }
        ]


class _FakeListHeavyAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        # Build a list-heavy single page (p1) so read.pages suggests expanding to contiguous pages.
        rows = []
        for i in range(6):
            rows.append(
                {
                    "chunk_id": f"l{i}",
                    "content": f"- item {i}\n",
                    "metadata": {"chunk_index": i, "page_start": 1, "page_end": 1, "semantic_unit_type": "list"},
                }
            )
        return rows


class _FakeTableHeavyAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        file_id = params.get("file_id")
        if file_id != "11111111-1111-1111-1111-111111111111":
            return []
        # Build a table-like single page (p1) so read.pages suggests bidirectional expansion.
        rows = []
        for i in range(2):
            rows.append(
                {
                    "chunk_id": f"t{i}",
                    "content": f"| col | val {i} |\n",
                    "metadata": {"chunk_index": i, "page_start": 1, "page_end": 1, "semantic_unit_type": "table"},
                }
            )
        return rows


@pytest.mark.asyncio
async def test_read_pages_filters_by_page_range() -> None:
    tool = ReadPagesTool()
    req = SimpleNamespace(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "page_start": 1, "page_end": 2, "max_chars": 5000, "max_chunks": 20},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "read.pages returned" in result.summary
    # read.pages returns one primary evidence item per page (full-page content, no truncation).
    assert len(result.evidences) == 2
    ev_p1 = result.evidences[0]
    ev_p2 = result.evidences[1]
    assert ev_p1.source == "read.pages"
    assert ev_p2.source == "read.pages"
    assert "Safety warning A." in ev_p1.content
    assert "Safety warning B." not in ev_p1.content
    assert "Safety warning B." in ev_p2.content
    assert "Procedure step 1." not in ev_p1.content
    assert "Procedure step 1." not in ev_p2.content
    assert ev_p1.provenance.get("node_type") == "page"
    assert ev_p2.provenance.get("node_type") == "page"
    meta1 = ev_p1.provenance.get("metadata") if isinstance(ev_p1.provenance, dict) else {}
    meta2 = ev_p2.provenance.get("metadata") if isinstance(ev_p2.provenance, dict) else {}
    assert isinstance(meta1, dict)
    assert isinstance(meta2, dict)
    assert meta1.get("source_file_id") == "11111111-1111-1111-1111-111111111111"
    assert meta2.get("source_file_id") == "11111111-1111-1111-1111-111111111111"
    assert meta1.get("page_start") == 1
    assert meta2.get("page_start") == 2
    assert (result.diagnostics or {}).get("query_mode") == "indexed_page_overlap"


@pytest.mark.asyncio
async def test_read_pages_suggests_contiguous_expansion_for_list_heavy_pages() -> None:
    tool = ReadPagesTool()
    req = SimpleNamespace(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeListHeavyAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "page_start": 1, "page_end": 1, "goal": "verify continuity hints"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert result.evidences
    assert result.evidences[0].source == "read.pages"
    diag = result.diagnostics or {}
    steps = diag.get("suggested_next_steps")
    assert isinstance(steps, list)
    assert steps, "expected a continuity suggestion for list-heavy pages"
    first = steps[0]
    assert isinstance(first, dict)
    assert first.get("tool") == "read.pages"
    args = first.get("args") if isinstance(first.get("args"), dict) else {}
    # Directional expansion (non-table/equation): list-heavy page at the boundary should prefer forward expansion.
    assert args.get("page_start") == 2
    assert args.get("page_end") == 2


@pytest.mark.asyncio
async def test_read_pages_suggests_bidirectional_expansion_for_table_pages() -> None:
    tool = ReadPagesTool()
    req = SimpleNamespace(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeTableHeavyAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "page_start": 1, "page_end": 1, "goal": "verify continuity hints"},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    diag = result.diagnostics or {}
    steps = diag.get("suggested_next_steps")
    assert isinstance(steps, list)
    assert steps, "expected a continuity suggestion for table pages"
    first = steps[0]
    assert isinstance(first, dict)
    args = first.get("args") if isinstance(first.get("args"), dict) else {}
    assert args.get("page_start") == 0
    assert args.get("page_end") == 2


@pytest.mark.asyncio
async def test_read_pages_legacy_scan_fallback_when_page_props_missing() -> None:
    tool = ReadPagesTool()
    req = SimpleNamespace(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_LegacyFallbackAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"file_id": "11111111-1111-1111-1111-111111111111", "page_start": 1, "page_end": 1},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert result.evidences
    assert "Legacy page metadata only." in result.evidences[0].content
    assert (result.diagnostics or {}).get("query_mode") == "legacy_full_scan_fallback"
