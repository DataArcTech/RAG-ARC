from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import pytest

from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.tools.base import ToolRunRequest
from core.deepsearch.tools.explore.section_select import SectionSelectTool


@dataclass(frozen=True)
class _StubCandidate:
    entity_id: str
    entity_name: str
    strategy: str = "faiss"
    score: float = 0.9


@dataclass(frozen=True)
class _StubResolutionResult:
    raw: str
    normalized: str
    entity_type_hint: Optional[str]
    resolved: bool
    resolved_candidate: Optional[_StubCandidate]
    candidates: List[_StubCandidate]
    diagnostics: Mapping[str, Any]


class _StubResolver:
    async def resolve(self, *, adapter: Any, access_scope: Any, raw_entity: str, entity_type_hint: str = "") -> _StubResolutionResult:
        cand = _StubCandidate(entity_id="E1", entity_name="后备受保人")
        return _StubResolutionResult(
            raw=raw_entity,
            normalized=str(raw_entity).lower(),
            entity_type_hint=None,
            resolved=True,
            resolved_candidate=cand,
            candidates=[cand],
            diagnostics={"stub": True},
        )


class _StubAdapter:
    supports_concurrent_calls = True

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params: Mapping[str, Any] | None = None, *, access_scope: Any = None) -> list[Dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": dict(params or {}), "access_scope": access_scope})
        # Return one projected section within the file.
        return [{"section_id": "S1", "mentions": 3, "matched_entities": ["后备受保人"]}]


@pytest.mark.asyncio
async def test_graph_bridge_is_file_scoped_and_navigation_only(monkeypatch) -> None:
    # Avoid loading PageIndex retrievers in this unit test.
    monkeypatch.setattr("config.pageindex.pageindex_enabled", lambda: False)
    monkeypatch.setattr("core.deepsearch.tools.explore.section_select.build_default_entity_resolver", lambda **_kwargs: _StubResolver())

    tool = SectionSelectTool(llm_connector=object(), pageindex_retriever=None)
    adapter = _StubAdapter()
    scope = GraphAccessScope(scope_id="owner-1")
    request = ToolRunRequest(
        question="q",
        plan_step="p",
        context_evidences=[],
        adapter=adapter,
        access_scope=scope,
        extra={},
    )

    # Minimal section_map for candidate shaping.
    section_map = {"S1": {"section_id": "S1", "section_title": "t", "section_path": "p", "page_start": 1, "page_end": 2}}
    cands, diag = await tool._graph_bridge_candidates(  # type: ignore[attr-defined]
        request=request,
        file_id="FILE-1",
        section_map=section_map,
        raw_terms=["后备受保人"],
        limit_sections=5,
    )

    assert diag.get("enabled") is True
    assert adapter.calls, "bridge should query the graph to project entities back to file-scoped sections"
    call0 = adapter.calls[0]
    assert "s.source_file_id = $file_id" in call0["cypher"]
    assert call0["params"]["file_id"] == "FILE-1"
    assert cands and cands[0].section_id == "S1"
    assert cands[0].source == "graph_bridge"
    # Navigation-only summary payload includes entity names; evidence still comes from read.pages downstream.
    assert cands[0].graph_entities == ["后备受保人"]

