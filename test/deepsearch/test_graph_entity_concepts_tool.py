import pytest

from core.deepsearch.tools import GraphEntityConceptsTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata


class _StubCypherAdapter:
    def __init__(self, rows_by_mode: dict[str, list[dict]]):
        self._rows_by_mode = rows_by_mode
        self._metadata = GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type="hipporag",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )

    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ARG002
        text = str(cypher or "")
        if "MATCH (e0:Entity)" in text and "CANONICAL_OF" in text:
            return self._rows_by_mode.get("entity", [])
        if "MATCH (a:EntityAlias)" in text and "ALIAS_OF" in text:
            return self._rows_by_mode.get("term", [])
        return []

    def metadata(self):
        return self._metadata


@pytest.mark.asyncio
async def test_entity_concepts_resolves_entity() -> None:
    adapter = _StubCypherAdapter(
        rows_by_mode={
            "entity": [
                {
                    "candidate_count": 1,
                    "entity_name": "AXA",
                    "entity_type": "ORG",
                    "canonical_id": "canonical-1",
                    "canonical_key": "axa|org",
                    "canonical_name": "axa",
                    "aliases": ["AXA", "AXA Limited"],
                }
            ]
        }
    )
    tool = GraphEntityConceptsTool()
    req = ToolRunRequest(
        question="who is AXA?",
        plan_step="p1",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "AXA", "entity_type": "ORG", "limit": 10},
    )
    result = await tool.run(req)
    assert "entity_concept" in result.summary
    assert result.evidences
    assert result.diagnostics.get("result", {}).get("canonical_name") == "axa"


@pytest.mark.asyncio
async def test_entity_concepts_aborts_on_ambiguous_entity() -> None:
    adapter = _StubCypherAdapter(rows_by_mode={"entity": [{"candidate_count": 2}]})
    tool = GraphEntityConceptsTool()
    req = ToolRunRequest(
        question="who is AXA?",
        plan_step="p1",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"entity": "AXA", "limit": 10},
    )
    result = await tool.run(req)
    assert "aborted" in result.summary.lower()
    assert not result.evidences
    assert result.diagnostics.get("candidate_count") == 2


@pytest.mark.asyncio
async def test_entity_concepts_search_aliases_by_term() -> None:
    adapter = _StubCypherAdapter(
        rows_by_mode={
            "term": [
                {"canonical_id": "canonical-1", "canonical_key": "axa|org", "canonical_name": "axa", "aliases": ["AXA", "AXA Limited"]},
                {"canonical_id": "canonical-2", "canonical_key": "axa-hk|org", "canonical_name": "axa hong kong", "aliases": ["AXA Hong Kong"]},
            ]
        }
    )
    tool = GraphEntityConceptsTool()
    req = ToolRunRequest(
        question="find AXA aliases",
        plan_step="p1",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"term": "axa", "limit": 10},
    )
    result = await tool.run(req)
    assert "entity_concepts" in result.summary
    assert result.evidences
    assert result.diagnostics.get("term") == "axa"

