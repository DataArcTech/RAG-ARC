import asyncio

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.tools import GraphOpsTool
from core.deepsearch.tools.base import ToolRunRequest
from core.graph_adapter.base import GraphAccessScope


class _StubAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def cypher_capable(self) -> bool:  # adapter_supports_cypher() fast-path
        return True

    async def acypher(self, cypher: str, params, *, access_scope=None):
        self.calls.append((str(cypher), dict(params or {})))
        text = str(cypher)
        if "AND e0.entity_name_normalized = $entity" in text:
            # Entity does not exist -> candidate_count=0
            return [{"candidate_count": 0, "neighbor": None, "predicate": None, "fact_id": None, "source_chunk_ids": None}]
        if "UNWIND $tokens AS tok" in text:
            return [
                {
                    "entity_id": "E1",
                    "entity_name": "Singapore American School",
                    "entity_name_normalized": "singapore american school",
                    "entity_type": "组织",
                    "entity_type_key": "org",
                    "hit_count": 3,
                    "edge_count": 2,
                }
            ]
        if "AND e.entity_id = $entity_id" in text:
            return [
                {"neighbor": "WASC", "predicate": "ACCREDITED_BY", "fact_id": "F1", "source_chunk_ids": ["C1"]},
                {"neighbor": "AP", "predicate": "OFFERS", "fact_id": "F2", "source_chunk_ids": ["C2"]},
            ]
        return []

    def metadata(self):
        # Minimal metadata to satisfy adapter lock helper.
        return type("_Meta", (), {"capabilities": ()})()


def test_graph_neighbors_auto_resolves_messy_entity_name() -> None:
    adapter = _StubAdapter()
    tool = GraphOpsTool()
    req = ToolRunRequest(
        question="probe",
        plan_step="probe_01",
        extra={
            "mode": "template",
            "template": "neighbors",
            "template_args": {"entity": "Singapore American School (SAS)", "direction": "both", "limit": 10},
        },
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-x"),
        context_evidences=[],
    )
    out = asyncio.run(tool.run(req))
    diag = out.diagnostics or {}
    assert diag.get("resolved") is True
    assert diag.get("resolved_entity", {}).get("entity_id") == "E1"
    assert len(diag.get("neighbors") or []) == 2
    assert out.evidences and isinstance(out.evidences[0], EvidenceChunk)


def test_graph_neighbors_resolution_can_be_forced_off_by_threshold() -> None:
    adapter = _StubAdapter()
    tool = GraphOpsTool()
    req = ToolRunRequest(
        question="probe",
        plan_step="probe_01",
        extra={
            "mode": "template",
            "template": "neighbors",
            "template_args": {
                "entity": "Singapore American School (SAS)",
                "direction": "both",
                "limit": 10,
                "resolution": {"auto_score_min": 0.99},
            },
        },
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-x"),
        context_evidences=[],
    )
    out = asyncio.run(tool.run(req))
    diag = out.diagnostics or {}
    assert diag.get("resolved") is False
    assert diag.get("resolution_candidates")
