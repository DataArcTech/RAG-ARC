import asyncio
import json

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.tools.base import ToolRunRequest
from core.deepsearch.tools.hybrid.evidence_crosscheck import EvidenceCrosscheckTool
from core.graph_adapter.base import GraphAccessScope


class _StubLLM:
    def chat(self, messages, **kwargs):  # noqa: ANN001
        return json.dumps(
            {
                "supported": [],
                "unsupported": [{"triple": "Singapore American School -[ACCREDITED_BY]-> WASC", "reason": "No chunk cited."}],
                "summary": "Crosscheck completed.",
            }
        )


class _StubAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params, *, access_scope=None):  # noqa: ANN001
        # Return provenance chunk ids for the triple (simulating KG fact provenance).
        triples = (params or {}).get("triples") or []
        out = []
        for row in triples:
            out.append({"triple": row.get("triple"), "fact_id": "F1", "source_chunk_ids": ["C42", "C99"]})
        return out

    def metadata(self):
        return type("_Meta", (), {"capabilities": ()})()


def test_crosscheck_backfills_chunks_from_graph_provenance() -> None:
    tool = EvidenceCrosscheckTool(llm_connector=_StubLLM(), enable_graph_backfill=True, graph_backfill_max_chunks=2)
    req = ToolRunRequest(
        question="probe",
        plan_step="probe_01",
        extra={"triples": [{"head": "Singapore American School", "relation": "ACCREDITED_BY", "tail": "WASC"}]},
        adapter=_StubAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-x"),
        context_evidences=[EvidenceChunk(chunk_id="c1", source="stub", content="dummy chunk mentions WASC")],
    )
    out = asyncio.run(tool.run(req))
    diag = out.diagnostics or {}
    assert diag.get("graph_backfill", {}).get("filled") == 1
    assert out.evidences, "tool should emit diagnostic evidence"
    payload_text = out.evidences[0].content or ""
    assert "C42" in payload_text

