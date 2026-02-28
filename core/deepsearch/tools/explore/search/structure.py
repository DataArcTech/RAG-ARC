"""Structure (ToC) search channel for page-level locate (reranker-based)."""
from typing import Any, Dict, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.deepsearch.utils.section_tree import SectionNode, fetch_section_nodes

from ...base import ToolRunRequest
from .base import _ChannelResult, _SearchToolBase


def _build_rerank_client():
    """Lazily build a DashScope rerank client."""
    from config.encapsulation.llm.rerank.dashscope import DashScopeRerankConfig

    cfg = DashScopeRerankConfig()
    client = cfg.build()
    return client if client.is_configured() else None


class _StructureChannel:
    """Top-down ToC channel: reranker scores section titles → page-level signals."""

    async def _run_structure(
        self: _SearchToolBase,
        *,
        request: ToolRunRequest,
        query: str,
        file_id: str,
        sections: Optional[List[SectionNode]] = None,
    ) -> _ChannelResult:
        diag: Dict[str, Any] = {"query": query, "file_id": file_id}

        if sections is None:
            if request.adapter is None or request.access_scope is None:
                diag["reason"] = "adapter_or_scope_missing"
                return _ChannelResult(channel="structure", evidences=[], diagnostics=diag, summary="structure skipped")
            sections, sec_diag = await fetch_section_nodes(
                adapter=request.adapter, access_scope=request.access_scope, file_id=file_id,
            )
            diag["fetch"] = sec_diag

        scorable = [s for s in sections if s.page_start is not None and (s.title or s.path)]
        if not scorable:
            diag["reason"] = "no_scorable_sections"
            return _ChannelResult(channel="structure", evidences=[], diagnostics=diag, summary="structure: no sections")

        client = _build_rerank_client()
        if client is None:
            diag["reason"] = "rerank_api_not_configured"
            return _ChannelResult(channel="structure", evidences=[], diagnostics=diag, summary="structure skipped: no reranker")

        documents = [s.title or s.path for s in scorable]
        _structure_rerank_instruct = (
            f"User query is: {query}\n"
            "Select the document section most relevant to this query."
        )
        try:
            scored = await client.arerank(
                query=query, documents=documents, top_k=len(documents),
                instruct=_structure_rerank_instruct,
            )
        except Exception as exc:
            diag["error"] = str(exc)
            return _ChannelResult(channel="structure", evidences=[], diagnostics=diag, summary="structure: rerank failed")

        evidences: List[EvidenceChunk] = []
        for rank, (idx, score) in enumerate(scored):
            if idx < 0 or idx >= len(scorable):
                continue
            sec = scorable[idx]
            score_f = float(score)
            if score_f <= 0:
                continue
            p_end = sec.page_end if sec.page_end is not None else sec.page_start
            evidences.append(EvidenceChunk(
                chunk_id=f"structure-{sec.section_id}",
                source="structure",
                content=f"[Section] {sec.title or sec.path} (p{sec.page_start}-p{p_end})",
                kind=EVIDENCE_KIND_DERIVED,
                score=score_f,
                provenance={
                    "channel": "structure", "rank": rank,
                    "metadata": {
                        "page_start": sec.page_start, "page_end": p_end,
                        "section_id": sec.section_id, "section_title": sec.title,
                        "source_file_id": file_id,
                    },
                },
            ))

        diag["sections_scored"] = len(evidences)
        diag["model"] = getattr(client, "model_name", None)
        return _ChannelResult(
            channel="structure", evidences=evidences, diagnostics=diag,
            summary=f"structure: {len(evidences)} sections scored.",
        )
