from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from encapsulation.data_model.schema import Chunk
from application.rag_inference.cli_module import PipelineArtifacts


@dataclass
class ChunkPreview:
    """Lightweight chunk summary used by chat/pipeline outputs."""

    index: int
    chunk_id: Optional[str]
    owner_id: Optional[str]
    preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, index: int, chunk: Chunk, max_chars: int = 160) -> "ChunkPreview":
        content = (chunk.content or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."
        return cls(
            index=index,
            chunk_id=getattr(chunk, "id", None),
            owner_id=getattr(chunk, "owner_id", None),
            preview=content,
            metadata=getattr(chunk, "metadata", {}) or {},
        )


@dataclass
class PipelineSummary:
    """Structured payload used for serialization/printing."""

    owner_id: str
    original_query: str
    rewritten_query: str
    llm_response: Optional[str]
    chunk_previews: List[ChunkPreview]
    subgraph: Optional[Dict[str, Any]]
    raw_chunks: List[Dict[str, Any]]
    evidence: Optional[Dict[str, Any]] = None

    @classmethod
    def from_artifacts(
        cls,
        owner_id: str,
        artifacts: PipelineArtifacts,
        *,
        max_chunks: Optional[int] = None,
        max_chars: int = 160,
        include_evidence: bool = False,
    ) -> "PipelineSummary":
        reranked_chunks = artifacts.reranked_chunks[:max_chunks] if max_chunks else artifacts.reranked_chunks
        previews = [
            ChunkPreview.from_chunk(idx + 1, chunk, max_chars=max_chars)
            for idx, chunk in enumerate(reranked_chunks)
        ]
        raw_chunks = [
            {
                "id": getattr(chunk, "id", None),
                "owner_id": getattr(chunk, "owner_id", None),
                "content": chunk.content,
                "metadata": getattr(chunk, "metadata", {}) or {},
            }
            for chunk in reranked_chunks
        ]
        summary = cls(
            owner_id=owner_id,
            original_query=artifacts.original_query,
            rewritten_query=artifacts.rewritten_query,
            llm_response=artifacts.llm_response,
            chunk_previews=previews,
            subgraph=artifacts.subgraph_data,
            raw_chunks=raw_chunks,
        )
        if include_evidence:
            from core.presentation.evidence import build_chat_evidence

            summary.evidence = build_chat_evidence(
                artifacts.reranked_chunks,
                subgraph_data=artifacts.subgraph_data,
                subgraph_info=artifacts.subgraph_info,
                max_chunks=max_chunks,
            )
        return summary


@dataclass
class PlanStepSummary:
    """Concise summary of a DeepSearch planner step and its execution outcome."""

    step_id: str
    description: str
    channel: str
    tool: Optional[str]
    status: Optional[str]
    output_summary: Optional[str]
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class EvidencePreview:
    """Short preview of an evidence chunk surfaced during DeepSearch."""

    chunk_id: Optional[str]
    source: Optional[str]
    preview: str
    score: Optional[float] = None


@dataclass
class DeepSearchReport:
    """Structured payload mirroring DeepSearch reporting style."""

    question: str
    final_answer: str
    highlights: List[str]
    plan_steps: List[PlanStepSummary]
    top_chunks: List[EvidencePreview]
    coverage: Dict[str, Any]
    gap_decision: Optional[str]
    stage_timings: Dict[str, Any]
    graph_chain: List[str]
    evidence: Optional[Dict[str, Any]] = None

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        max_preview_chars: int = 240,
        top_chunk_limit: int = 5,
        graph_chain_builder: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
    ) -> "DeepSearchReport":
        plan_block = payload.get("plan") or {}
        plan_body = plan_block.get("plan") or {}
        question = plan_body.get("question") or (payload.get("report") or {}).get("question") or ""
        plan_steps = plan_body.get("steps") or []

        reasoning_block = payload.get("reasoning") or {}
        reasoning_steps = reasoning_block.get("reasoning_steps") or []
        reasoning_index = {
            entry.get("step_id"): entry for entry in reasoning_steps if entry.get("step_id")
        }

        def _tool_name(step: Dict[str, Any]) -> Optional[str]:
            metadata = step.get("metadata") or {}
            return step.get("tool") or metadata.get("tool")

        def _status(step_id: str) -> Dict[str, Any]:
            return reasoning_index.get(step_id, {})

        summaries: List[PlanStepSummary] = []
        for step in plan_steps:
            step_id = step.get("step_id") or ""
            execution = _status(step_id)
            summaries.append(
                PlanStepSummary(
                    step_id=step_id,
                    description=step.get("description") or "",
                    channel=step.get("channel") or "",
                    tool=_tool_name(step),
                    status=execution.get("status"),
                    output_summary=execution.get("output_summary"),
                    evidence_ids=list(execution.get("produced_evidence_ids") or execution.get("evidence_ids") or []),
                )
            )

        reasoning_evidences = reasoning_block.get("evidences") or []
        metadata_lookup: Dict[str, Dict[str, Any]] = {}
        for item in reasoning_evidences:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            meta = ((item.get("provenance") or {}).get("metadata")) or {}
            subgraph = meta.get("_subgraph_info")
            if subgraph:
                metadata_lookup[chunk_id] = subgraph

        report_block = payload.get("report") or {}
        highlights = [
            item.get("summary") or item.get("content") or ""
            for item in report_block.get("highlights") or []
            if isinstance(item, dict)
        ]
        fallback_highlights = [text for text in highlights if text]
        raw_evidences = [
            item for item in report_block.get("evidences") or [] if isinstance(item, dict)
        ]
        sorted_evidences = sorted(
            raw_evidences,
            key=lambda entry: entry.get("score") or 0.0,
            reverse=True,
        )
        top_chunks = [
            EvidencePreview(
                chunk_id=item.get("chunk_id"),
                source=item.get("source"),
                preview=_truncate_text(item.get("content") or "", max_preview_chars),
                score=item.get("score"),
            )
            for item in sorted_evidences[:top_chunk_limit]
        ]

        evidence_payload = payload.get("evidence") or {}

        best_chain: List[str] = payload.get("graph_chain") or evidence_payload.get("graph_chain") or []
        if not best_chain and graph_chain_builder:
            for candidate in sorted_evidences:
                chunk_id = candidate.get("chunk_id")
                if not chunk_id:
                    continue
                subgraph_payload = metadata_lookup.get(chunk_id)
                if not subgraph_payload:
                    continue
                chain = graph_chain_builder(subgraph_payload)
                if chain:
                    best_chain = chain
                    break

        coverage = reasoning_block.get("coverage_metrics") or {}
        gap_result = reasoning_block.get("gap_result") or {}
        stage_timings = ((payload.get("state") or {}).get("cost_telemetry") or {}).get("stage_timings") or {}

        raw_answer = (report_block.get("answer") or "").strip()
        if not raw_answer:
            raw_answer = (reasoning_block.get("final_answer") or "").strip()
        if not raw_answer:
            rollup = next(
                (
                    item
                    for item in sorted_evidences
                    if (item.get("source") or "").lower() in {"graph.context_rollup", "context_rollup"}
                ),
                None,
            )
            if rollup:
                raw_answer = (rollup.get("content") or "").strip()
        if not raw_answer and fallback_highlights:
            enumerated = [f"{idx}. {text}" for idx, text in enumerate(fallback_highlights[:5], start=1)]
            raw_answer = "Key findings from graph reasoning:\n" + "\n".join(enumerated)

        return cls(
            question=question,
            final_answer=raw_answer,
            highlights=[text for text in highlights if text],
            plan_steps=summaries,
            top_chunks=top_chunks,
            coverage=coverage,
            gap_decision=_summarize_gap_result(gap_result),
            stage_timings=stage_timings,
            graph_chain=best_chain,
            evidence=evidence_payload if evidence_payload else None,
        )


def _truncate_text(text: str, max_chars: int) -> str:
    sanitized = " ".join(text.split())
    if len(sanitized) <= max_chars:
        return sanitized
    return f"{sanitized[:max_chars].rstrip()}..."


def _summarize_gap_result(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if not result:
        return None
    reason = result.get("reason") or ""
    if result.get("should_trigger_external"):
        return f"triggered ({reason or 'gap detected'})"
    return f"skipped ({reason or 'sufficient coverage'})"
