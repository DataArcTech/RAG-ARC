"""Hybrid tool that cross-validates chunks and triples."""
import json
from typing import Any, Dict, Iterable, List, Tuple

from pydantic import BaseModel, Field

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DIAGNOSTIC

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, build_input_schema, safe_json_loads
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER
from core.prompts.deepsearch import EVIDENCE_CROSSCHECK_PROMPT
from core.deepsearch.utils.evidence_ids import derived_chunk_id


class _SupportedEntry(BaseModel):
    triple: str = Field(..., min_length=1)
    chunks: List[str] = Field(...)
    reason: str = Field(..., min_length=1)


class _UnsupportedEntry(BaseModel):
    triple: str = Field(..., min_length=1)
    chunks: List[str] | None = Field(default=None)
    reason: str = Field(..., min_length=1)


class _CrosscheckResponse(BaseModel):
    supported: List[_SupportedEntry] = Field(...)
    unsupported: List[_UnsupportedEntry] = Field(...)
    summary: str = Field(..., min_length=1)


class EvidenceCrosscheckTool(GraphTool):
    """Verifies chunk↔triple consistency before assembling a report."""

    descriptor = ToolDescriptor(
        name="graph.evidence_crosscheck",
        channel="text",
        description=(
            "Cross-validates chunk/text-channel evidence against graph triples before reporting. "
            "Evidence: derived verdicts (NOT citeable; cite underlying chunks + deterministic graph tools). "
            "Requires triples (request.extra.triples or evidence provenance) and non-empty context_evidences."
        ),
        speed="medium",
        cost="medium",
        strategy_tags=("verification", "triple", "chunk", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
        profile="X",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.x.evidence_crosscheck",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "head": {"type": "string"},
                            "relation": {"type": "string"},
                            "tail": {"type": "string"},
                        },
                        "required": ["head", "relation", "tail"],
                    },
                    "description": "Optional triples supplied by planner to validate chunk evidence.",
                }
            }
        ),
        example_args={
            "question": "Who founded OpenAI and when?",
            "plan_step": "plan_02",
            "context_evidences": [
                {"chunk_id": "c1", "source": "hipporag", "content": "OpenAI was founded by Sam Altman."}
            ],
            "extra": {
                "triples": [
                    {"head": "OpenAI", "relation": "founded_by", "tail": "Sam Altman"},
                ]
            },
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        prompt_template: str = EVIDENCE_CROSSCHECK_PROMPT,
    ):
        if llm_connector is None:
            raise ValueError("EvidenceCrosscheckTool requires an LLM connector (no heuristic fallback).")
        self.llm_connector = llm_connector
        self.prompt_template = prompt_template

    async def run(self, request: ToolRunRequest) -> ToolResult:
        triples = self._collect_triples(request)
        if not triples:
            note = ThinkNote(
                plan_step_id=request.plan_step,
                reasoning="Skipped evidence crosscheck because no triples were supplied; proceed without crosscheck.",
                next_actions=[
                    "Provide triples via request.extra.triples (or attach triples to evidence provenance).",
                    "Run graph traversal to extract candidate triples before crosschecking.",
                ],
                metadata={"reason": "missing_triples"},
            )
            return ToolResult(
                summary="Skipped evidence crosscheck (missing triples).",
                diagnostics={"skipped": True, "reason": "missing_triples", "triple_count": 0},
                think_notes=[note],
            )

        chunk_payload = self._extract_chunks(request.context_evidences)
        if not chunk_payload:
            note = ThinkNote(
                plan_step_id=request.plan_step,
                reasoning="Skipped evidence crosscheck because no non-empty context evidences were provided.",
                next_actions=[
                    "Run retrieval tools (chunk_scan/hybrid_neighborhood) to collect supporting chunks.",
                    "Ensure context_evidences entries include non-empty content.",
                ],
                metadata={"reason": "missing_context_evidences"},
            )
            return ToolResult(
                summary="Skipped evidence crosscheck (missing context evidences).",
                diagnostics={"skipped": True, "reason": "missing_context_evidences", "triple_count": len(triples)},
                think_notes=[note],
            )

        try:
            llm_report = await self._llm_crosscheck(request, chunk_payload, triples)
        except Exception as exc:  # noqa: BLE001
            note = ThinkNote(
                plan_step_id=request.plan_step,
                reasoning="Evidence crosscheck failed to produce a usable structured result; proceed without crosscheck.",
                next_actions=[
                    "Retry crosscheck with a smaller set of chunks/triples.",
                    "Verify tool prompt expects JSON and the LLM is configured correctly.",
                ],
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )
            return ToolResult(
                summary="Evidence crosscheck failed (unusable output). Proceeding without crosscheck.",
                diagnostics={
                    "skipped": True,
                    "reason": "llm_crosscheck_failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "triple_count": len(triples),
                    "token_breakdown": self._token_breakdown(chunk_payload, bool(self.llm_connector)),
                },
                think_notes=[note],
            )

        confirmed = [entry.model_dump() for entry in llm_report.supported]
        missing = [entry.model_dump() for entry in llm_report.unsupported]
        coverage_ratio = len(confirmed) / max(1, len(confirmed) + len(missing))

        diagnostics = {
            "triple_count": len(confirmed) + len(missing),
            "confirmed": len(confirmed),
            "missing": len(missing),
            "coverage_ratio": coverage_ratio,
        }
        diagnostics["token_breakdown"] = self._token_breakdown(chunk_payload, bool(self.llm_connector))
        evidences = self._build_evidence_payload(
            confirmed,
            missing,
            tool_name=self.descriptor.name,
            plan_step=request.plan_step,
        )
        summary = llm_report.summary
        think_notes = self._maybe_build_think_note(request, coverage_ratio, len(missing))

        return ToolResult(
            summary=summary,
            evidences=evidences,
            diagnostics=diagnostics,
            think_notes=think_notes,
        )

    def _collect_triples(self, request: ToolRunRequest) -> List[Dict[str, str]]:
        triples: List[Dict[str, str]] = []
        extra_triples = request.extra.get("triples")
        if isinstance(extra_triples, list):
            triples.extend(self._normalize_triples(extra_triples))

        for evidence in request.context_evidences:
            provenance = evidence.provenance or {}
            triple_payload = provenance.get("triple") or provenance.get("triples")
            if isinstance(triple_payload, list):
                triples.extend(self._normalize_triples(triple_payload))
            elif isinstance(triple_payload, dict):
                triples.extend(self._normalize_triples([triple_payload]))

        return triples

    @staticmethod
    def _normalize_triples(payload: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        triples: List[Dict[str, str]] = []
        for item in payload:
            head = item.get("head") or item.get("subject") or item.get("entity")
            tail = item.get("tail") or item.get("object") or item.get("target")
            relation = item.get("relation") or item.get("predicate") or item.get("edge")
            if head and tail and relation:
                triples.append(
                    {
                        "head": str(head),
                        "tail": str(tail),
                        "relation": str(relation),
                    }
                )
        return triples

    @staticmethod
    def _extract_chunks(evidences: List[EvidenceChunk]) -> List[Tuple[str, str]]:
        return [(ev.chunk_id, ev.content) for ev in evidences if ev.content]

    async def _llm_crosscheck(
        self,
        request: ToolRunRequest,
        chunk_payload: List[Tuple[str, str]],
        triples: List[Dict[str, str]],
    ) -> _CrosscheckResponse:
        prompt_payload = {
            "question": request.question,
            "chunks": [{"chunk_id": cid, "content": text} for cid, text in chunk_payload],
            "triples": triples,
        }
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        response = await call_llm_async(self.llm_connector, messages, temperature=0.0)
        parsed = safe_json_loads(response, expected="dict")
        if not isinstance(parsed, dict):
            raise ValueError("EvidenceCrosscheckTool returned non-JSON or non-dict output")
        return _CrosscheckResponse.model_validate(parsed)

    def _build_evidence_payload(
        self,
        confirmed: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
        *,
        tool_name: str,
        plan_step: str | None,
    ) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        if confirmed:
            content = self._format_entries("Supported triples", confirmed)
            evidences.append(
                EvidenceChunk(
                    chunk_id=derived_chunk_id(
                        tool_name=tool_name,
                        plan_step=plan_step,
                        label="confirmed",
                        content=content,
                    ),
                    source="crosscheck",
                    content=content,
                    kind=EVIDENCE_KIND_DIAGNOSTIC,
                    score=1.0,
                    provenance={"status": "confirmed", "count": len(confirmed)},
                )
            )
        if missing:
            content = self._format_entries("Unsupported triples", missing)
            evidences.append(
                EvidenceChunk(
                    chunk_id=derived_chunk_id(
                        tool_name=tool_name,
                        plan_step=plan_step,
                        label="missing",
                        content=content,
                    ),
                    source="crosscheck",
                    content=content,
                    kind=EVIDENCE_KIND_DIAGNOSTIC,
                    score=0.2,
                    provenance={"status": "missing", "count": len(missing)},
                )
            )
        return evidences

    @staticmethod
    def _format_entries(title: str, entries: List[Dict[str, Any]]) -> str:
        lines = [title + ":"]
        for item in entries:
            lines.append(
                f"- {item['triple']} | chunks: {', '.join(item.get('chunks') or []) or 'none'} | reason: {item.get('reason', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _default_summary(diagnostics: Dict[str, Any]) -> str:
        return (
            f"Evidence crosscheck processed {diagnostics['triple_count']} triples: "
            f"{diagnostics['confirmed']} confirmed, {diagnostics['missing']} unsupported."
        )

    def _maybe_build_think_note(
        self,
        request: ToolRunRequest,
        coverage_ratio: float,
        missing_count: int,
    ) -> List[ThinkNote]:
        if missing_count == 0:
            return []
        note = ThinkNote(
            plan_step_id=request.plan_step,
            reasoning="Crosscheck detected unsupported triples; rerun targeted retrieval before finalizing.",
            coverage_delta=coverage_ratio,
            next_actions=[
                "Issue focused graph probes for unsupported triples.",
                "Escalate to heavy reasoning or external search if gaps persist.",
            ],
            metadata={"missing_triples": missing_count},
        )
        return [note]

    @staticmethod
    def _token_breakdown(chunk_payload: List[Tuple[str, str]], llm_used: bool) -> Dict[str, int]:
        deterministic_tokens = sum(len(text.split()) for _, text in chunk_payload)
        llm_tokens = deterministic_tokens if llm_used else 0
        return {
            "deterministic_tokens": deterministic_tokens,
            "llm_tokens": llm_tokens,
        }
