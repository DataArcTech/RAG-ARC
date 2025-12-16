"""Hybrid tool that cross-validates chunks and triples."""
import json
from typing import Any, Dict, Iterable, List, Tuple

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, build_input_schema, safe_json_loads
from core.prompts.deepsearch import EVIDENCE_CROSSCHECK_PROMPT


class EvidenceCrosscheckTool(GraphTool):
    """Verifies chunk↔triple consistency before assembling a report."""

    descriptor = ToolDescriptor(
        name="graph.evidence_crosscheck",
        channel="text",
        description="Cross-validates chunk/text-channel evidence against graph triples before reporting.",
        speed="medium",
        cost="medium",
        strategy_tags=("verification", "triple", "chunk"),
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
        llm_connector=None,
        *,
        min_match_ratio: float = 0.75,
        prompt_template: str | None = None,
    ):
        self.llm_connector = llm_connector
        self.min_match_ratio = min_match_ratio
        self.prompt_template = prompt_template or EVIDENCE_CROSSCHECK_PROMPT

    async def run(self, request: ToolRunRequest) -> ToolResult:
        triples = self._collect_triples(request)
        if not triples:
            return ToolResult(
                summary="Evidence crosscheck skipped because no triples were provided.",
                diagnostics={"triple_count": 0},
            )

        chunk_payload = self._extract_chunks(request.context_evidences)
        if not chunk_payload:
            return ToolResult(
                summary="Evidence crosscheck skipped because no chunk context was supplied.",
                diagnostics={"triple_count": len(triples)},
            )

        heuristic_hits = self._analyze_chunks(chunk_payload, triples)
        if self.llm_connector:
            llm_report = await self._llm_crosscheck(request, chunk_payload, triples, heuristic_hits)
        else:
            llm_report = self._from_heuristic(heuristic_hits)

        confirmed, missing = llm_report["supported"], llm_report["unsupported"]
        coverage_ratio = len(confirmed) / max(1, len(confirmed) + len(missing))

        diagnostics = {
            "triple_count": len(confirmed) + len(missing),
            "confirmed": len(confirmed),
            "missing": len(missing),
            "coverage_ratio": coverage_ratio,
        }
        diagnostics["token_breakdown"] = self._token_breakdown(chunk_payload, bool(self.llm_connector))
        evidences = self._build_evidence_payload(confirmed, missing)
        summary = llm_report.get("summary") or self._default_summary(diagnostics)
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

    @staticmethod
    def _analyze_chunks(
        chunk_payload: List[Tuple[str, str]],
        triples: List[Dict[str, str]],
    ) -> List[Dict[str, object]]:
        chunk_hits: List[Dict[str, object]] = []
        for triple in triples:
            matched_chunks: List[str] = []
            for chunk_id, content in chunk_payload:
                if EvidenceCrosscheckTool._chunk_supports_triple(content, triple):
                    matched_chunks.append(chunk_id)
            chunk_hits.append({"triple": triple, "matched_chunks": matched_chunks})
        return chunk_hits

    @staticmethod
    def _chunk_supports_triple(content: str, triple: Dict[str, str]) -> bool:
        text = (content or "").lower()
        head = triple["head"].lower()
        tail = triple["tail"].lower()
        relation = triple["relation"].lower()
        if head in text and tail in text:
            return True
        relation_tokens = [tok for tok in relation.split() if tok]
        return all(token in text for token in relation_tokens[:2])

    async def _llm_crosscheck(
        self,
        request: ToolRunRequest,
        chunk_payload: List[Tuple[str, str]],
        triples: List[Dict[str, str]],
        heuristic_hits: List[Dict[str, object]],
    ) -> Dict[str, Any]:
        prompt_payload = {
            "question": request.question,
            "chunks": [{"chunk_id": cid, "content": text} for cid, text in chunk_payload],
            "triples": triples,
            "heuristic": heuristic_hits,
        }
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=0.0)
            parsed = safe_json_loads(response, expected="dict") or {}
            return {
                "supported": self._coerce_entries(parsed.get("supported")),
                "unsupported": self._coerce_entries(parsed.get("unsupported")),
                "summary": str(parsed.get("summary") or "").strip(),
            }
        except Exception:
            return self._from_heuristic(heuristic_hits)

    @staticmethod
    def _coerce_entries(payload: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "triple" in item:
                    entries.append(
                        {
                            "triple": item["triple"],
                            "chunks": item.get("chunks", []),
                            "reason": item.get("reason", ""),
                        }
                    )
        return entries

    @staticmethod
    def _from_heuristic(heuristic_hits: List[Dict[str, object]]) -> Dict[str, Any]:
        supported, unsupported = [], []
        for item in heuristic_hits:
            triple = item["triple"]
            formatted = {
                "triple": f"{triple['head']} -[{triple['relation']}]-> {triple['tail']}",
                "chunks": item["matched_chunks"],
                "reason": "keyword match" if item["matched_chunks"] else "no chunk match",
            }
            if item["matched_chunks"]:
                supported.append(formatted)
            else:
                unsupported.append(formatted)
        return {
            "supported": supported,
            "unsupported": unsupported,
            "summary": (
                f"Heuristic crosscheck confirmed {len(supported)} triples and flagged {len(unsupported)} gaps."
            ),
        }

    def _build_evidence_payload(
        self,
        confirmed: List[Dict[str, Any]],
        missing: List[Dict[str, Any]],
    ) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        if confirmed:
            content = self._format_entries("Supported triples", confirmed)
            evidences.append(
                EvidenceChunk(
                    chunk_id="crosscheck-confirmed",
                    source="crosscheck",
                    content=content,
                    score=1.0,
                    provenance={"status": "confirmed", "count": len(confirmed)},
                )
            )
        if missing:
            content = self._format_entries("Unsupported triples", missing)
            evidences.append(
                EvidenceChunk(
                    chunk_id="crosscheck-missing",
                    source="crosscheck",
                    content=content,
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
        if coverage_ratio >= self.min_match_ratio or missing_count == 0:
            return []
        delta = coverage_ratio - self.min_match_ratio
        note = ThinkNote(
            plan_step_id=request.plan_step,
            reasoning="Crosscheck detected unsupported triples; rerun targeted retrieval before finalizing.",
            confidence_delta=delta,
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
